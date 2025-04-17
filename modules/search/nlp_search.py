import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import joblib
import faiss
import spacy
from thefuzz import process, fuzz
from flask_sqlalchemy import SQLAlchemy
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load spaCy model (run `python -m spacy download en_core_web_sm` once)
nlp = spacy.load("en_core_web_sm")

class NLPSearch:
    def __init__(self, product_data, db=None):
        """
        Initialize NLP search with product data and optional database for user context
        
        Parameters:
        -----------
        product_data : pandas DataFrame
            Product data with features like Name, Brand, Tags, Description, etc.
        db : SQLAlchemy database, optional
            Flask-SQLAlchemy database instance for user activity integration
        """
        self.product_data = product_data
        self.product_data['search_text'] = self.product_data.apply(self._create_search_text, axis=1)
        self.model_dir = 'models'
        self.embedding_model_path = os.path.join(self.model_dir, 'sentence_bert_model.pkl')
        self.embedding_matrix_path = os.path.join(self.model_dir, 'sentence_bert_matrix.pkl')
        self.model = None
        self.embedding_matrix = None
        self.db = db  # For user activity integration
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_or_create_embeddings()

    def _create_search_text(self, row):
        """Create a weighted text field combining all searchable product attributes"""
        weights = {
            'Name': 3.0,  # Higher weight for names
            'Brand': 2.5,
            'Tags': 2.0,
            'Description': 1.0,
            'Category': 1.5
        }
        text_parts = []
        for field, weight in weights.items():
            value = row.get(field, '') if pd.notna(row.get(field, '')) else ''
            if value:
                text_parts.append(f"{value} " * int(weight))
        return ' '.join(text_parts).strip()

    def _load_or_create_embeddings(self):
        """Load pre-computed embeddings or create new ones using Sentence-BERT"""
        try:
            self.model = joblib.load(self.embedding_model_path)
            self.embedding_matrix = joblib.load(self.embedding_matrix_path)
            print("Loaded pre-computed Sentence-BERT embeddings")
        except (FileNotFoundError, IOError):
            print("Computing new Sentence-BERT embeddings for products...")
            self._create_embeddings()

    def _create_embeddings(self):
        """Create Sentence-BERT embeddings for all products"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, efficient model
        product_texts = self.product_data['search_text'].tolist()
        self.embedding_matrix = self.model.encode(product_texts, convert_to_tensor=False).astype(np.float32)
        
        # Save embeddings
        joblib.dump(self.model, self.embedding_model_path)
        joblib.dump(self.embedding_matrix, self.embedding_matrix_path)
        print("Saved Sentence-BERT embeddings")

    def process_query(self, query):
        """
        Process natural language query to extract context, intent, and correct typos
        
        Parameters:
        -----------
        query : str
            Natural language query
            
        Returns:
        --------
        dict
            Extracted entities and intent
        """
        query = query.lower().strip()
        
        # Correct typos using fuzzy matching
        possible_corrections = process.extractOne(query, self.product_data['Name'].tolist(), scorer=fuzz.token_sort_ratio)
        if possible_corrections and possible_corrections[1] > 80:  # Threshold for confidence
            query = possible_corrections[0]

        attributes = {
            'price_range': None,
            'brand': None,
            'category': None,
            'color': None,
            'season': None,
            'gender': None,
            'size': None
        }
        
        # Use spaCy for entity recognition and attribute extraction
        doc = nlp(query)
        
        # Extract price ranges
        for ent in doc.ents:
            if ent.label_ == "MONEY" or "price" in ent.text.lower():
                price_match = re.search(r'under\s+\$?(\d+)|less\s+than\s+\$?(\d+)|\$?(\d+)\s*-\s*\$?(\d+)', ent.text)
                if price_match:
                    groups = price_match.groups()
                    if groups[0] or groups[1]:  # under X or less than X
                        max_price = int(groups[0] or groups[1])
                        attributes['price_range'] = (0, max_price)
                    elif groups[2] and groups[3]:  # X - Y range
                        min_price = int(groups[2])
                        max_price = int(groups[3])
                        attributes['price_range'] = (min_price, max_price)

        # Extract seasons, colors, and gender using spaCy tokens and entities
        seasons = ['winter', 'summer', 'fall', 'autumn', 'spring']
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'purple', 'pink', 'orange', 'brown', 'gray', 'grey']
        gender_words = ['men', 'man', "men's", 'male', 'gentleman', 'women', 'woman', "women's", 'female', 'lady', 'ladies']

        for token in doc:
            text = token.text.lower()
            if text in seasons:
                attributes['season'] = text
            elif text in colors:
                attributes['color'] = text
            elif text in gender_words:
                attributes['gender'] = text if text in ['men', 'women'] else ('men' if text in ['male', 'gentleman'] else 'women')
        
        # Extract brand and category from tokens or entities if possible
        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.text.lower() in [b.lower() for b in self.product_data['Brand'].unique() if pd.notna(b)]:
                attributes['brand'] = ent.text
            if ent.label_ == "PRODUCT" or ent.label_ == "NOUN":
                attributes['category'] = ent.text

        return {
            'query': query,
            'attributes': attributes,
            'original_query': query
        }

    def semantic_search(self, query, top_n=10, threshold=None):
        """
        Perform semantic search using Sentence-BERT embeddings and FAISS for efficiency
        
        Parameters:
        -----------
        query : str
            Natural language query
        top_n : int
            Number of results to return
        threshold : float, optional
            Similarity threshold (0-1)
            
        Returns:
        --------
        pandas DataFrame
            Matching products with similarity scores
        """
        query_vector = self.model.encode([query], convert_to_tensor=False).astype(np.float32)
        
        # Use FAISS for fast similarity search
        dimension = query_vector.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Use L2 distance (can switch to other indices for speed)
        
        # Check if embedding matrix exists and is not empty
        if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
            print("Warning: Empty embedding matrix, cannot perform search")
            return pd.DataFrame()
        
        index.add(self.embedding_matrix)
        
        # Determine safe number of results to request
        safe_top_n = min(top_n * 2, len(self.embedding_matrix))
        
        if safe_top_n == 0:
            print("Warning: No embeddings available for search")
            return pd.DataFrame()
        
        # Search for similar vectors
        _, indices = index.search(query_vector, safe_top_n)
        
        # Validate indices to ensure they are within bounds
        valid_mask = indices[0] < len(self.product_data)
        valid_indices = indices[0][valid_mask]
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid indices found for query: {query}")
            return pd.DataFrame()
        
        similarities = cosine_similarity(query_vector, self.embedding_matrix[valid_indices])[0]
        
        # Create result DataFrame
        results = pd.DataFrame({
            'index': valid_indices,
            'similarity': similarities
        })
        
        if threshold is not None:
            results = results[results['similarity'] >= threshold]
        
        results = results.sort_values('similarity', ascending=False).head(top_n)
        
        if results.empty:
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_columns = ['Name', 'Brand', 'ImageURL', 'Rating']
        available_cols = []
        for col in required_columns:
            if col in self.product_data.columns:
                available_cols.append(col)
        
        # At minimum, we need the Name column
        if 'Name' not in available_cols:
            print("Error: 'Name' column is required but not found in product data")
            return pd.DataFrame()
        
        # Get product data for the top results (safely)
        try:
            top_products = self.product_data.iloc[results['index']].copy()
            top_products['similarity'] = results['similarity'].values
            return top_products[available_cols + ['similarity']]
        except Exception as e:
            print(f"Error retrieving products: {e}")
            return pd.DataFrame()

    def _apply_attribute_filters(self, results, attributes):
        """
        Apply attribute-based filtering to search results
        
        Parameters:
        -----------
        results : pandas DataFrame
            Initial search results
        attributes : dict
            Extracted query attributes
                
        Returns:
        --------
        pandas DataFrame
            Filtered results
        """
        if results.empty:
            return results

        filtered = results.copy()
        
        # Check if season attribute exists and if the Tags column exists
        if attributes['season'] and 'Tags' in filtered.columns:
            season_terms = [attributes['season'], attributes['season'].capitalize()]
            mask = (filtered['Tags'].str.contains('|'.join(season_terms), case=False, na=False))
            
            # Also check Category and Description columns if they exist
            if 'Category' in filtered.columns:
                mask = mask | (filtered['Category'].str.contains('|'.join(season_terms), case=False, na=False))
            
            if 'Description' in filtered.columns:
                mask = mask | (filtered['Description'].str.contains('|'.join(season_terms), case=False, na=False))
                
            if mask.any():
                filtered = filtered[mask]
        
        # Check if color attribute exists and necessary columns exist
        if attributes['color']:
            color_terms = [attributes['color'], attributes['color'].capitalize()]
            
            # Initialize mask with False values
            mask = pd.Series([False] * len(filtered))
            
            # Check Tags if it exists
            if 'Tags' in filtered.columns:
                mask = mask | (filtered['Tags'].str.contains('|'.join(color_terms), case=False, na=False))
            
            # Check Description if it exists
            if 'Description' in filtered.columns:
                mask = mask | (filtered['Description'].str.contains('|'.join(color_terms), case=False, na=False))
                
            # Check Category if it exists
            if 'Category' in filtered.columns:
                mask = mask | (filtered['Category'].str.contains('|'.join(color_terms), case=False, na=False))
                
            if mask.any():
                filtered = filtered[mask]
        
        # Check if brand attribute exists and Brand column exists
        if attributes['brand']:
            brand_terms = [attributes['brand'], attributes['brand'].lower()]
            
            if 'Brand' in filtered.columns:
                mask = (filtered['Brand'].str.contains('|'.join(brand_terms), case=False, na=False))
                if mask.any():
                    filtered = filtered[mask]
        
        # Check if gender attribute exists and necessary columns exist
        if attributes['gender']:
            gender_terms = [attributes['gender'], attributes['gender'].lower()]
            
            # Initialize mask with False values
            mask = pd.Series([False] * len(filtered))
            
            # Check Tags if it exists
            if 'Tags' in filtered.columns:
                mask = mask | (filtered['Tags'].str.contains('|'.join(gender_terms), case=False, na=False))
            
            # Check Category if it exists
            if 'Category' in filtered.columns:
                mask = mask | (filtered['Category'].str.contains('|'.join(gender_terms), case=False, na=False))
                
            if mask.any():
                filtered = filtered[mask]
        
        # Check if price_range attribute exists and Price column exists
        if attributes['price_range'] and 'Price' in filtered.columns:
            min_price, max_price = attributes['price_range']
            mask = (filtered['Price'].between(min_price, max_price, inclusive='both'))
            if mask.any():
                filtered = filtered[mask]
        
        return filtered


    def enhanced_search(self, query_text, user_id=None, top_n=10):
        """
        Combine semantic search with attribute filtering and user context
        
        Parameters:
        -----------
        query_text : str
            Natural language query
        user_id : int, optional
            User ID for personalization
        top_n : int
            Number of results to return
            
        Returns:
        --------
        tuple
            (pandas DataFrame of matching products, dict of query info)
        """
        query_info = self.process_query(query_text)
        search_results = self.semantic_search(query_info['query'], top_n=top_n*2)
        
        if user_id and self.db:
            try:
                user_activities = pd.read_sql_query(
                    "SELECT product_name FROM user_activity WHERE user_id = ?",
                    self.db.session.bind, params=[user_id]
                )
                if not user_activities.empty:
                    for product in user_activities['product_name'].unique():
                        if product in search_results['Name'].values:
                            search_results.loc[search_results['Name'] == product, 'similarity'] *= 1.2  # Boost by 20%
            except Exception as e:
                print(f"Error fetching user activities: {e}")
        
        filtered_results = self._apply_attribute_filters(search_results, query_info['attributes'])
        return filtered_results.sort_values('similarity', ascending=False).head(top_n), query_info

# Example usage within app.py context (partial)
"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///ecom.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

with app.app_context():
    # Load product data
    train_data = pd.read_csv("models/clean_data.csv")
    # Initialize NLP search with database
    nlp_search = NLPSearch(train_data, db=db)
"""