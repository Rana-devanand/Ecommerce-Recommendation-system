import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import joblib
import os

class RecommendationModels:
    def __init__(self, train_data, user_activity_data=None):
        """
        Initialize recommendation models
        
        Parameters:
        -----------
        train_data : pandas DataFrame
            Product data with features like Name, Brand, Tags, etc.
        user_activity_data : pandas DataFrame, optional
            User activity data with columns user_id, product_name, timestamp, activity_type
        """
        self.train_data = train_data
        self.user_activity_data = user_activity_data
        
        # Model paths
        self.model_dir = 'models'
        self.tfidf_model_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        self.tfidf_matrix_path = os.path.join(self.model_dir, 'tfidf_matrix.pkl')
        self.svd_model_path = os.path.join(self.model_dir, 'svd_model.pkl')
        self.embeddings_path = os.path.join(self.model_dir, 'product_embeddings.pkl')
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.product_embeddings = None
        
        # Load or train models
        self._load_or_train_models()

    def _load_or_train_models(self):
        """Load pre-trained models if they exist, otherwise train new ones"""
        try:
            # Try to load pre-trained models
            self.tfidf_vectorizer = joblib.load(self.tfidf_model_path)
            self.tfidf_matrix = joblib.load(self.tfidf_matrix_path)
            self.svd_model = joblib.load(self.svd_model_path)
            self.product_embeddings = joblib.load(self.embeddings_path)
            print("Loaded pre-trained recommendation models")
        except (FileNotFoundError, IOError):
            print("Training new recommendation models...")
            self._train_models()
    
    def _train_models(self):
        """Train and save recommendation models"""
        # 1. Train TF-IDF model for content-based recommendations
        # Create a combined feature for TF-IDF
        self.train_data['combined_features'] = self.train_data['Name'].astype(str) + ' ' + \
                                          self.train_data['Brand'].astype(str) + ' ' + \
                                          self.train_data['Tags'].fillna('').astype(str)
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, 
                                              ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.train_data['combined_features'])
        
        # 2. Train SVD model for dimensionality reduction and latent factor extraction
        n_components = min(100, self.tfidf_matrix.shape[1] - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.product_embeddings = self.svd_model.fit_transform(self.tfidf_matrix)
        
        # Save models
        joblib.dump(self.tfidf_vectorizer, self.tfidf_model_path)
        joblib.dump(self.tfidf_matrix, self.tfidf_matrix_path)
        joblib.dump(self.svd_model, self.svd_model_path)
        joblib.dump(self.product_embeddings, self.embeddings_path)
        
        print(f"Models trained and saved to {self.model_dir}")

    def get_content_based_recommendations(self, item_name, top_n=10):
        """
        Get content-based recommendations using advanced TF-IDF and SVD
        
        Parameters:
        -----------
        item_name : str
            Name of the product to get recommendations for
        top_n : int, optional
            Number of recommendations to return
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with recommended products
        """
        # First try exact match
        if item_name in self.train_data['Name'].values:
            exact_match = True
            item_idx = self.train_data[self.train_data['Name'] == item_name].index[0]
        else:
            # If no exact match, try to find similar products
            exact_match = False
            # Convert search term to lowercase for better matching
            search_term = item_name.lower()
            
            # Find products containing the search term
            matching_products = self.train_data[self.train_data['Name'].str.lower().str.contains(search_term, na=False)]
            
            if matching_products.empty:
                # If still no matches, try searching in Tags
                matching_products = self.train_data[self.train_data['Tags'].str.lower().str.contains(search_term, na=False)]
                
                if matching_products.empty:
                    print(f"No products found matching '{item_name}'")
                    return pd.DataFrame()
            
            # Use the first matching product
            item_idx = matching_products.index[0]
        
        # Get the product's embedding
        item_embedding = self.product_embeddings[item_idx].reshape(1, -1)
        
        # Calculate similarity with all other products
        similarities = cosine_similarity(item_embedding, self.product_embeddings)
        
        # Get indices of most similar products
        similar_indices = similarities[0].argsort()[::-1]
        
        # If we used a partial match, include the matched item too
        start_idx = 0 if not exact_match else 1
        
        # Get recommended indices
        recommended_indices = similar_indices[start_idx:top_n+start_idx]
        
        # SAFETY CHECK: Validate indices are within bounds of DataFrame
        valid_indices = [idx for idx in recommended_indices if idx < len(self.train_data)]
        
        if not valid_indices:
            print(f"Warning: No valid indices found for recommendations for '{item_name}'")
            return pd.DataFrame()
        
        # Check which columns are available
        required_columns = ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
        available_cols = []
        for col in required_columns:
            if col in self.train_data.columns:
                available_cols.append(col)
            else:
                print(f"Warning: Required column '{col}' not found in product data")
        
        # If Name column is missing, we can't proceed
        if 'Name' not in available_cols:
            print("Error: 'Name' column is required but not found in product data")
            return pd.DataFrame()
        
        # Get the details of the recommended products using only valid indices and available columns
        recommended_items = self.train_data.iloc[valid_indices][available_cols]
        
        return recommended_items

    def get_collaborative_recommendations(self, user_id, top_n=10):
        """
        Get collaborative filtering recommendations using user interaction history
        
        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        top_n : int, optional
            Number of recommendations to return
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with recommended products
        """
        if self.user_activity_data is None:
            print("No user activity data provided")
            return pd.DataFrame()
        
        # Create user activity DataFrame for the specific user
        user_activities = self.user_activity_data[self.user_activity_data['user_id'] == user_id]
        
        if user_activities.empty:
            print(f"No activity found for user {user_id}")
            return pd.DataFrame()
        
        # Get the products the user has interacted with
        user_products = user_activities['product_name'].unique()
        
        # Calculate weights for each product based on activity type and recency
        # More recent and 'stronger' interactions get higher weights
        product_weights = {}
        
        # Define weights for different activity types
        activity_weights = {
            'view': 1.0,
            'search': 0.5,
            'purchase': 3.0,
        }
        
        # Current time for recency calculation
        max_timestamp = user_activities['timestamp'].max()
        
        # Calculate product weights
        for _, activity in user_activities.iterrows():
            product = activity['product_name']
            activity_type = activity['activity_type']
            
            # Calculate recency weight (more recent = higher weight)
            time_diff = (max_timestamp - activity['timestamp']).total_seconds() / 86400  # Convert to days
            recency_weight = 1.0 / (1.0 + time_diff)  # Decay function
            
            # Calculate total weight for this interaction
            interaction_weight = activity_weights.get(activity_type, 1.0) * recency_weight
            
            # Add to product weights
            if product in product_weights:
                product_weights[product] += interaction_weight
            else:
                product_weights[product] = interaction_weight
        
        # Get indices of the user's products in the training data
        user_product_indices = []
        for product in user_products:
            product_matches = self.train_data[self.train_data['Name'] == product]
            if not product_matches.empty:
                user_product_indices.append(product_matches.index[0])
        
        if not user_product_indices:
            print(f"None of user {user_id}'s products found in dataset")
            return pd.DataFrame()
        
        # Calculate weighted average of the user's product embeddings
        user_profile = np.zeros(self.product_embeddings.shape[1])
        total_weight = 0
        
        for idx in user_product_indices:
            product_name = self.train_data.iloc[idx]['Name']
            weight = product_weights.get(product_name, 1.0)
            user_profile += weight * self.product_embeddings[idx]
            total_weight += weight
        
        if total_weight > 0:
            user_profile /= total_weight
        
        # Calculate similarity with all products
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.product_embeddings)
        
        # Get indices of most similar products, excluding those the user has already interacted with
        all_indices = similarities[0].argsort()[::-1]
        recommended_indices = [idx for idx in all_indices 
                              if self.train_data.iloc[idx]['Name'] not in user_products][:top_n]
        
        # Get the details of the recommended products
        recommended_items = self.train_data.iloc[recommended_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
        
        return recommended_items
    
    def get_hybrid_recommendations(self, user_id, item_name=None, alpha=0.6, top_n=10):
        """
        Get hybrid recommendations combining content-based and collaborative filtering
        
        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        item_name : str, optional
            Name of the product to use for content-based recommendations
        alpha : float, optional
            Weight for content-based recommendations (1-alpha for collaborative)
        top_n : int, optional
            Number of recommendations to return
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with recommended products
        """
        # Get content-based recommendations if item_name is provided
        if item_name:
            content_recs = self.get_content_based_recommendations(item_name, top_n=top_n*2)
        else:
            content_recs = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])
        
        # Get collaborative filtering recommendations
        collab_recs = self.get_collaborative_recommendations(user_id, top_n=top_n*2)
        
        if content_recs.empty and collab_recs.empty:
            # If both recommendation approaches yielded no results, return popular products
            return self.get_popular_products(top_n)
        
        # If one approach yielded no results, return the other
        if content_recs.empty:
            return collab_recs.head(top_n)
        if collab_recs.empty:
            return content_recs.head(top_n)
        
        # Combine the two recommendation sets with weights
        # Create a set of all product names in both recommendation sets
        all_products = set(content_recs['Name']).union(set(collab_recs['Name']))
        
        # Create a dictionary to store the combined scores
        combined_scores = {}
        
        # Min-max scale for normalizing scores
        scaler = MinMaxScaler()
        
        # Score content-based recommendations
        content_scores = {}
        for i, (_, row) in enumerate(content_recs.iterrows()):
            # Inverse rank scoring - higher score for higher ranked items
            content_scores[row['Name']] = 1.0 / (i + 1)
        
        # Score collaborative recommendations
        collab_scores = {}
        for i, (_, row) in enumerate(collab_recs.iterrows()):
            # Inverse rank scoring - higher score for higher ranked items
            collab_scores[row['Name']] = 1.0 / (i + 1)
        
        # Combine scores
        for product in all_products:
            c_score = content_scores.get(product, 0)
            cf_score = collab_scores.get(product, 0)
            combined_scores[product] = alpha * c_score + (1 - alpha) * cf_score
        
        # Sort products by combined score
        sorted_products = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get the details of the top recommended products
        recommended_products = []
        for product_name, _ in sorted_products:
            # Find the product in the training data
            product_data = self.train_data[self.train_data['Name'] == product_name]
            if not product_data.empty:
                recommended_products.append(product_data.iloc[0])
        
        return pd.DataFrame(recommended_products)[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    
    def get_popular_products(self, top_n=10):
        """
        Get popular products based on review count and rating
        
        Parameters:
        -----------
        top_n : int, optional
            Number of products to return
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with popular products
        """
        # Calculate popularity score (weighted combination of review count and rating)
        # Normalize values first
        scaler = MinMaxScaler()
        
        # Create a copy to avoid modifying the original dataframe
        data_copy = self.train_data.copy()
        
        # Handle cases with few or no reviews
        data_copy['ReviewCount'] = data_copy['ReviewCount'].fillna(0)
        
        # Normalize review count and rating
        normalized_review_count = scaler.fit_transform(data_copy[['ReviewCount']])
        normalized_rating = scaler.fit_transform(data_copy[['Rating']])
        
        # Calculate popularity score (70% weight to review count, 30% to rating)
        data_copy['popularity_score'] = 0.7 * normalized_review_count.flatten() + 0.3 * normalized_rating.flatten()
        
        # Sort by popularity score and get top_n
        popular_products = data_copy.sort_values('popularity_score', ascending=False).head(top_n)
        
        return popular_products[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

# Usage example:
# rec_models = RecommendationModels(train_data, user_activity_data)
# content_recommendations = rec_models.get_content_based_recommendations("Product Name", top_n=5)
# collaborative_recommendations = rec_models.get_collaborative_recommendations(user_id=123, top_n=5)
# hybrid_recommendations = rec_models.get_hybrid_recommendations(user_id=123, item_name="Product Name", top_n=5)