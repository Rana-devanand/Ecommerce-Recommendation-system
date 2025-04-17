import pandas as pd
import numpy as np
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import os
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class AISalesAssistant:
    def __init__(self, product_data, nlp_search=None, image_search_extractor=None, multimodal_search=None, recommendation_system=None):
        """
        Initialize the AI Sales Assistant
        
        Parameters:
        -----------
        product_data : pandas DataFrame
            Product data with features
        nlp_search : NLPSearch, optional
            NLP search component
        image_search_extractor : ImageFeatureExtractor, optional
            Image search component
        multimodal_search : MultimodalSearch, optional
            Multimodal search component
        recommendation_system : EnhancedRecommendationSystem, optional
            Recommendation system
        """
        self.product_data = product_data
        self.nlp_search = nlp_search
        self.image_search_extractor = image_search_extractor
        self.multimodal_search = multimodal_search
        self.recommendation_system = recommendation_system
        
        print(f"AI Assistant initialized with: "
          f"nlp_search={nlp_search is not None}, "
          f"image_search={image_search_extractor is not None}, "
          f"multimodal_search={multimodal_search is not None}, "
          f"recommendation_system={recommendation_system is not None}")
        
        # Initialize conversation memory
        self.model_dir = 'models'
        self.memory_path = os.path.join(self.model_dir, 'assistant_memory.json')
        self.user_preferences = {}
        self.conversation_history = {}
        self.intent_classifier = None
        self.entity_extractor = None
        
        # Load or initialize memory
        self._load_or_create_memory()
        
        # Intent classification
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'talk to you later', 'have a good day'],
            'product_search': ['find', 'search', 'looking for', 'show me', 'need', 'want'],
            'recommendation': ['recommend', 'suggestion', 'what would you recommend', 'what should i buy'],
            'product_info': ['tell me about', 'details', 'information', 'specs', 'describe'],
            'comparison': ['compare', 'difference', 'better', 'versus', 'vs'],
            'price_query': ['price', 'cost', 'how much', 'expensive', 'cheap', 'affordable'],
            'availability': ['available', 'in stock', 'can i buy', 'when can i get'],
            'opinion': ['what do you think', 'is it good', 'should i', 'worth it', 'opinion'],
            'filter': ['filter', 'limit', 'only show', 'exclude', 'include', 'with'],
            'sort': ['sort', 'order by', 'highest', 'lowest', 'best rated', 'cheapest'],
            'help': ['help', 'how to', 'how do i', 'guide me', 'assist me', 'can you help']
        }
        
        # List of common qualifying entities
        self.entity_types = {
            'product_category': self._get_unique_categories(),
            'color': ['black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 'pink', 
                      'brown', 'orange', 'gray', 'grey', 'silver', 'gold', 'beige', 'navy', 'teal'],
            'brand': self._get_unique_brands(),
            'size': ['small', 'medium', 'large', 'xl', 'xxl', 'extra large', 'extra small', 's', 'm', 'l', 'xs'],
            'price_range': ['under', 'over', 'between', 'less than', 'more than', 'maximum', 'minimum', 'cheap', 'expensive', 'affordable', 'budget'],
            'occasion': ['casual', 'formal', 'business', 'sports', 'workout', 'party', 'wedding', 'everyday', 'outdoor', 'indoor'],
            'season': ['summer', 'winter', 'fall', 'spring', 'autumn', 'hot', 'cold', 'warm', 'cool'],
            'material': ['cotton', 'leather', 'wool', 'silk', 'polyester', 'nylon', 'denim', 'linen', 'suede', 'velvet'],
            'gender': ['men', 'women', 'unisex', 'boys', 'girls', 'male', 'female', "men's", "women's", 'kids', 'children'],
            'rating': ['top rated', 'best', 'highly rated', 'well reviewed', 'popular', 'star', 'stars']
        }
        
        # Initialize TF-IDF vectorizer for intent classification
        self._train_intent_classifier()

    def _get_unique_categories(self):
        """Extract unique product categories from the data"""
        if 'Category' not in self.product_data.columns:
            return []
        
        # Flatten category hierarchy and extract unique categories
        all_categories = []
        for cat in self.product_data['Category'].dropna().unique():
            if isinstance(cat, str):
                # Split category hierarchies (e.g., "Electronics > Computers > Laptops")
                categories = [c.strip().lower() for c in cat.split('>')]
                # Also add individual words from each category
                for category in categories:
                    all_categories.append(category)
                    words = category.split()
                    if len(words) > 1:
                        all_categories.extend(words)
        
        # Remove duplicates and return
        return list(set(all_categories))
    
    def _get_unique_brands(self):
        """Extract unique brands from the data"""
        if 'Brand' not in self.product_data.columns:
            return []
        
        return [brand.lower() for brand in self.product_data['Brand'].dropna().unique() if isinstance(brand, str)]

    def _load_or_create_memory(self):
        """Load assistant memory or create new if not exists"""
        try:
            with open(self.memory_path, 'r') as f:
                memory_data = json.load(f)
                self.user_preferences = memory_data.get('user_preferences', {})
                self.conversation_history = memory_data.get('conversation_history', {})
            print("Loaded assistant memory")
        except (FileNotFoundError, json.JSONDecodeError):
            print("Creating new assistant memory")
            self.user_preferences = {}
            self.conversation_history = {}
            self._save_memory()
    
    def _save_memory(self):
        """Save assistant memory to file"""
        os.makedirs(self.model_dir, exist_ok=True)
        memory_data = {
            'user_preferences': self.user_preferences,
            'conversation_history': self.conversation_history
        }
        with open(self.memory_path, 'w') as f:
            json.dump(memory_data, f)
    
    def _train_intent_classifier(self):
        """Train a simple intent classifier using TF-IDF and cosine similarity"""
        # Prepare training data
        intent_texts = []
        intent_labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                intent_texts.append(phrase)
                intent_labels.append(intent)
        
        # Create TF-IDF vectorizer
        self.intent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.intent_vectors = self.intent_vectorizer.fit_transform(intent_texts)
        self.intent_labels = intent_labels
    
    def _classify_intent(self, text):
        """Classify the intent of the user message"""
        text_vector = self.intent_vectorizer.transform([text.lower()])
        similarities = cosine_similarity(text_vector, self.intent_vectors)[0]
        
        if max(similarities) < 0.2:
            # If no good match, try to determine if it's a search or question
            if any(q in text.lower() for q in ['what', 'how', 'where', 'when', 'why', 'who', 'is', 'can', 'do']):
                return 'product_info'
            elif any(s in text.lower() for s in ['find', 'search', 'looking', 'want', 'need']):
                return 'product_search'
            else:
                return 'general_chat'
        
        # Return the intent with highest similarity
        max_index = similarities.argmax()
        return self.intent_labels[max_index]
    
    def _extract_entities(self, text):
        """Extract entities from the user message using spaCy"""
        entities = {}
        doc = nlp(text)
        
        # Extract product categories, colors, brands, etc.
        text_lower = text.lower()
        
        # Check for each entity type in the text
        for entity_type, entity_values in self.entity_types.items():
            for entity in entity_values:
                # Look for whole word matches using regex
                if re.search(r'\b' + re.escape(entity) + r'\b', text_lower):
                    if entity_type not in entities:
                        entities[entity_type] = []
                    if entity not in entities[entity_type]:
                        entities[entity_type].append(entity)
        
        # Extract numeric values for price ranges
        price_pattern = r'\$?\d+(\.\d+)?'
        price_matches = re.findall(price_pattern, text)
        if price_matches:
            price_values = [float(re.sub(r'[^\d.]', '', match)) for match in re.findall(price_pattern, text)]
            if 'under' in text_lower or 'less than' in text_lower or 'maximum' in text_lower:
                entities['max_price'] = min(price_values)  # Use min in case multiple values
            elif 'over' in text_lower or 'more than' in text_lower or 'minimum' in text_lower:
                entities['min_price'] = max(price_values)  # Use max in case multiple values
            elif 'between' in text_lower and len(price_values) >= 2:
                entities['min_price'] = min(price_values)
                entities['max_price'] = max(price_values)
            else:
                # Default to treating as max_price if no qualifier
                entities['max_price'] = max(price_values)
        
        # Extract spaCy entities
        for ent in doc.ents:
            if ent.label_ == "PRODUCT":
                if 'product' not in entities:
                    entities['product'] = []
                entities['product'].append(ent.text)
            elif ent.label_ == "ORG" and ent.text.lower() in self.entity_types['brand']:
                if 'brand' not in entities:
                    entities['brand'] = []
                entities['brand'].append(ent.text)
            elif ent.label_ == "MONEY":
                # Already handled by regex
                pass
        
        return entities
    
    def update_user_preferences(self, user_id, entities, intent, message=None):
        """Update user preferences based on the conversation"""
        if str(user_id) not in self.user_preferences:
            self.user_preferences[str(user_id)] = {
                'interests': {},
                'preferred_categories': {},
                'preferred_brands': {},
                'price_sensitivity': 'medium',  # default value
                'style_preferences': {},
                'last_interaction': None
            }
        
        # Update last interaction time
        self.user_preferences[str(user_id)]['last_interaction'] = datetime.datetime.now().isoformat()
        
        # Update category preferences
        if 'product_category' in entities:
            for category in entities['product_category']:
                if category not in self.user_preferences[str(user_id)]['preferred_categories']:
                    self.user_preferences[str(user_id)]['preferred_categories'][category] = 1
                else:
                    self.user_preferences[str(user_id)]['preferred_categories'][category] += 1
        
        # Update brand preferences
        if 'brand' in entities:
            for brand in entities['brand']:
                if brand not in self.user_preferences[str(user_id)]['preferred_brands']:
                    self.user_preferences[str(user_id)]['preferred_brands'][brand] = 1
                else:
                    self.user_preferences[str(user_id)]['preferred_brands'][brand] += 1
        
        # Update style preferences
        for style_attribute in ['color', 'material', 'occasion', 'season']:
            if style_attribute in entities:
                for value in entities[style_attribute]:
                    if style_attribute not in self.user_preferences[str(user_id)]['style_preferences']:
                        self.user_preferences[str(user_id)]['style_preferences'][style_attribute] = {}
                    
                    if value not in self.user_preferences[str(user_id)]['style_preferences'][style_attribute]:
                        self.user_preferences[str(user_id)]['style_preferences'][style_attribute][value] = 1
                    else:
                        self.user_preferences[str(user_id)]['style_preferences'][style_attribute][value] += 1
        
        # Update price sensitivity
        if 'max_price' in entities:
            max_price = entities['max_price']
            if max_price < 50:
                self.user_preferences[str(user_id)]['price_sensitivity'] = 'high'
            elif max_price < 200:
                self.user_preferences[str(user_id)]['price_sensitivity'] = 'medium'
            else:
                self.user_preferences[str(user_id)]['price_sensitivity'] = 'low'
        
        # Add message to conversation history
        if message:
            if str(user_id) not in self.conversation_history:
                self.conversation_history[str(user_id)] = []
            
            self.conversation_history[str(user_id)].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'message': message,
                'intent': intent,
                'entities': entities
            })
            
            # Keep only the last 20 messages
            if len(self.conversation_history[str(user_id)]) > 20:
                self.conversation_history[str(user_id)] = self.conversation_history[str(user_id)][-20:]
        
        # Save updated memory
        self._save_memory()
    
    def get_user_preferences(self, user_id):
        """Get user preferences for a user"""
        return self.user_preferences.get(str(user_id), {})
    
    def get_conversation_history(self, user_id):
        """Get conversation history for a user"""
        return self.conversation_history.get(str(user_id), [])
    
    def process_message(self, user_id, message, top_n=5):
        """
        Process user message and generate a response
        
        Parameters:
        -----------
        user_id : int
            User ID
        message : str
            User message
        top_n : int, optional
            Number of products to return
            
        Returns:
        --------
        dict
            Response with message, products, and metadata
        """
        # 1. Classify intent
        intent = self._classify_intent(message)
        
        # 2. Extract entities
        entities = self._extract_entities(message)
        
        # 3. Update user preferences
        self.update_user_preferences(user_id, entities, intent, message)
        
        # 4. Generate response based on intent
        response = self._generate_response(user_id, intent, entities, message, top_n)
        
        return response
    
    def _generate_search_query(self, entities):
        """Generate a search query from extracted entities"""
        query_parts = []
        
        # Add product category
        if 'product_category' in entities and entities['product_category']:
            query_parts.extend(entities['product_category'])
        
        # Add product
        if 'product' in entities and entities['product']:
            query_parts.extend(entities['product'])
        
        # Add brand
        if 'brand' in entities and entities['brand']:
            query_parts.extend([f"{brand} brand" for brand in entities['brand']])
        
        # Add color
        if 'color' in entities and entities['color']:
            query_parts.extend([f"{color} color" for color in entities['color']])
        
        # Add material
        if 'material' in entities and entities['material']:
            query_parts.extend([f"{material} material" for material in entities['material']])
        
        # Add occasion
        if 'occasion' in entities and entities['occasion']:
            query_parts.extend([f"for {occasion}" for occasion in entities['occasion']])
        
        # Add season
        if 'season' in entities and entities['season']:
            query_parts.extend([f"for {season}" for season in entities['season']])
        
        # Add gender
        if 'gender' in entities and entities['gender']:
            query_parts.extend([f"for {gender}" for gender in entities['gender']])
        
        # Add price constraints
        if 'max_price' in entities:
            query_parts.append(f"under ${entities['max_price']}")
        if 'min_price' in entities:
            query_parts.append(f"over ${entities['min_price']}")
        
        # Add rating constraints
        if 'rating' in entities and entities['rating']:
            query_parts.append("top rated")
        
        # Join all parts
        query = ' '.join(query_parts)
        return query
    
    def _generate_response(self, user_id, intent, entities, message, top_n=5):
        """Generate response based on intent and entities"""
        response = {
            'text': '',
            'products': [],
            'metadata': {
                'intent': intent,
                'entities': entities,
                'search_query': ''
            }
        }
        
        # Handle greeting
        if intent == 'greeting':
            user_preferences = self.get_user_preferences(user_id)
            if user_preferences and user_preferences.get('last_interaction'):
                response['text'] = f"Welcome back! How can I help with your shopping today?"
            else:
                response['text'] = "Hello! I'm your AI shopping assistant. How can I help you find products today?"
        
        # Handle farewell
        elif intent == 'farewell':
            response['text'] = "Thank you for chatting! Feel free to come back if you need more shopping assistance."
        
        # Handle help
        elif intent == 'help':
            response['text'] = "I can help you find products, provide recommendations, answer questions about products, " + \
                              "compare items, and more. Just tell me what you're looking for!"
        
        # Handle product search
        elif intent in ['product_search', 'recommendation', 'filter', 'sort']:
            search_query = self._generate_search_query(entities)
            response['metadata']['search_query'] = search_query
            
            if search_query:
                response['text'] = f"Here are some products matching your search for {search_query}:"
            else:
                # Use user preferences to suggest products if no specific search
                user_preferences = self.get_user_preferences(user_id)
                preferred_categories = user_preferences.get('preferred_categories', {})
                preferred_brands = user_preferences.get('preferred_brands', {})
                
                if preferred_categories or preferred_brands:
                    # Get top 2 preferred categories and brands
                    top_categories = sorted(preferred_categories.items(), key=lambda x: x[1], reverse=True)[:2]
                    top_brands = sorted(preferred_brands.items(), key=lambda x: x[1], reverse=True)[:2]
                    
                    preference_parts = []
                    if top_categories:
                        categories_str = " or ".join([cat for cat, _ in top_categories])
                        preference_parts.append(f"{categories_str}")
                    if top_brands:
                        brands_str = " or ".join([brand for brand, _ in top_brands])
                        preference_parts.append(f"{brands_str} brand")
                    
                    search_query = " ".join(preference_parts)
                    response['metadata']['search_query'] = search_query
                    
                    if search_query:
                        response['text'] = f"Based on your interests, here are some {search_query} products you might like:"
                    else:
                        response['text'] = "Here are some products you might be interested in:"
                else:
                    response['text'] = "Here are some popular products you might like:"
                    search_query = "popular"
                    response['metadata']['search_query'] = search_query
            
            # Search for products
            try:
                if self.nlp_search is not None:
                    # Modified to use direct search without db access
                    search_results, _ = self._safe_nlp_search(search_query, user_id, top_n)
                    response['products'] = search_results.to_dict('records') if not search_results.empty else []
                elif self.recommendation_system is not None:
                    recommended_products = self.recommendation_system.get_recommendations(user_id, top_n=top_n)
                    response['products'] = recommended_products.to_dict('records') if not recommended_products.empty else []
                else:
                    # Fallback to simple search if no search systems available
                    search_tokens = search_query.lower().split()
                    if 'search_text' in self.product_data.columns:
                        search_col = 'search_text'
                    else:
                        # Use Name and Description as fallback
                        self.product_data['search_text'] = self.product_data.apply(
                            lambda row: f"{row.get('Name', '')} {row.get('Description', '')} {row.get('Brand', '')}", 
                            axis=1
                        )
                        search_col = 'search_text'
                    
                    matches = self.product_data[
                        self.product_data[search_col].fillna('').str.lower().apply(
                            lambda text: any(token in text for token in search_tokens)
                        )
                    ]
                    
                    if not matches.empty:
                        response['products'] = matches.head(top_n).to_dict('records')
                    else:
                        response['text'] = "I couldn't find products matching your search. Could you try with different keywords?"
            except Exception as e:
                print(f"Error in product search: {e}")
                response['text'] = f"I had trouble searching for products. Please try again with different keywords."
        
        # Handle product info
        elif intent == 'product_info':
            product_name = None
            if 'product' in entities and entities['product']:
                product_name = entities['product'][0]
            elif 'product_category' in entities and entities['product_category']:
                product_name = entities['product_category'][0]
            
            if product_name:
                # Try to find the product
                product_matches = self.product_data[
                    self.product_data['Name'].str.contains(product_name, case=False, na=False)
                ]
                
                if not product_matches.empty:
                    product = product_matches.iloc[0]
                    response['text'] = f"Here's information about {product['Name']}:"
                    response['products'] = [product.to_dict()]
                else:
                    response['text'] = f"I couldn't find specific information about {product_name}. Would you like to search for similar products?"
                    search_query = product_name
                    response['metadata']['search_query'] = search_query
                    
                    # Try to find similar products
                    try:
                        if self.nlp_search is not None:
                            search_results, _ = self._safe_nlp_search(search_query, user_id, 3)
                            response['products'] = search_results.to_dict('records') if not search_results.empty else []
                    except Exception as e:
                        print(f"Error in product info search: {e}")
            else:
                response['text'] = "What product would you like information about?"
        
        # Handle price query
        elif intent == 'price_query':
            product_name = None
            if 'product' in entities and entities['product']:
                product_name = entities['product'][0]
            elif 'product_category' in entities and entities['product_category']:
                product_name = entities['product_category'][0]
            
            if product_name:
                # Try to find the product
                product_matches = self.product_data[
                    self.product_data['Name'].str.contains(product_name, case=False, na=False)
                ]
                
                if not product_matches.empty:
                    product = product_matches.iloc[0]
                    if 'Price' in product and pd.notna(product['Price']):
                        response['text'] = f"The price of {product['Name']} is ${product['Price']}."
                    else:
                        response['text'] = f"I couldn't find the price for {product['Name']}."
                    response['products'] = [product.to_dict()]
                else:
                    response['text'] = f"I couldn't find specific information about {product_name}. Would you like to search for similar products?"
                    search_query = product_name
                    response['metadata']['search_query'] = search_query
                    
                    # Try to find similar products
                    try:
                        if self.nlp_search is not None:
                            search_results, _ = self._safe_nlp_search(search_query, user_id, 3)
                            response['products'] = search_results.to_dict('records') if not search_results.empty else []
                    except Exception as e:
                        print(f"Error in price query search: {e}")
            else:
                response['text'] = "Which product's price would you like to know?"
        
        # Handle opinion
        elif intent == 'opinion':
            product_name = None
            if 'product' in entities and entities['product']:
                product_name = entities['product'][0]
            elif 'product_category' in entities and entities['product_category']:
                product_name = entities['product_category'][0]
            
            if product_name:
                # Try to find the product
                product_matches = self.product_data[
                    self.product_data['Name'].str.contains(product_name, case=False, na=False)
                ]
                
                if not product_matches.empty:
                    product = product_matches.iloc[0]
                    rating = product.get('Rating', None)
                    review_count = product.get('ReviewCount', 0)
                    
                    if rating and pd.notna(rating):
                        if rating >= 4.5:
                            opinion = "highly recommended"
                        elif rating >= 4.0:
                            opinion = "well-rated"
                        elif rating >= 3.0:
                            opinion = "decent"
                        else:
                            opinion = "mixed reviews"
                        
                        response['text'] = f"{product['Name']} is {opinion} with a rating of {rating}/5 based on {review_count} reviews."
                    else:
                        response['text'] = f"I don't have enough information to give an opinion on {product['Name']}."
                    
                    response['products'] = [product.to_dict()]
                else:
                    response['text'] = f"I couldn't find specific information about {product_name}. Would you like to search for similar products?"
                    search_query = product_name
                    response['metadata']['search_query'] = search_query
                    
                    # Try to find similar products
                    try:
                        if self.nlp_search is not None:
                            search_results, _ = self._safe_nlp_search(search_query, user_id, 3)
                            response['products'] = search_results.to_dict('records') if not search_results.empty else []
                    except Exception as e:
                        print(f"Error in opinion search: {e}")
            else:
                response['text'] = "Which product would you like my opinion on?"
        
        # Handle general chat
        else:
            response['text'] = "I'm here to help with your shopping. You can ask me to find products, get information about them, or get recommendations based on your preferences."
        
        return response
    
    def _safe_nlp_search(self, query, user_id=None, top_n=10):
        """
        A safe wrapper around NLP search that doesn't rely on database access
        """
        try:
            # Use the direct search method without database filtering
            if hasattr(self.nlp_search, 'semantic_search'):
                results = self.nlp_search.semantic_search(query, top_n=top_n*2)
                
                # Extract query info using process_query
                if hasattr(self.nlp_search, 'process_query'):
                    query_info = self.nlp_search.process_query(query)
                else:
                    query_info = {
                        'query': query,
                        'attributes': {},
                        'original_query': query
                    }
                
                # Apply attribute filtering manually if possible
                if hasattr(self.nlp_search, '_apply_attribute_filters'):
                    filtered_results = self.nlp_search._apply_attribute_filters(results, query_info['attributes'])
                    return filtered_results.sort_values('similarity', ascending=False).head(top_n), query_info
                
                return results.head(top_n), query_info
            else:
                # Fallback to basic search
                search_tokens = query.lower().split()
                if 'search_text' in self.product_data.columns:
                    search_col = 'search_text'
                else:
                    search_col = 'Name'
                
                matches = self.product_data[
                    self.product_data[search_col].fillna('').str.lower().apply(
                        lambda text: any(token in text for token in search_tokens)
                    )
                ]
                
                # Add a dummy similarity score
                if not matches.empty:
                    matches['similarity'] = 0.8
                
                query_info = {
                    'query': query,
                    'attributes': {},
                    'original_query': query
                }
                
                return matches.head(top_n), query_info
                
        except Exception as e:
            print(f"Error in safe NLP search: {e}")
            # Return empty DataFrame with required columns and query info on error
            empty_df = pd.DataFrame(columns=['Name', 'Brand', 'ImageURL', 'Rating', 'ReviewCount', 'similarity'])
            query_info = {
                'query': query,
                'attributes': {},
                'original_query': query
            }
            return empty_df, query_info