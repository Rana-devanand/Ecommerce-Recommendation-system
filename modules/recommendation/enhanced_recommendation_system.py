import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modules.recommendation.recommendation_models import RecommendationModels
from modules.recommendation.neural_collaborative_filtering import NeuralCollaborativeFiltering
import os
import joblib

class EnhancedRecommendationSystem:
    def __init__(self, train_data, user_activity_data=None):
        """
        Initialize enhanced recommendation system with both traditional and deep learning models
        
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
        self.model_weights_path = os.path.join(self.model_dir, 'ensemble_weights.pkl')
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize component models
        self.traditional_model = RecommendationModels(train_data, user_activity_data)
        
        # Initialize NCF model with proper error handling
        try:
            self.ncf_model = NeuralCollaborativeFiltering(user_activity_data, train_data)
        except Exception as e:
            print(f"Error initializing Neural CF model: {e}")
            print("Will continue with traditional models only")
            self.ncf_model = None
        
        # Default ensemble weights
        self.ensemble_weights = {
            'content_based': 0.3,
            'collaborative': 0.2,
            'neural': 0.5
        }
        
        # Load saved weights if they exist
        self._load_weights()
    
    def _load_weights(self):
        """Load saved ensemble weights if they exist"""
        try:
            self.ensemble_weights = joblib.load(self.model_weights_path)
            print("Loaded ensemble weights")
        except (FileNotFoundError, IOError):
            print("Using default ensemble weights")
    
    def save_weights(self):
        """Save ensemble weights"""
        joblib.dump(self.ensemble_weights, self.model_weights_path)
        print(f"Saved ensemble weights to {self.model_weights_path}")
    
    def set_ensemble_weights(self, content_based=0.3, collaborative=0.2, neural=0.5):
        """
        Set custom weights for the ensemble model
        
        Parameters:
        -----------
        content_based : float
            Weight for content-based recommendations
        collaborative : float
            Weight for traditional collaborative filtering
        neural : float
            Weight for neural collaborative filtering
        """
        # Normalize weights to sum to 1
        total = content_based + collaborative + neural
        
        self.ensemble_weights = {
            'content_based': content_based / total,
            'collaborative': collaborative / total,
            'neural': neural / total
        }
        
        self.save_weights()
    
    def get_recommendations(self, user_id, item_name=None, top_n=10):
        """
        Get recommendations using ensemble of all recommendation models
        
        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        item_name : str, optional
            Name of the product to use for content-based recommendations
        top_n : int, optional
            Number of recommendations to return
                
        Returns:
        --------
        pandas DataFrame
            DataFrame with recommended products
        """
        recommendations = []
        all_products = set()
        product_scores = {}
        
        # 1. Get content-based recommendations
        if item_name:
            try:
                content_recs = self.traditional_model.get_content_based_recommendations(item_name, top_n=top_n*2)
                if not content_recs.empty:
                    for i, (_, row) in enumerate(content_recs.iterrows()):
                        score = 1.0 / (i + 1)  # Inverse rank scoring
                        product_name = row['Name']
                        all_products.add(product_name)
                        
                        if product_name not in product_scores:
                            product_scores[product_name] = {
                                'content_based': score,
                                'collaborative': 0.0,
                                'neural': 0.0,
                                'data': row
                            }
                        else:
                            product_scores[product_name]['content_based'] = score
            except Exception as e:
                print(f"Error getting content-based recommendations: {e}")
        
        # 2. Get traditional collaborative filtering recommendations
        if user_id is not None:
            try:
                collab_recs = self.traditional_model.get_collaborative_recommendations(user_id, top_n=top_n*2)
                if not collab_recs.empty:
                    for i, (_, row) in enumerate(collab_recs.iterrows()):
                        score = 1.0 / (i + 1)  # Inverse rank scoring
                        product_name = row['Name']
                        all_products.add(product_name)
                        
                        if product_name not in product_scores:
                            product_scores[product_name] = {
                                'content_based': 0.0,
                                'collaborative': score,
                                'neural': 0.0,
                                'data': row
                            }
                        else:
                            product_scores[product_name]['collaborative'] = score
            except Exception as e:
                print(f"Error getting collaborative recommendations: {e}")
        
        # 3. Get neural collaborative filtering recommendations
        if user_id is not None and self.ncf_model is not None:
            try:
                ncf_recs = self.ncf_model.get_recommendations(user_id, top_n=top_n*2)
                if not ncf_recs.empty:
                    for i, (_, row) in enumerate(ncf_recs.iterrows()):
                        score = 1.0 / (i + 1)  # Inverse rank scoring
                        product_name = row['Name']
                        all_products.add(product_name)
                        
                        if product_name not in product_scores:
                            product_scores[product_name] = {
                                'content_based': 0.0,
                                'collaborative': 0.0,
                                'neural': score,
                                'data': row
                            }
                        else:
                            product_scores[product_name]['neural'] = score
            except Exception as e:
                print(f"Error getting neural recommendations: {e}")
                # Adjust weights to ignore neural if it failed
                if self.ensemble_weights['neural'] > 0:
                    temp_weights = {
                        'content_based': self.ensemble_weights['content_based'] / (1 - self.ensemble_weights['neural']),
                        'collaborative': self.ensemble_weights['collaborative'] / (1 - self.ensemble_weights['neural']),
                        'neural': 0.0
                    }
                else:
                    temp_weights = self.ensemble_weights.copy()
        
        # If no recommendations from any model, return popular products
        if not product_scores:
            return self.traditional_model.get_popular_products(top_n)
        
        # Calculate combined scores using ensemble weights
        combined_scores = {}
        for product_name, scores in product_scores.items():
            combined_score = (
                self.ensemble_weights['content_based'] * scores['content_based'] +
                self.ensemble_weights['collaborative'] * scores['collaborative'] +
                self.ensemble_weights['neural'] * scores['neural']
            )
            combined_scores[product_name] = (combined_score, scores['data'])
        
        # Sort by combined score
        sorted_products = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
        
        # Create DataFrame from results
        results = []
        for product_name, (score, data) in sorted_products:
            # Add metadata about recommendation source
            product_data = data.to_dict()
            product_data['RecommendationScore'] = score
            product_data['ContentBasedWeight'] = product_scores[product_name]['content_based'] * self.ensemble_weights['content_based']
            product_data['CollaborativeWeight'] = product_scores[product_name]['collaborative'] * self.ensemble_weights['collaborative']
            product_data['NeuralWeight'] = product_scores[product_name]['neural'] * self.ensemble_weights['neural']
            
            # Calculate dominant recommendation type
            weights = {
                'Content-Based': product_data['ContentBasedWeight'],
                'Collaborative': product_data['CollaborativeWeight'],
                'Neural': product_data['NeuralWeight']
            }
            product_data['DominantRecommendationType'] = max(weights.items(), key=lambda x: x[1])[0]
            
            results.append(product_data)
        
        # Return DataFrame with recommendation metadata
        recommendation_df = pd.DataFrame(results)
        
        # Make sure all the required columns exist (even if empty)
        # Return only the standard columns by default
        standard_columns = ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
        
        # Only include columns that actually exist in the data
        available_standard_columns = [col for col in standard_columns if col in recommendation_df.columns]
        
        # Add recommendation metadata columns
        metadata_columns = ['RecommendationScore', 'DominantRecommendationType']
        
        # Combine standard and metadata columns, only including those that exist
        return_columns = available_standard_columns + metadata_columns
        
        # Make sure we have at least the Name column
        if 'Name' not in recommendation_df.columns:
            print("Error: Recommendations missing 'Name' column")
            # Create a minimal empty DataFrame
            return pd.DataFrame(columns=return_columns)
        
        # Some columns might be missing, so we need to ensure they exist in the DataFrame
        for col in return_columns:
            if col not in recommendation_df.columns:
                recommendation_df[col] = None
        
        return recommendation_df[return_columns]
    
    def get_advanced_recommendations(self, user_id, item_name=None, top_n=10, include_similar_users=False):
        """
        Get advanced recommendations with additional features like similar users
        
        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        item_name : str, optional
            Name of the product to use for content-based recommendations
        top_n : int, optional
            Number of recommendations to return
        include_similar_users : bool, optional
            Whether to include recommendations from similar users
            
        Returns:
        --------
        dict
            Dictionary with recommended products and additional information
        """
        # Get base recommendations
        recommendations = self.get_recommendations(user_id, item_name, top_n)
        
        result = {
            'recommendations': recommendations
        }
        
        # If requested, find similar users and their preferred products
        if include_similar_users and user_id is not None and self.user_activity_data is not None:
            similar_users = []
            if self.ncf_model is not None:
                try:
                    similar_users = self.ncf_model.get_similar_users(user_id, top_n=3)
                except Exception as e:
                    print(f"Error getting similar users: {e}")
            
            if similar_users:
                similar_user_data = []
                
                for similar_user_id, similarity in similar_users:
                    # Get this user's most viewed/purchased products
                    user_activities = self.user_activity_data[
                        self.user_activity_data['user_id'] == similar_user_id
                    ].sort_values('timestamp', ascending=False)
                    
                    if not user_activities.empty:
                        recent_products = user_activities['product_name'].unique()[:5]
                        
                        # Get product details
                        product_details = []
                        for product_name in recent_products:
                            product_data = self.train_data[self.train_data['Name'] == product_name]
                            if not product_data.empty:
                                product_details.append(product_data.iloc[0][['Name', 'Brand', 'Rating']].to_dict())
                        
                        similar_user_data.append({
                            'user_id': similar_user_id,
                            'similarity': similarity,
                            'recent_products': product_details
                        })
                
                result['similar_users'] = similar_user_data
        
        return result
    
    def refresh_models(self, user_activity_data):
        """
        Refresh all recommendation models with updated user activity data
        
        Parameters:
        -----------
        user_activity_data : pandas DataFrame
            Updated user activity data
        """
        self.user_activity_data = user_activity_data
        
        # Refresh traditional models
        self.traditional_model = RecommendationModels(self.train_data, user_activity_data)
        
        # Refresh NCF model
        try:
            self.ncf_model = NeuralCollaborativeFiltering(user_activity_data, self.train_data)
        except Exception as e:
            print(f"Error refreshing Neural CF model: {e}")
            print("Will continue with traditional models only")
            self.ncf_model = None
        
        print("Recommendation models refreshed with updated user data")