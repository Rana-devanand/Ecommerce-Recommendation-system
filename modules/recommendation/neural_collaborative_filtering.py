import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam  # Standard optimizer, not legacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import json
from datetime import datetime

class NeuralCollaborativeFiltering:
    def __init__(self, user_activity_data=None, product_data=None, embedding_dim=32):
        """
        Initialize Neural Collaborative Filtering model
        
        Parameters:
        -----------
        user_activity_data : pandas DataFrame
            User activity data with columns user_id, product_name, timestamp, activity_type
        product_data : pandas DataFrame
            Product data with features like Name, Brand, Tags, etc.
        embedding_dim : int
            Dimension of embedding vectors for users and items
        """
        self.user_activity_data = user_activity_data
        self.product_data = product_data
        self.embedding_dim = embedding_dim
        
        # Model paths
        self.model_dir = 'models'
        self.weights_path = os.path.join(self.model_dir, 'ncf_weights.h5')
        self.model_config_path = os.path.join(self.model_dir, 'ncf_model_config.json')
        self.user_encoder_path = os.path.join(self.model_dir, 'user_encoder.pkl')
        self.item_encoder_path = os.path.join(self.model_dir, 'item_encoder.pkl')
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize encoders and model
        self.user_encoder = None
        self.item_encoder = None
        self.model = None
        self.num_users = 0
        self.num_items = 0
        
        # Load or train model
        self._load_or_train_model()

    def _prepare_data(self):
        """
        Prepare training data from user activity and product data
        
        Returns:
        --------
        tuple
            user_ids, item_ids, labels (implicit feedback)
        """
        if self.user_activity_data is None or self.product_data is None:
            print("User activity data or product data not provided")
            return None, None, None
        
        # Get unique products from the product data
        products = self.product_data['Name'].unique()
        
        # Filter user activity to only include products in our product data
        filtered_activity = self.user_activity_data[
            self.user_activity_data['product_name'].isin(products)
        ]
        
        if filtered_activity.empty:
            print("No matching activities found in product data")
            return None, None, None
        
        # Create user and item encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Extract user and item ids
        user_ids = self.user_encoder.fit_transform(filtered_activity['user_id'])
        item_ids = self.item_encoder.fit_transform(filtered_activity['product_name'])
        
        # Store counts for model building
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        # Create implicit feedback based on activity type
        # Define weights for different activity types
        activity_weights = {
            'view': 1.0,
            'search': 0.5,
            'purchase': 3.0,
        }
        
        # Calculate ratings based on activity type and recency
        max_timestamp = filtered_activity['timestamp'].max()
        
        # Create labels (ratings)
        labels = []
        for _, row in filtered_activity.iterrows():
            activity_type = row['activity_type']
            
            # Calculate recency weight
            time_diff = (max_timestamp - row['timestamp']).total_seconds() / 86400  # Convert to days
            recency_weight = 1.0 / (1.0 + time_diff * 0.1)  # Decay function
            
            # Calculate final weight
            weight = activity_weights.get(activity_type, 1.0) * recency_weight
            
            # Normalize weight to 0-1 range
            normalized_weight = min(weight / 3.0, 1.0)
            
            labels.append(normalized_weight)
        
        # Save encoders
        joblib.dump(self.user_encoder, self.user_encoder_path)
        joblib.dump(self.item_encoder, self.item_encoder_path)
        
        return user_ids, item_ids, np.array(labels)
    
    def _build_model(self, num_users, num_items):
        """
        Build the Neural Collaborative Filtering model architecture
        
        Parameters:
        -----------
        num_users : int
            Number of unique users
        num_items : int
            Number of unique items
            
        Returns:
        --------
        keras.Model
            Compiled NCF model
        """
        # User embedding input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            input_dim=num_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_vec = Flatten(name='flatten_users')(user_embedding)
        
        # Item embedding input
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(
            input_dim=num_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        item_vec = Flatten(name='flatten_items')(item_embedding)
        
        # Concatenate user and item embeddings
        concat = Concatenate(name='concat')([user_vec, item_vec])
        
        # Deep Network
        fc1 = Dense(64, activation='relu', name='fc1')(concat)
        dropout1 = Dropout(0.2)(fc1)
        fc2 = Dense(32, activation='relu', name='fc2')(dropout1)
        dropout2 = Dropout(0.2)(fc2)
        fc3 = Dense(16, activation='relu', name='fc3')(dropout2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(fc3)
        
        # Create and compile the model
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Standard Adam optimizer
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _save_model(self):
        """Save model weights and configuration separately for compatibility"""
        # Save model weights
        self.model.save_weights(self.weights_path)
        
        # Save model configuration
        model_config = self.model.get_config()
        with open(self.model_config_path, 'w') as f:
            json.dump({
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Model weights saved to {self.weights_path}")
        print(f"Model configuration saved to {self.model_config_path}")
    
    def _train_model(self):
        """Train the NCF model on user activity data"""
        # Prepare data
        user_ids, item_ids, labels = self._prepare_data()
        
        if user_ids is None:
            print("Could not prepare training data")
            return
        
        print(f"Training NCF model with {self.num_users} users and {self.num_items} items")
        
        # Build model
        self.model = self._build_model(self.num_users, self.num_items)
        
        # Split data into train and validation sets
        user_ids_train, user_ids_val, item_ids_train, item_ids_val, labels_train, labels_val = train_test_split(
            user_ids, item_ids, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(
            [user_ids_train, item_ids_train],
            labels_train,
            batch_size=64,
            epochs=5,
            validation_data=([user_ids_val, item_ids_val], labels_val),
            verbose=1
        )
        
        # Save model weights and configuration
        self._save_model()
    
    def _load_or_train_model(self):
        """Load pre-trained model weights if they exist, otherwise train new model"""
        try:
            # Check if model files exist
            if (os.path.exists(self.weights_path) and 
                os.path.exists(self.model_config_path) and 
                os.path.exists(self.user_encoder_path) and 
                os.path.exists(self.item_encoder_path)):
                
                try:
                    # Load encoders
                    self.user_encoder = joblib.load(self.user_encoder_path)
                    self.item_encoder = joblib.load(self.item_encoder_path)
                    
                    # Load model configuration
                    with open(self.model_config_path, 'r') as f:
                        config = json.load(f)
                    
                    self.num_users = config['num_users']
                    self.num_items = config['num_items']
                    self.embedding_dim = config.get('embedding_dim', 32)
                    
                    # Rebuild model architecture
                    self.model = self._build_model(self.num_users, self.num_items)
                    
                    # Load weights
                    self.model.load_weights(self.weights_path)
                    
                    print("Loaded pre-trained NCF model weights")
                except Exception as e:
                    print(f"Error loading pre-trained NCF model: {e}")
                    print("Training a new model instead")
                    self._train_model()
            else:
                print("No pre-trained NCF model found, training new model...")
                self._train_model()
        except Exception as e:
            print(f"Unexpected error in model initialization: {e}")
            print("Training new model...")
            self._train_model()
    
    def get_recommendations(self, user_id, top_n=10):
        """
        Get recommendations for a user using the neural collaborative filtering model
        
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
        if self.model is None or self.user_encoder is None or self.item_encoder is None:
            print("Model or encoders not initialized")
            return pd.DataFrame()
        
        # Check if user is in the encoder
        if user_id not in self.user_encoder.classes_:
            print(f"User {user_id} not found in training data")
            return pd.DataFrame()
        
        # Transform user id
        user_id_encoded = self.user_encoder.transform([user_id])[0]
        
        # Get all items
        all_items = self.item_encoder.classes_
        item_ids_encoded = np.arange(len(all_items))
        
        # Create input arrays
        user_input = np.full(len(item_ids_encoded), user_id_encoded)
        
        # Predict ratings
        predictions = self.model.predict([user_input, item_ids_encoded], verbose=0)
        
        # Get user's already interacted items
        user_activities = self.user_activity_data[self.user_activity_data['user_id'] == user_id]
        user_items = set(user_activities['product_name'])
        
        # Filter out already interacted items
        item_scores = []
        for idx, item_name in enumerate(all_items):
            if item_name not in user_items:
                item_scores.append((item_name, predictions[idx][0]))
        
        # Sort by predicted score
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get the details of the recommended products
        recommended_items = []
        for item_name, _ in item_scores:
            # Find the product in the product data
            product_info = self.product_data[self.product_data['Name'] == item_name]
            if not product_info.empty:
                recommended_items.append(product_info.iloc[0])
        
        if not recommended_items:
            return pd.DataFrame()
        
        return pd.DataFrame(recommended_items)[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    def get_similar_users(self, user_id, top_n=5):
        """
        Find similar users based on embedding vectors
        
        Parameters:
        -----------
        user_id : int
            User ID to find similar users for
        top_n : int, optional
            Number of similar users to return
            
        Returns:
        --------
        list
            List of tuples (user_id, similarity_score)
        """
        if self.model is None or self.user_encoder is None:
            print("Model or user encoder not initialized")
            return []
        
        # Check if user is in the encoder
        if user_id not in self.user_encoder.classes_:
            print(f"User {user_id} not found in training data")
            return []
        
        # Get the user embedding layer
        user_embedding_layer = self.model.get_layer('user_embedding')
        
        # Get all user embeddings
        all_users = self.user_encoder.classes_
        user_indices = np.arange(len(all_users))
        user_embeddings = user_embedding_layer(user_indices).numpy()
        
        # Get target user embedding
        target_user_idx = self.user_encoder.transform([user_id])[0]
        target_embedding = user_embeddings[target_user_idx]
        
        # Calculate cosine similarity
        similarities = []
        for idx, user_embedding in enumerate(user_embeddings):
            if idx != target_user_idx:
                similarity = np.dot(target_embedding, user_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(user_embedding)
                )
                similarities.append((all_users[idx], similarity))
        
        # Sort by similarity
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        
        return similarities