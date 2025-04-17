import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image, ImageEnhance
import io
import requests
from urllib3.exceptions import InsecureRequestWarning
import warnings
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Suppress SSL warnings (temporary for testing, remove in production)
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

class ImageFeatureExtractor:
    def __init__(self, product_data=None):
        self.product_data = product_data
        self.model_dir = 'models'
        self.features_path = os.path.join(self.model_dir, 'product_image_features.pkl')
        self.model = None
        self.image_features = None
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_or_create_model()
        self._load_or_extract_features()

    def _load_or_create_model(self):
        try:
            base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.model = Model(inputs=base_model.input, outputs=base_model.output)
            print("Loaded MobileNetV2 model for feature extraction")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_or_extract_features(self):
        try:
            self.image_features = joblib.load(self.features_path)
            print(f"Loaded pre-computed image features for {len(self.image_features)} products")
        except (FileNotFoundError, IOError):
            if self.product_data is not None:
                print("Extracting image features for products...")
                self._extract_features()
            else:
                print("No product data provided, will extract features on demand")
                self.image_features = {}

    def _setup_session(self):
        """Set up a session with retries for robust image downloading"""
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _download_image(self, url, max_attempts=3):
        """Download image from URL with retries and handle multiple URLs"""
        try:
            session = self._setup_session()
            # Split URLs if multiple are provided (separated by |)
            urls = [u.strip() for u in url.split('|') if u.strip()]
            for attempt, img_url in enumerate(urls, 1):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = session.get(img_url, headers=headers, timeout=10, verify=False)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content)).convert('RGB')  # Ensure RGB
                    # Enhance contrast and brightness for better feature extraction
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)  # Slight contrast boost
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.1)  # Slight brightness boost
                    return img
                except Exception as e:
                    print(f"Attempt {attempt} failed for URL {img_url}: {e}")
                    if attempt == max_attempts:
                        print(f"Failed to download image after {max_attempts} attempts for {url}")
                        return None
            return None
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None

    def _process_image(self, img):
        """Process PIL Image for feature extraction with background removal (simplified)"""
        try:
            # Resize to MobileNetV2 input size
            img = img.resize((224, 224))
            # Convert to array and preprocess
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def _extract_features(self):
        if self.product_data is None or 'ImageURL' not in self.product_data.columns or 'Name' not in self.product_data.columns:
            print("No product data or required columns available for feature extraction")
            return

        self.image_features = {}
        total_products = len(self.product_data)

        for idx, row in self.product_data.iterrows():
            try:
                product_name = row['Name']
                image_url = row['ImageURL']
                
                # Download and process image
                img = self._download_image(image_url)
                if img is None:
                    print(f"Skipping {product_name} due to image download failure")
                    continue

                img_array = self._process_image(img)
                if img_array is None:
                    continue

                # Extract features
                features = self.model.predict(img_array, verbose=0)
                self.image_features[product_name] = features

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{total_products} product images")

            except Exception as e:
                print(f"Error processing image for product {product_name}: {e}")

        if self.image_features:
            joblib.dump(self.image_features, self.features_path)
            print(f"Saved image features for {len(self.image_features)} products")
        else:
            print("No features extracted successfully")

    def extract_image_features(self, img_data):
        try:
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            # Enhance image for better feature extraction
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            img_array = self._process_image(img)
            if img_array is None:
                return None
            features = self.model.predict(img_array, verbose=0)
            return features
        except Exception as e:
            print(f"Error extracting features from image: {e}")
            return None

    def find_similar_products(self, img_data, top_n=10, category_filter=None):
        if self.product_data is None or not self.image_features:
            print("No product data or image features available for similarity search")
            return pd.DataFrame()

        query_features = self.extract_image_features(img_data)
        if query_features is None:
            return pd.DataFrame()

        similarities = {}
        for product_name, features in self.image_features.items():
            try:
                similarity = cosine_similarity(query_features, features)[0][0]
                similarities[product_name] = similarity
            except Exception as e:
                print(f"Error calculating similarity for {product_name}: {e}")
                continue

        if not similarities:
            return pd.DataFrame()

        # Sort by similarity
        sorted_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Apply category filter if provided
        similar_products = []
        for product_name, similarity in sorted_products[:top_n * 2]:  # Get extra to filter
            product = self.product_data[self.product_data['Name'] == product_name]
            if not product.empty:
                product_info = product.iloc[0]
                # Check if category matches (e.g., "beauty, skin, care, facial, cleansers")
                if category_filter:
                    if category_filter in str(product_info.get('Category', '')):
                        product_dict = product_info.to_dict()
                        product_dict['similarity'] = similarity
                        similar_products.append(product_dict)
                else:
                    product_dict = product_info.to_dict()
                    product_dict['similarity'] = similarity
                    similar_products.append(product_dict)

        if not similar_products:
            return pd.DataFrame()

        result_df = pd.DataFrame(similar_products)
        # Sort by similarity again after filtering
        result_df = result_df.sort_values(by='similarity', ascending=False).head(top_n)

        relevant_cols = ['Name', 'Brand', 'Rating', 'ReviewCount', 'ImageURL', 'Category', 'similarity']
        available_cols = [col for col in relevant_cols if col in result_df.columns]
        return result_df[available_cols]