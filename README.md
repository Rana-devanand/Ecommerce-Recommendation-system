# Advanced E-commerce Recommendation System

A sophisticated, AI-powered recommendation engine for e-commerce platforms that combines multiple recommendation strategies, natural language processing, computer vision, and A/B testing to provide highly personalized product suggestions through various interaction methods.

## Features

### Core Recommendation System
- **Hybrid Recommendation System**: Combines three powerful recommendation approaches:
  - Content-Based Filtering
  - Collaborative Filtering
  - Neural Collaborative Filtering (Deep Learning)

- **Ensemble Methodology**: Weighted combination of multiple recommendation techniques with configurable weights

- **User Activity Tracking**: Captures views, searches, and purchases with temporal weighting

- **User Profiles**: Authentication system with personalized recommendations based on activity history

### Advanced Search Capabilities
- **Natural Language Search**: Understand complex queries like "comfortable running shoes for winter" using semantic meaning
  
- **Image-Based Search**: Find visually similar products by uploading product images using deep learning

- **Multimodal Search**: Combine image and text for refined search (e.g., "this dress but in blue")

- **Voice Search**: Voice input capability for hands-free product search and navigation

- **Search Autocomplete**: Real-time query suggestions based on product catalog and popular searches

- **AI Shopping Assistant**: Conversational interface that understands shopping requests and provides personalized recommendations

### E-commerce Features
- **Shopping Cart**: Fully functional cart management system

- **Checkout Process**: Complete purchase flow with shipping and payment options

- **Order History**: View past orders and purchase details

- **Product Recommendations**: Cross-selling suggestions based on cart contents

### A/B Testing & Optimization
- **A/B Testing Framework**: Create and manage experiments with different recommendation strategies
  
- **Statistical Analysis**: Measure the performance of different variants with significance testing

- **Automated Optimization**: Apply the best-performing variant based on key metrics

### User Interface & Visualization
- **Admin Dashboard**: Monitor system performance, test management, and recommendation weight adjustment

- **Recommendation Visualization**: Clear explanation of how each recommendation method works

- **Search Visualization**: Interactive diagrams showing how search algorithms process queries

- **Feedback System**: Collect and analyze user ratings on recommendations

### API Integration
- **RESTful API**: Dedicated API service for easy integration with any e-commerce platform

- **Comprehensive Endpoints**: Access all recommendation, search, and user activity features via API

- **Swagger Documentation**: Interactive API documentation for developers

- **CORS Support**: Cross-Origin Resource Sharing enabled for multi-platform integration

## Technical Implementation

### Recommendation Models
- **Content-Based Filtering**: TF-IDF vectorization and cosine similarity to find related products
  
- **Collaborative Filtering**: User-item interaction analysis with SVD dimensionality reduction
  
- **Neural Collaborative Filtering**: Deep learning with user and item embeddings for complex pattern recognition

### Natural Language Processing
- **Semantic Search**: SentenceTransformer models for meaning-based product search
  
- **Entity Extraction**: Identify attributes like colors, brands, and price ranges from natural language
  
- **Intent Classification**: Understand user search intent for more relevant results

- **Autocomplete System**: Cache-based suggestion engine utilizing frequency analysis and prefix matching

### Computer Vision
- **MobileNetV2**: Pre-trained CNN for extracting visual features from product images
  
- **Visual Similarity**: Find products with similar visual characteristics using feature vector comparison
  
- **Image Enhancement**: Preprocessing techniques to improve feature extraction quality

### Voice Technologies
- **Speech Recognition**: Convert voice input to text for processing

- **Voice Search Integration**: Seamless voice query handling

### A/B Testing Infrastructure
- **Variant Management**: Create and manage different recommendation configurations
  
- **Random Assignment**: Statistical sampling for unbiased user assignment
  
- **Performance Metrics**: Track clicks, conversions, and ratings for each variant
  
- **Statistical Significance**: Calculate confidence levels and lift metrics

### Tech Stack
- **Backend**: Flask (Python)
- **API Service**: Flask-based RESTful API (migrateApi.py)
- **Database**: SQLite with SQLAlchemy ORM
- **Machine Learning**: TF-IDF, SVD, SentenceTransformers, spaCy
- **Deep Learning**: TensorFlow/Keras for Neural Networks and Computer Vision
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **API Documentation**: Swagger UI

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ecommerce-recommendation-system.git
cd ecommerce-recommendation-system
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install additional NLP components
```bash
python -m spacy download en_core_web_sm
```

5. Initialize the database
```bash
flask db init
flask db migrate
flask db upgrade
```

6. Run the application (choose one)
```bash
# For web application
flask run

# For API service
python migrateApi.py
```

## API Usage

The system provides a dedicated API service (migrateApi.py) for integration with external e-commerce platforms:

### Key API Endpoints

```
GET /api/info - Get API information and available components
GET /api/recommendations/content?item_name={name}&top_n={n} - Get content-based recommendations
GET /api/recommendations/user?user_id={id}&item_name={name}&top_n={n} - Get personalized recommendations
GET /api/search/nlp?query={text}&top_n={n} - Perform NLP-based search
POST /api/search/image - Perform image-based search (with image file upload)
POST /api/search/multimodal - Perform combined image and text search
GET /api/trending - Get trending products
POST /api/activity - Record user activity for improved recommendations
GET /api/ab_testing/assign?user_id={id} - Assign a user to A/B test variants
GET /api/products - Get products filtered by category, brand, etc.
```

### API Integration Examples

```python
# Python example using requests
import requests

# Get personalized recommendations
response = requests.get(
    "http://localhost:5000/api/recommendations/user",
    params={"user_id": 123, "top_n": 5}
)
recommendations = response.json()

# Perform NLP search
response = requests.get(
    "http://localhost:5000/api/search/nlp",
    params={"query": "comfortable running shoes for winter", "top_n": 10}
)
search_results = response.json()
```

## User Features

### Web Application
- **Personalized Recommendations**: Receive customized product suggestions based on browsing history
- **Multiple Search Options**: Use text, natural language, voice, image upload, or conversation-based search
- **Autocomplete**: Get real-time suggestions as you type in the search box
- **Voice Input**: Search for products hands-free using voice commands
- **AI Assistant**: Chat with the AI assistant for natural language product discovery
- **Shopping Cart**: Add products, manage quantities, and check out
- **Order Management**: View order history and track purchases
- **Feedback**: Rate recommendations to improve future suggestions

### API Access
- **Integration Options**: Incorporate recommendation functionality into any e-commerce platform
- **Customization**: Configure recommendation weights and search parameters
- **Activity Tracking**: Record user interactions to improve personalization
- **A/B Testing**: Test different recommendation strategies programmatically

## Admin Features
- **Dashboard**: View analytics on system performance and user engagement
- **Weight Adjustment**: Modify the influence of different recommendation models
- **A/B Testing**: Create and monitor tests for optimizing recommendation strategies
- **Performance Metrics**: Track click-through rates, conversion rates, and user ratings
- **User Management**: Monitor user activities and preferences

## How It Works

### Hybrid Recommendation System
The system combines three approaches with configurable weights:

1. **Content-Based Filtering**: Uses TF-IDF vectorization and cosine similarity to find products with similar features to those a user has interacted with.

2. **Collaborative Filtering**: Recommends products based on what similar users have liked, using patterns in user behavior with temporal decay.

3. **Neural Collaborative Filtering**: Uses deep learning with embedding layers to model complex user-item interactions for discovering non-linear patterns.

### Natural Language Search
The system processes natural language queries through:
1. Entity extraction to identify product attributes (color, price, brand)
2. Semantic embedding using SentenceTransformers
3. Attribute-based filtering and similarity scoring

### Autocomplete System
When a user types in the search box:
1. The system checks a prefix cache of common search terms
2. It retrieves popular searches matching the current input
3. It combines product names and previous user queries
4. Results are ranked and displayed in real-time

### Voice Search
The voice search process:
1. Captures audio input from the user
2. Converts speech to text using recognition services
3. Processes the text through the NLP search pipeline
4. Returns relevant products based on the voice query

### Image-Based Search
When a user uploads an image:
1. MobileNetV2 extracts a 1280-dimensional feature vector
2. The system compares this vector with pre-computed vectors for all products
3. Products with the highest cosine similarity are returned

### A/B Testing Framework
The A/B testing system:
1. Creates variants with different recommendation weights
2. Randomly assigns users to variants
3. Tracks performance metrics (CTR, conversion rate, ratings)
4. Calculates statistical significance and lift
5. Can automatically apply the best-performing configuration

## Visualizations

The system includes several visualizations to explain its inner workings:

- **Recommendation Visualization**: Shows how different models contribute to suggestions
- **NLP Search Visualization**: Illustrates how text queries are processed
- **Image Search Visualization**: Demonstrates feature extraction and matching
- **Multimodal Search Visualization**: Explains how text and images are combined
- **Voice Search Visualization**: Shows the speech-to-recommendations pipeline
- **AI Assistant Visualization**: Illustrates how the conversational system works

## Future Improvements

- **Reinforcement Learning**: Optimize recommendations based on long-term user satisfaction
- **Enhanced NLP**: Implement more advanced models (BERT, GPT) for complex query understanding
- **More Advanced CV**: Product segmentation and attribute extraction from images
- **Time-Aware Recommendations**: Capture seasonal trends and preference evolution
- **Explainable AI**: More detailed explanation of why products are recommended
- **Mobile Application**: Dedicated mobile app with camera integration for image search
- **Voice Assistant Integration**: Connect with smart home devices and virtual assistants

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Aayush Sharma