from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.utils import secure_filename
import io
from PIL import Image
import json
import traceback

# Import your recommendation system components
from modules.recommendation.enhanced_recommendation_system import EnhancedRecommendationSystem
from modules.search.nlp_search import NLPSearch
from modules.search.image_feature_extractor import ImageFeatureExtractor
from modules.search.multimodal_search import MultimodalSearch
from modules.testing.ab_testing import ABTestingManager
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from modules.utils.data_loader import load_product_data


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['MODEL_DIR'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
SWAGGER_URL = '/api/docs'  
API_YAML_URL = '/static/swagger.yaml'  

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_YAML_URL,
    config={
        'app_name': "E-commerce Recommendation API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# Load your data
try:
    train_data = load_product_data('models/clean_data.json')
    print(f"Loaded product data with {len(train_data)} products")
except Exception as e:
    print(f"Error loading product data: {e}")
    train_data = None

# Initialize your recommendation models
try:
    recommendation_system = EnhancedRecommendationSystem(train_data)
    print("Initialized Enhanced Recommendation System")
except Exception as e:
    print(f"Error initializing recommendation system: {e}")
    recommendation_system = None

try:
    nlp_search = NLPSearch(train_data)
    print("Initialized NLP Search system")
except Exception as e:
    print(f"Error initializing NLP Search: {e}")
    nlp_search = None

try:
    image_search_extractor = ImageFeatureExtractor(train_data)
    print("Initialized Image Search feature extractor")
except Exception as e:
    print(f"Error initializing Image Search: {e}")
    image_search_extractor = None

try:
    multimodal_search = MultimodalSearch(image_search_extractor, nlp_search)
    print("Initialized Multimodal Search")
except Exception as e:
    print(f"Error initializing Multimodal Search: {e}")
    multimodal_search = None

try:
    ab_testing_manager = ABTestingManager(model_dir='models')
    print("Initialized A/B Testing Manager")
except Exception as e:
    print(f"Error initializing A/B Testing Manager: {e}")
    ab_testing_manager = None

# Helper Functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def format_product_response(products_df):
    """Convert DataFrame to JSON-serializable format"""
    if products_df is None or products_df.empty:
        return []
    
    return products_df.to_dict(orient='records')

# API Routes

@app.route('/')
def api_home():
    """API home page with basic information and links"""
    return jsonify({
        "service": "E-commerce Recommendation API",
        "version": "1.0.0",
        "description": "AI-powered recommendation and search API for e-commerce platforms",
        "documentation": "/api/docs",
        "status": "online",
        "endpoints": {
            "info": "/api/info",
            "recommendations": "/api/recommendations/content, /api/recommendations/user",
            "search": "/api/search/nlp, /api/search/image, /api/search/multimodal, /api/search/voice",
            "ai_assistant": "/api/ai_assistant/chat, /api/ai_assistant/suggestions",
            "trending": "/api/trending",
            "autocomplete": "/api/autocomplete"
        },
        "message": "Please see the documentation at /api/docs for complete API information"
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get information about the recommendation API"""
    components = {
        "recommendation_system": recommendation_system is not None,
        "nlp_search": nlp_search is not None,
        "image_search": image_search_extractor is not None,
        "multimodal_search": multimodal_search is not None,
        "ab_testing": ab_testing_manager is not None
    }
    
    model_info = {
        "product_count": len(train_data) if train_data is not None else 0,
        "components_available": components,
        "version": "1.0.0"
    }
    
    return jsonify({
        "status": "success",
        "service": "E-commerce Recommendation API",
        "model_info": model_info
    })

@app.route('/api/recommendations/content', methods=['GET'])
def content_recommendations():
    """Get content-based recommendations"""
    item_name = request.args.get('item_name')
    top_n = request.args.get('top_n', default=5, type=int)
    
    if not item_name:
        return jsonify({
            "status": "error",
            "message": "Missing required parameter: item_name"
        }), 400
    
    try:
        if recommendation_system is None:
            return jsonify({
                "status": "error",
                "message": "Recommendation system not available"
            }), 503
        
        recommendations = recommendation_system.traditional_model.get_content_based_recommendations(
            item_name=item_name, 
            top_n=top_n
        )
        
        return jsonify({
            "status": "success",
            "item": item_name,
            "recommendations": format_product_response(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/recommendations/user', methods=['GET'])
def user_recommendations():
    """Get personalized user recommendations"""
    user_id = request.args.get('user_id', type=int)
    item_name = request.args.get('item_name')
    top_n = request.args.get('top_n', default=5, type=int)
    
    if not user_id:
        return jsonify({
            "status": "error",
            "message": "Missing required parameter: user_id"
        }), 400
    
    try:
        if recommendation_system is None:
            return jsonify({
                "status": "error",
                "message": "Recommendation system not available"
            }), 503
        
        recommendations = recommendation_system.get_recommendations(
            user_id=user_id,
            item_name=item_name,
            top_n=top_n
        )
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "item": item_name,
            "recommendations": format_product_response(recommendations)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/search/nlp', methods=['GET'])
def nlp_search_api():
    """Perform NLP-based search"""
    query = request.args.get('query')
    top_n = request.args.get('top_n', default=10, type=int)
    user_id = request.args.get('user_id', type=int)
    
    if not query:
        return jsonify({
            "status": "error",
            "message": "Missing required parameter: query"
        }), 400
    
    try:
        if nlp_search is None:
            return jsonify({
                "status": "error",
                "message": "NLP search not available"
            }), 503
        
        results, query_info = nlp_search.enhanced_search(
            query_text=query,
            user_id=user_id,
            top_n=top_n
        )
        
        return jsonify({
            "status": "success",
            "query": query,
            "detected_attributes": query_info['attributes'],
            "results": format_product_response(results)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/search/image', methods=['POST'])
def image_search_api():
    """Perform image-based search"""
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image file provided"
        }), 400
    
    file = request.files['image']
    top_n = request.form.get('top_n', default=10, type=int)
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No image selected"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Allowed types: png, jpg, jpeg, gif"
        }), 400
    
    try:
        if image_search_extractor is None:
            return jsonify({
                "status": "error",
                "message": "Image search not available"
            }), 503
        
        # Read the image data
        img_data = file.read()
        
        # Save the file (optional)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(img_data)
        
        # Perform image search
        results = image_search_extractor.find_similar_products(img_data, top_n=top_n)
        
        return jsonify({
            "status": "success",
            "image": filename,
            "results": format_product_response(results)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/search/multimodal', methods=['POST'])
def multimodal_search_api():
    """Perform multimodal (image + text) search"""
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image file provided"
        }), 400
    
    file = request.files['image']
    text_query = request.form.get('text_query', '')
    top_n = request.form.get('top_n', default=10, type=int)
    image_weight = float(request.form.get('image_weight', 50)) / 100
    text_weight = 1.0 - image_weight
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No image selected"
        }), 400
    
    if not text_query:
        return jsonify({
            "status": "error",
            "message": "No text query provided"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Allowed types: png, jpg, jpeg, gif"
        }), 400
    
    try:
        if multimodal_search is None:
            return jsonify({
                "status": "error",
                "message": "Multimodal search not available"
            }), 503
        
        # Read the image data
        img_data = file.read()
        
        # Save the file (optional)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(img_data)
        
        # Perform multimodal search
        results, query_info = multimodal_search.search(
            image_data=img_data,
            text_query=text_query,
            top_n=top_n,
            weight_image=image_weight,
            weight_text=text_weight
        )
        
        return jsonify({
            "status": "success",
            "image": filename,
            "text_query": text_query,
            "detected_attributes": query_info['attributes'],
            "results": format_product_response(results)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/trending', methods=['GET'])
def trending_products_api():
    """Get trending products"""
    top_n = request.args.get('top_n', default=10, type=int)
    
    try:
        try:
            trending_data = pd.read_csv("models/trending_products.csv")
            trending = trending_data.head(top_n)
        except Exception:
            # If trending products file is not available, use recommendation system
            if recommendation_system is not None:
                trending = recommendation_system.traditional_model.get_popular_products(top_n=top_n)
            else:
                return jsonify({
                    "status": "error",
                    "message": "Trending products not available"
                }), 503
        
        return jsonify({
            "status": "success",
            "trending_products": format_product_response(trending)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/activity', methods=['POST'])
def record_user_activity():
    """Record user activity for improved recommendations"""
    data = request.json
    
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400
    
    user_id = data.get('user_id')
    product_name = data.get('product_name')
    activity_type = data.get('activity_type')  # view, search, purchase, etc.
    
    if not all([user_id, product_name, activity_type]):
        return jsonify({
            "status": "error",
            "message": "Missing required parameters: user_id, product_name, activity_type"
        }), 400
    
    try:
        # For A/B testing analytics
        if ab_testing_manager is not None:
            if activity_type == 'view':
                ab_testing_manager.record_metric(user_id, 'impression')
            elif activity_type == 'click':
                ab_testing_manager.record_metric(user_id, 'click')
            elif activity_type == 'purchase':
                ab_testing_manager.record_metric(user_id, 'purchase')
            
        # Return success even if we're not storing in a database in this API version
        return jsonify({
            "status": "success",
            "message": "Activity recorded successfully"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/ab_testing/assign', methods=['GET'])
def assign_ab_test():
    """Assign a user to A/B test variants"""
    user_id = request.args.get('user_id', type=int)
    test_id = request.args.get('test_id')
    
    if not user_id:
        return jsonify({
            "status": "error",
            "message": "Missing required parameter: user_id"
        }), 400
    
    try:
        if ab_testing_manager is None:
            return jsonify({
                "status": "error",
                "message": "A/B testing not available"
            }), 503
        
        # Assign user to variant
        variant_assignments = ab_testing_manager.assign_user_to_variant(user_id, test_id)
        
        # Get user weights
        weights = ab_testing_manager.get_user_weights(user_id)
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "variant_assignments": variant_assignments,
            "weights": weights
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get products by category, brand, etc."""
    category = request.args.get('category')
    brand = request.args.get('brand')
    query = request.args.get('query')
    limit = request.args.get('limit', default=20, type=int)
    offset = request.args.get('offset', default=0, type=int)
    
    try:
        if train_data is None:
            return jsonify({
                "status": "error",
                "message": "Product data not available"
            }), 503
        
        filtered_data = train_data.copy()
        
        # Apply filters
        if category:
            filtered_data = filtered_data[filtered_data['Category'].str.contains(category, case=False, na=False)]
        
        if brand:
            filtered_data = filtered_data[filtered_data['Brand'].str.contains(brand, case=False, na=False)]
        
        if query:
            # Simple search across name, description, and tags
            filtered_data = filtered_data[
                filtered_data['Name'].str.contains(query, case=False, na=False) |
                filtered_data['Description'].str.contains(query, case=False, na=False) |
                filtered_data['Tags'].str.contains(query, case=False, na=False)
            ]
        
        # Get total count before pagination
        total_count = len(filtered_data)
        
        # Apply pagination
        paginated_data = filtered_data.iloc[offset:offset+limit]
        
        # Prepare response
        columns = ['Name', 'Brand', 'Rating', 'ReviewCount', 'ImageURL', 'Category']
        available_cols = [col for col in columns if col in paginated_data.columns]
        
        return jsonify({
            "status": "success",
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "products": paginated_data[available_cols].to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/add_product', methods=['POST'])
def add_product():
    """Add a new product to the dataset (for demonstration)"""
    data = request.json
    
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400
    
    required_fields = ['Name', 'Brand', 'Category']
    for field in required_fields:
        if field not in data:
            return jsonify({
                "status": "error",
                "message": f"Missing required field: {field}"
            }), 400
    
    try:
        # This is a simplified demonstration - in a real system, you'd persist to a database
        return jsonify({
            "status": "success",
            "message": "Product added successfully (demo)",
            "product": data
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Run the API server (if script is executed directly)
if __name__ == '__main__':
    # For production, you should use a proper WSGI server like Gunicorn
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)