from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, g
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import random
import functools
from datetime import datetime, timedelta
import os
import tensorflow as tf
import warnings
import numpy as np
from collections import Counter
import time
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from flask_cors import CORS
from modules.recommendation.enhanced_recommendation_system import EnhancedRecommendationSystem
from modules.search.nlp_search import NLPSearch  # Updated version from previous response
from modules.search.image_feature_extractor import ImageFeatureExtractor
from werkzeug.utils import secure_filename
from modules.search.multimodal_search import MultimodalSearch
from flask import request, jsonify
from modules.testing.ab_testing import ABTestingManager
from modules.utils.data_loader import load_product_data
from flask_swagger_ui import get_swaggerui_blueprint


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load files
trending_products = load_product_data('models/trending_products.json')
train_data = load_product_data('models/clean_data.json')

# Database configuration
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///ecom.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(days=7)
db = SQLAlchemy(app)
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

# Define your model classes

class AutocompleteCache:
    """Utility for caching common autocomplete suggestions for better performance"""
    
    def __init__(self, app, train_data, db):
        self.app = app
        self.train_data = train_data
        self.db = db
        self.cache = {}
        self.prefix_cache = {}
        self.popular_cache = []
        
        # Set up periodic cache refresh
        self.cache_refresh_interval = 3600  # 1 hour in seconds
        self.last_refresh_time = 0
        
        # Initialize cache
        with app.app_context():
            self.refresh_cache()
    
    def refresh_cache(self):
        """Refresh the autocomplete cache with common prefixes and popular searches"""
        try:
            # Cache popular product names
            product_names = self.train_data['Name'].dropna().tolist()
            
            # Build prefix cache for common beginning characters
            self.prefix_cache = {}
            for name in product_names:
                name_lower = name.lower()
                # Create prefix entries for 2-4 character prefixes
                for i in range(2, min(5, len(name_lower))):
                    prefix = name_lower[:i]
                    if prefix not in self.prefix_cache:
                        self.prefix_cache[prefix] = []
                    
                    # Only keep the top 500 products per prefix to avoid memory issues
                    if len(self.prefix_cache[prefix]) < 500:
                        self.prefix_cache[prefix].append(name)
            
            # Sort each prefix list by relevance (length and popularity)
            for prefix, suggestions in self.prefix_cache.items():
                # Sort by length (shorter first) and then alphabetically
                self.prefix_cache[prefix] = sorted(suggestions, key=lambda x: (len(x), x))[:50]
            
            # Cache popular search queries from user activities
            try:
                popular_searches = self.db.session.query(
                    UserActivity.product_name, 
                    self.db.func.count(UserActivity.id).label('count')
                ).filter(
                    UserActivity.activity_type.in_(['search', 'nlp_search', 'multimodal_search'])
                ).group_by(
                    UserActivity.product_name
                ).order_by(
                    self.db.desc('count')
                ).limit(100).all()
                
                self.popular_cache = [search[0] for search in popular_searches 
                                      if len(search[0]) > 3 and len(search[0]) < 50]
            except Exception as e:
                print(f"Error fetching popular searches: {e}")
                self.popular_cache = []
            
            # Update timestamp
            self.last_refresh_time = time.time()
            
            print(f"Autocomplete cache refreshed - {len(self.prefix_cache)} prefixes, {len(self.popular_cache)} popular searches")
            
        except Exception as e:
            print(f"Error refreshing autocomplete cache: {e}")
    
    def get_suggestions(self, query, max_results=10):
        """Get autocomplete suggestions for a query"""
        # Check if cache needs refresh
        if time.time() - self.last_refresh_time > self.cache_refresh_interval:
            self.refresh_cache()
        
        # Normalize query
        query = query.lower().strip()
        if not query or len(query) < 2:
            return []
        
        # Check if we have cached suggestions for this prefix
        if query in self.prefix_cache:
            return self.prefix_cache[query][:max_results]
        
        # For longer queries, find the best matching popular searches
        matching_popular = [s for s in self.popular_cache if query in s.lower()][:max_results]
        
        # Find products with similar beginnings
        matching_prefixes = []
        for prefix, suggestions in self.prefix_cache.items():
            if len(query) >= 2 and prefix.startswith(query[:2]):  # Match on first 2 chars
                for suggestion in suggestions:
                    if query in suggestion.lower() and suggestion not in matching_prefixes:
                        matching_prefixes.append(suggestion)
                        if len(matching_prefixes) >= max_results:
                            break
        
        # Combine results and deduplicate
        all_suggestions = []
        seen = set()
        
        for suggestion in matching_popular + matching_prefixes:
            if suggestion.lower() not in seen:
                all_suggestions.append(suggestion)
                seen.add(suggestion.lower())
                if len(all_suggestions) >= max_results:
                    break
        
        return all_suggestions



class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    activity_type = db.Column(db.String(50))  # search, view, purchase, etc.
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class RecommendationFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    recommendation_type = db.Column(db.String(50))  # Content-Based, Collaborative, Neural, etc.
    rating = db.Column(db.Integer)  # User feedback rating (1-5)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    comments = db.Column(db.Text, nullable=True)

# Add these models to your app.py file in the model definitions section

class CartItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    price = db.Column(db.Float, nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'product_name': self.product_name,
            'quantity': self.quantity,
            'price': self.price,
            'total': round(self.quantity * self.price, 2)
        }

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    order_date = db.Column(db.DateTime, default=datetime.utcnow)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, paid, shipped, delivered
    payment_id = db.Column(db.String(100), nullable=True)
    shipping_address = db.Column(db.Text, nullable=True)
    
    items = db.relationship('OrderItem', backref='order', lazy=True, cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'order_date': self.order_date,
            'total_amount': self.total_amount,
            'status': self.status,
            'payment_id': self.payment_id,
            'items': [item.to_dict() for item in self.items]
        }

class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    price = db.Column(db.Float, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'product_name': self.product_name,
            'quantity': self.quantity,
            'price': self.price,
            'total': round(self.quantity * self.price, 2)
        }

class Address(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    street_address = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    postal_code = db.Column(db.String(20), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    is_default = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'full_name': self.full_name,
            'street_address': self.street_address,
            'city': self.city,
            'state': self.state,
            'postal_code': self.postal_code,
            'country': self.country,
            'is_default': self.is_default
        }

# Define global variables for the recommendation models
user_activity_data = None
recommendation_system = None
nlp_search = None 
image_search_extractor = None
autocomplete_cache = None

def get_user_activity_data():
    """Query all user activities from the database and convert to DataFrame"""
    activities = UserActivity.query.all()
    if activities:
        activity_data = [{
            'user_id': activity.user_id,
            'product_name': activity.product_name,
            'activity_type': activity.activity_type,
            'timestamp': activity.timestamp
        } for activity in activities]
        return pd.DataFrame(activity_data)
    return pd.DataFrame(columns=['user_id', 'product_name', 'activity_type', 'timestamp'])


def track_search_query(query, user_id=None, search_type='search'):
    """Track search queries to improve autocomplete suggestions"""
    try:
        if not query or len(query.strip()) < 3:
            return
        
        # Clean the query (keep only meaningful content)
        clean_query = query.strip()
        
        # Don't track very long queries or queries that are likely product names
        if len(clean_query) > 100:
            clean_query = clean_query[:100]
        
        if user_id:
            # Record as user activity
            new_activity = UserActivity(
                user_id=user_id,
                product_name=clean_query,
                activity_type=search_type
            )
            db.session.add(new_activity)
            db.session.commit()
        else:
            # For anonymous users, we could implement a separate tracking mechanism
            # or just skip tracking to avoid database bloat
            pass
            
    except Exception as e:
        print(f"Error tracking search query: {e}")
        # Don't raise the exception - this is a non-critical feature


# Initialize database and models within application context
with app.app_context():
    db.create_all()
    user_activity_data = get_user_activity_data()
    
    # Initialize the recommendation system with error handling
    try:
        recommendation_system = EnhancedRecommendationSystem(train_data, user_activity_data)
    except Exception as e:
        print(f"Error initializing Enhanced Recommendation System: {e}")
        print("Falling back to traditional RecommendationModels")
        from modules.recommendation.recommendation_models import RecommendationModels
        recommendation_system = RecommendationModels(train_data, user_activity_data)

    # Initialize NLP search with database
    try:
        nlp_search = NLPSearch(train_data, db=db)
        print("Initialized NLP Search system")
    except Exception as e:
        print(f"Error initializing NLP Search system: {e}")
        print("NLP search functionality may be limited")

    # Initialize image search with error handling
    try:
        print("About to initialize image search extractor...")
        image_search_extractor = ImageFeatureExtractor(train_data)
        print("Initialized Image Search feature extractor")
    except Exception as e:
        import traceback
        print(f"Error initializing Image Search feature extractor: {e}")
        print(traceback.format_exc())
        print("Image search functionality may be limited")

    # Initialize multimodal search - add this after initializing image_search_extractor and nlp_search
    try:
        print("Initializing Multimodal Search...")
        multimodal_search = MultimodalSearch(image_search_extractor, nlp_search)
        print("Multimodal Search initialized successfully")
    except Exception as e:
        import traceback
        print(f"Error initializing Multimodal Search: {e}")
        print(traceback.format_exc())
        print("Multimodal search functionality may be limited")
        multimodal_search = None

    # AI Sales Assistant
    try:
        # Import the fixed version
        from modules.assistant.ai_assistant import AISalesAssistant
        ai_assistant = AISalesAssistant(
            product_data=train_data,
            nlp_search=nlp_search,
            image_search_extractor=image_search_extractor,
            multimodal_search=multimodal_search,
            recommendation_system=recommendation_system
        )
        print("Initialized Fixed AI Sales Assistant")
    except Exception as e:
        import traceback
        print(f"Error initializing AI Assistant: {e}")
        print(traceback.format_exc())
        ai_assistant = None
        print("AI Assistant functionality may be limited")


        # Initialize A/B testing manager
    try:
        ab_testing_manager = ABTestingManager(model_dir='models')
        print("Initialized A/B Testing Manager")
    except Exception as e:
        import traceback
        print(f"Error initializing A/B Testing Manager: {e}")
        print(traceback.format_exc())
        ab_testing_manager = None
        print("A/B Testing functionality may be limited")


    # Initialize autocomplete cache
    try:
        import time  # Make sure this import is at the top of your file
        autocomplete_cache = AutocompleteCache(app, train_data, db)
        print("Autocomplete cache initialized successfully")
    except Exception as e:
        import traceback
        print(f"Error initializing autocomplete cache: {e}")
        print(traceback.format_exc())
        autocomplete_cache = None


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Login required decorator
def login_required(view_function):
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('index'))
        return view_function(*args, **kwargs)
    return decorated_function

# Recommendations functions
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    return text

def content_based_recommendations(train_data, item_name, top_n=10):
    """Get content-based recommendations using the enhanced system"""
    global recommendation_system
    return recommendation_system.get_recommendations(None, item_name, top_n)

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    """Get collaborative filtering recommendations using the enhanced system"""
    global recommendation_system
    return recommendation_system.get_recommendations(target_user_id, None, top_n)

def enhanced_recommendations(train_data, target_user_id, item_name=None, top_n=10):
    """Get enhanced recommendations using the new system"""
    global recommendation_system
    return recommendation_system.get_recommendations(target_user_id, item_name, top_n=top_n)

def refresh_recommendation_models():
    """Refresh recommendation models with updated user activity data"""
    global recommendation_system, user_activity_data
    user_activity_data = get_user_activity_data()
    recommendation_system.refresh_models(user_activity_data)
    print("Recommendation models refreshed with updated user data")


def enhanced_recommendations_with_ab_testing(train_data, target_user_id, item_name=None, top_n=10):
    """Get enhanced recommendations with A/B testing integration"""
    global recommendation_system, ab_testing_manager
    
    # Check if A/B testing is available and user is logged in
    if ab_testing_manager is not None and target_user_id is not None:
        # Assign user to test variants if needed
        ab_testing_manager.assign_user_to_variant(target_user_id)
        
        # Get weights for the user
        weights = ab_testing_manager.get_user_weights(target_user_id)
        
        # Apply weights temporarily for this user
        original_weights = recommendation_system.ensemble_weights.copy()
        recommendation_system.set_ensemble_weights(
            content_based=weights.get('content_based', 0.3),
            collaborative=weights.get('collaborative', 0.2),
            neural=weights.get('neural', 0.5)
        )
        
        # Record an impression for all active tests the user is assigned to
        ab_testing_manager.record_metric(target_user_id, 'impression')
        
        # Get recommendations with user-specific weights
        recommendations = recommendation_system.get_recommendations(target_user_id, item_name, top_n)
        
        # Restore original weights
        recommendation_system.ensemble_weights = original_weights
        
        return recommendations
    else:
        # Fall back to standard recommendations if A/B testing is unavailable
        return recommendation_system.get_recommendations(target_user_id, item_name, top_n)


# Routes
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


# Template filters for datetime formatting
@app.template_filter('parse_iso_datetime')
def parse_iso_datetime(value):
    if value and isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, AttributeError):
            # Fall back for older Python versions or different formats
            try:
                return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                # Try without microseconds
                return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    return value

@app.template_filter('format_datetime')
def format_datetime(value):
    if value and isinstance(value, datetime):
        return value.strftime('%b %d, %I:%M %p')
    return value

@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    if 'user_id' in session:
        user_activities = UserActivity.query.filter_by(user_id=session['user_id']).order_by(UserActivity.timestamp.desc()).limit(3).all()
        if user_activities:
            recent_product = user_activities[0].product_name
            try:
                personalized_recommendations = enhanced_recommendations(
                    train_data, session['user_id'], item_name=recent_product, top_n=8
                )
                if not personalized_recommendations.empty:
                    recommendation_types = personalized_recommendations['DominantRecommendationType'].tolist() if 'DominantRecommendationType' in personalized_recommendations.columns else None
                    if 'RecommendationScore' in personalized_recommendations.columns:
                        display_recommendations = personalized_recommendations.drop(columns=['RecommendationScore'])
                    else:
                        display_recommendations = personalized_recommendations
                    return render_template('index.html', 
                                          trending_products=display_recommendations, 
                                          truncate=truncate,
                                          random_price=random.choice(price),
                                          random_product_image_urls=random_product_image_urls,
                                          user_name=session.get('username', ''),
                                          personalized=True,
                                          recommendation_types=recommendation_types)
            except Exception as e:
                print(f"Error generating personalized recommendations: {e}")
    
    return render_template('index.html', 
                          trending_products=trending_products.head(8), 
                          truncate=truncate,
                          random_product_image_urls=random_product_image_urls,
                          random_price=random.choice(price),
                          user_name=session.get('username', ''))

@app.route("/main")
def main():
    empty_df = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'DominantRecommendationType'])
    return render_template('main.html', content_based_rec=empty_df, truncate=truncate)

@app.route("/index")
def indexredirect():
    return redirect(url_for('index'))

@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = Signup.query.filter_by(username=username).first()
        if existing_user:
            return render_template('index.html', signup_message='Username already exists!')

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        
        session['user_id'] = new_signup.id
        session['username'] = username
        session.permanent = True

        return redirect(url_for('index'))

@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        
        user = Signup.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = username
            session.permanent = True
            
            new_signin = Signin(username=username, password=password)
            db.session.add(new_signin)
            db.session.commit()
            
            return redirect(url_for('index'))
        return render_template('index.html', signup_message='Invalid username or password!')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        
        # Track the search query for autocomplete improvements
        if 'user_id' in session:
            track_search_query(prod, session['user_id'], 'search')
        else:
            track_search_query(prod, None, 'search')
        
        if 'user_id' in session:
            new_activity = UserActivity(
                user_id=session['user_id'],
                product_name=prod,
                activity_type='search'
            )
            db.session.add(new_activity)
            db.session.commit()
            
            # Use A/B testing enhanced recommendations
            recommended_products = enhanced_recommendations_with_ab_testing(
                train_data, session['user_id'], item_name=prod, top_n=nbr
            )
            
            if not recommended_products.empty:
                recommendation_types = recommended_products['DominantRecommendationType'].tolist() if 'DominantRecommendationType' in recommended_products.columns else None
                if 'RecommendationScore' in recommended_products.columns:
                    display_recommendations = recommended_products.drop(columns=['RecommendationScore'])
                else:
                    display_recommendations = recommended_products
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('main.html', 
                                      content_based_rec=display_recommendations, 
                                      truncate=truncate,
                                      random_price=random.choice(price),
                                      personalized=True,
                                      recommendation_types=recommendation_types)
        else:
            content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
            if not content_based_rec.empty:
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('main.html', 
                                      content_based_rec=content_based_rec, 
                                      truncate=truncate,
                                      random_price=random.choice(price))
        
        message = "No recommendations available for this product."
        return render_template('main.html', message=message, content_based_rec=pd.DataFrame(), truncate=truncate)
    
    return render_template('main.html', content_based_rec=pd.DataFrame(), truncate=truncate)


@app.route("/view_product/<path:product_name>")
def view_product(product_name):
    if 'user_id' in session:
        new_activity = UserActivity(
            user_id=session['user_id'],
            product_name=product_name,
            activity_type='view'
        )
        db.session.add(new_activity)
        db.session.commit()
        
        user_activity_count = UserActivity.query.filter_by(user_id=session['user_id']).count()
        if user_activity_count % 5 == 0:  # Refresh every 5 activities
            refresh_recommendation_models()
        
        # Record click event for A/B testing
        if ab_testing_manager is not None:
            ab_testing_manager.record_metric(session['user_id'], 'click')
    
    product = train_data[train_data['Name'] == product_name].iloc[0].to_dict() if not train_data[train_data['Name'] == product_name].empty else None
    if not product:
        return "Product not found", 404
    
    price = random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50])
    
    # Add random product image URLs for the "You Might Also Like" section
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(4)]
    
    return render_template('product_detail.html', product=product, price=price, random_product_image_urls=random_product_image_urls)

@app.route('/profile')
@login_required
def profile():
    user = Signup.query.get(session['user_id'])
    if not user:
        return redirect(url_for('logout'))
    
    activities = UserActivity.query.filter_by(user_id=user.id).order_by(UserActivity.timestamp.desc()).limit(10).all()
    recommendations = pd.DataFrame()
    similar_users = []
    
    if activities:
        recent_product = activities[0].product_name
        try:
            try:
                results = recommendation_system.get_advanced_recommendations(
                    user.id, item_name=recent_product, top_n=5, include_similar_users=True
                )
                recommendations = results['recommendations']
                similar_users = results.get('similar_users', [])
            except (AttributeError, Exception) as e:
                print(f"Falling back to basic recommendations: {e}")
                recommendations = enhanced_recommendations(
                    train_data, session['user_id'], item_name=recent_product, top_n=5
                )
        except Exception as e:
            print(f"Error getting recommendations: {e}")
    
    return render_template('profile.html', 
                          user=user, 
                          activities=activities, 
                          recommendations=recommendations, 
                          truncate=truncate,
                          personalized=True,
                          similar_users=similar_users,
                          ai_assistant=ai_assistant)  # Pass ai_assistant to the template

@app.route('/recommendation_feedback', methods=['POST'])
@login_required
def recommendation_feedback():
    product_name = request.form.get('product_name')
    recommendation_type = request.form.get('recommendation_type')
    rating = request.form.get('rating')
    comments = request.form.get('comments', '')
    
    if not all([product_name, recommendation_type, rating]):
        flash('Missing required information for feedback', 'warning')
        return redirect(request.referrer or url_for('index'))
    
    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
    except ValueError:
        flash('Rating must be a number between 1 and 5', 'warning')
        return redirect(request.referrer or url_for('index'))
    
    feedback = RecommendationFeedback(
        user_id=session['user_id'],
        product_name=product_name,
        recommendation_type=recommendation_type,
        rating=rating,
        comments=comments
    )
    db.session.add(feedback)
    db.session.commit()
    
    # Record rating for A/B testing
    if ab_testing_manager is not None:
        ab_testing_manager.record_metric(session['user_id'], 'rating', rating)
    
    flash('Thank you for your feedback!', 'success')
    return redirect(request.referrer or url_for('index'))

# Auto-optimize weights periodically
@app.before_request
def auto_optimize_ab_testing():
    # Run optimization once per day (check based on current time)
    now = datetime.now()
    
    # Only run at midnight (hour 0)
    if now.hour == 0 and now.minute < 10:  # Run in first 10 minutes of midnight
        try:
            if ab_testing_manager is not None and recommendation_system is not None:
                # Automatically optimize weights based on test results
                ab_testing_manager.auto_optimize_weights(recommendation_system)
                print(f"[{now}] Auto-optimized recommendation weights based on A/B test results")
        except Exception as e:
            print(f"Error in auto-optimization: {e}")
            

@app.route('/recommendation_visualization')
def recommendation_visualization():
    weights = recommendation_system.ensemble_weights if hasattr(recommendation_system, 'ensemble_weights') else {
        'content_based': 0.6, 'collaborative': 0.4, 'neural': 0.0
    }
    return render_template('recommendation_visualization.html', weights=weights)


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if session['user_id'] != 1:
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    weights = recommendation_system.ensemble_weights if hasattr(recommendation_system, 'ensemble_weights') else {
        'content_based': 0.6, 'collaborative': 0.4, 'neural': 0.0
    }
    
    # Get recommendation feedback
    feedback = RecommendationFeedback.query.all()
    feedback_df = pd.DataFrame([{
        'user_id': f.user_id, 'product_name': f.product_name, 'recommendation_type': f.recommendation_type,
        'rating': f.rating, 'timestamp': f.timestamp
    } for f in feedback])
    
    if not feedback_df.empty:
        avg_rating_by_type = feedback_df.groupby('recommendation_type')['rating'].mean().to_dict()
        rating_counts = feedback_df.groupby('rating').size().to_dict()
        recent_feedback = feedback_df.sort_values('timestamp', ascending=False).head(10)
    else:
        avg_rating_by_type, rating_counts, recent_feedback = {}, {}, pd.DataFrame()
    
    # User activities
    activities = UserActivity.query.all()
    activity_df = pd.DataFrame([{
        'user_id': a.user_id, 'product_name': a.product_name, 'activity_type': a.activity_type,
        'timestamp': a.timestamp
    } for a in activities])
    
    if not activity_df.empty:
        activity_counts = activity_df.groupby('activity_type').size().to_dict()
        top_products = activity_df.groupby('product_name').size().sort_values(ascending=False).head(10).to_dict()
        active_users = activity_df.groupby('user_id').size().sort_values(ascending=False).head(10).to_dict()
    else:
        activity_counts, top_products, active_users = {}, {}, {}
    
    # Order data for admin dashboard
    orders = Order.query.all()
    
    # Calculate sales metrics
    total_revenue = sum(order.total_amount for order in orders) if orders else 0
    avg_order_value = total_revenue / len(orders) if orders else 0
    
    # Calculate cart conversion rate
    add_to_cart_count = activity_df[activity_df['activity_type'] == 'add_to_cart'].shape[0] if not activity_df.empty else 0
    purchase_count = activity_df[activity_df['activity_type'] == 'purchase'].shape[0] if not activity_df.empty else 0
    cart_conversion_rate = purchase_count / add_to_cart_count if add_to_cart_count > 0 else 0
    
    # Get abandoned carts
    # Here we're defining "abandoned" as carts with items that haven't been checked out in 24 hours
    one_day_ago = datetime.now() - timedelta(days=1)
    abandoned_cart_items = CartItem.query.filter(CartItem.added_at < one_day_ago).all()
    
    # Group abandoned cart items by user
    abandoned_carts = []
    if abandoned_cart_items:
        cart_by_user = {}
        for item in abandoned_cart_items:
            if item.user_id not in cart_by_user:
                cart_by_user[item.user_id] = {
                    'user_id': item.user_id,
                    'items': [],
                    'total_value': 0,
                    'last_updated': item.added_at,
                    'products': []
                }
            
            cart_by_user[item.user_id]['items'].append(item)
            cart_by_user[item.user_id]['total_value'] += item.price * item.quantity
            cart_by_user[item.user_id]['products'].append(item.product_name)
            
            # Update last_updated time if this item is more recent
            if item.added_at > cart_by_user[item.user_id]['last_updated']:
                cart_by_user[item.user_id]['last_updated'] = item.added_at
        
        abandoned_carts = list(cart_by_user.values())
    
    # Calculate best selling products
    best_selling_products = {}
    for order in orders:
        for item in order.items:
            product_name = item.product_name
            if product_name not in best_selling_products:
                best_selling_products[product_name] = {
                    'quantity': 0,
                    'revenue': 0,
                    'rating': 0
                }
            
            best_selling_products[product_name]['quantity'] += item.quantity
            best_selling_products[product_name]['revenue'] += item.price * item.quantity
            
            # Try to get product rating from the dataset
            product_data = train_data[train_data['Name'] == product_name]
            if not product_data.empty:
                best_selling_products[product_name]['rating'] = product_data.iloc[0]['Rating']
    
    # Sort best selling products by quantity in descending order
    best_selling_products = dict(sorted(
        best_selling_products.items(), 
        key=lambda item: item[1]['quantity'], 
        reverse=True
    )[:10])  # Show top 10 best sellers
    
    return render_template('admin_dashboard.html',
                          avg_rating_by_type=avg_rating_by_type,
                          rating_counts=rating_counts,
                          recent_feedback=recent_feedback,
                          activity_counts=activity_counts,
                          top_products=top_products,
                          active_users=active_users,
                          activity_df=activity_df,
                          weights=weights,
                          # New order-related variables
                          orders=orders,
                          total_revenue=total_revenue,
                          avg_order_value=avg_order_value,
                          cart_conversion_rate=cart_conversion_rate,
                          abandoned_carts=abandoned_carts,
                          best_selling_products=best_selling_products)

@app.route('/admin/adjust_weights', methods=['GET', 'POST'])
@login_required
def adjust_weights():
    if session['user_id'] != 1:
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    if not hasattr(recommendation_system, 'set_ensemble_weights'):
        flash('Weight adjustment not available with the current recommendation system', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        content_weight = float(request.form.get('content_weight', 0.3))
        collaborative_weight = float(request.form.get('collaborative_weight', 0.2))
        neural_weight = float(request.form.get('neural_weight', 0.5))
        
        recommendation_system.set_ensemble_weights(
            content_based=content_weight,
            collaborative=collaborative_weight,
            neural=neural_weight
        )
        
        flash('Recommendation weights updated successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    
    current_weights = recommendation_system.ensemble_weights
    return render_template('adjust_weights.html', weights=current_weights)

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    user_id = request.args.get('user_id')
    item_name = request.args.get('item_name')
    top_n = request.args.get('top_n', 5, type=int)
    
    if not user_id and not item_name:
        return jsonify({'error': 'Either user_id or item_name must be provided'}), 400
    
    try:
        if user_id:
            user_id = int(user_id)
            user = Signup.query.get(user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            recommendations = enhanced_recommendations(train_data, user_id, item_name=item_name, top_n=top_n)
        else:
            recommendations = content_based_recommendations(train_data, item_name, top_n=top_n)
        
        if recommendations.empty:
            return jsonify({'message': 'No recommendations found', 'recommendations': []}), 200
        
        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify({'message': 'Success', 'recommendations': recommendations_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/nlp_search", methods=['POST', 'GET'])
def nlp_search_route():
    if request.method == 'POST':
        query = request.form.get('query')
        top_n = int(request.form.get('top_n', 10))
        
        if not query:
            return render_template('nlp_search.html', message="Please enter a search query")
        
        # Track the NLP search query for autocomplete improvements
        if 'user_id' in session:
            track_search_query(query, session['user_id'], 'nlp_search')
        else:
            track_search_query(query, None, 'nlp_search')
        
        if 'user_id' in session:
            new_activity = UserActivity(
                user_id=session['user_id'],
                product_name=query[:100],
                activity_type='nlp_search'
            )
            db.session.add(new_activity)
            db.session.commit()
        
        try:
            search_results, query_info = nlp_search.enhanced_search(query, user_id=session.get('user_id'), top_n=top_n)
            
            if search_results.empty:
                message = "No products found matching your search criteria."
                return render_template('nlp_search.html', message=message, query=query)
            
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            search_metadata = {
                'original_query': query,
                'detected_attributes': {k: v for k, v in query_info['attributes'].items() if v}
            }
            
            return render_template('nlp_search.html', 
                                  search_results=search_results, 
                                  truncate=truncate,
                                  random_price=random.choice(price),
                                  query=query,
                                  search_metadata=search_metadata)
        except Exception as e:
            import traceback
            print(f"Error in NLP search: {e}")
            print(traceback.format_exc())
            message = "An error occurred while processing your search request."
            return render_template('nlp_search.html', message=message, query=query)
    
    return render_template('nlp_search.html')

@app.route("/nlp_visualization")
def nlp_visualization():
    """Show visualization of how the NLP search system works"""
    return render_template('nlp_visualization.html')

@app.route("/image_search", methods=['POST', 'GET'])
def image_search():
    if request.method == 'POST':
        if image_search_extractor is None:
            message = "Image search functionality is not available at the moment."
            return render_template('image_search.html', message=message)
        
        if 'image_file' not in request.files:
            message = "No file part in the request."
            return render_template('image_search.html', message=message)
        
        file = request.files['image_file']
        if file.filename == '':
            message = "No image selected for uploading."
            return render_template('image_search.html', message=message)
        
        if file and allowed_file(file.filename):
            file_data = file.read()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            if 'user_id' in session:
                new_activity = UserActivity(
                    user_id=session['user_id'],
                    product_name=f"Image search: {filename}",
                    activity_type='image_search'
                )
                db.session.add(new_activity)
                db.session.commit()
            
            try:
                similar_products = image_search_extractor.find_similar_products(file_data, top_n=10)
                
                if similar_products.empty:
                    message = "No similar products found for your image."
                    return render_template('image_search.html', 
                                          message=message,
                                          uploaded_image=f"uploads/{filename}")
                
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                return render_template('image_search.html', 
                                      search_results=similar_products, 
                                      truncate=truncate,
                                      random_price=random.choice(price),
                                      uploaded_image=f"uploads/{filename}")
            except Exception as e:
                import traceback
                print(f"Error in image search: {e}")
                print(traceback.format_exc())
                message = "An error occurred while processing your image search request."
                return render_template('image_search.html', 
                                      message=message,
                                      uploaded_image=f"uploads/{filename}")
        else:
            message = "Allowed image types are: png, jpg, jpeg, gif"
            return render_template('image_search.html', message=message)
    
    return render_template('image_search.html')

@app.route("/image_search_visualization")
def image_search_visualization():
    """Show visualization of how the image search system works"""
    return render_template('image_search_visualization.html')

@app.route("/multimodal_search_route", methods=['POST', 'GET'])
def multimodal_search_route(): 
    """Handle multimodal search requests combining image and text"""
    if multimodal_search is None:  # This refers to the global variable
        return render_template('multimodal_search.html', 
                              message="Multimodal search is not available at the moment.")
    
    if request.method == 'POST':
        # Check if image file is provided
        if 'image_file' not in request.files:
            return render_template('multimodal_search.html', 
                                  message="No image file uploaded. Please provide both image and text.")
        
        file = request.files['image_file']
        text_query = request.form.get('text_query', '')
        image_weight = float(request.form.get('image_weight', 50)) / 100  # Convert to 0-1 scale
        text_weight = 1.0 - image_weight
        
        # Track the text part of the multimodal search for autocomplete improvements
        if text_query and text_query.strip():
            if 'user_id' in session:
                track_search_query(text_query, session['user_id'], 'multimodal_search')
            else:
                track_search_query(text_query, None, 'multimodal_search')
        
        if file.filename == '':
            return render_template('multimodal_search.html', 
                                  message="No image selected. Please upload an image.",
                                  text_query=text_query)
        
        if not text_query.strip():
            return render_template('multimodal_search.html', 
                                  message="No text query provided. Please describe what you're looking for.",
                                  text_query=text_query)
        
        if file and allowed_file(file.filename):
            file_data = file.read()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Record user activity if logged in
            if 'user_id' in session:
                new_activity = UserActivity(
                    user_id=session['user_id'],
                    product_name=f"Multimodal search: {text_query[:50]}",
                    activity_type='multimodal_search'
                )
                db.session.add(new_activity)
                db.session.commit()
            
            try:
                # Perform multimodal search
                search_results, query_info = multimodal_search.search(
                    file_data, 
                    text_query, 
                    top_n=10,
                    weight_image=image_weight,
                    weight_text=text_weight
                )
                
                if search_results.empty:
                    message = "No products found matching your criteria."
                    return render_template('multimodal_search.html', 
                                          message=message,
                                          uploaded_image=f"uploads/{filename}",
                                          text_query=text_query)
                
                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
                search_metadata = {
                    'original_query': text_query,
                    'detected_attributes': {k: v for k, v in query_info['attributes'].items() if v}
                }
                
                return render_template('multimodal_search.html', 
                                      search_results=search_results, 
                                      truncate=truncate,
                                      random_price=random.choice(price),
                                      uploaded_image=f"uploads/{filename}",
                                      text_query=text_query,
                                      search_metadata=search_metadata)
                
            except Exception as e:
                import traceback
                print(f"Error in multimodal search: {e}")
                print(traceback.format_exc())
                message = "An error occurred while processing your search request."
                return render_template('multimodal_search.html', 
                                      message=message,
                                      uploaded_image=f"uploads/{filename}",
                                      text_query=text_query)
        else:
            message = "Allowed image types are: png, jpg, jpeg, gif"
            return render_template('multimodal_search.html', 
                                  message=message,
                                  text_query=text_query)
    
    return render_template('multimodal_search.html')


@app.route("/multimodal_visualization")
def multimodal_visualization():
    """Show visualization of how the multimodal search system works"""
    return render_template('multimodal_visualization.html')



@app.route("/ai_assistant")
def ai_assistant_page():
    """Render the AI shopping assistant page"""
    return render_template('ai_assistant.html')

@app.route("/api/ai_assistant/chat", methods=['POST'])
@login_required
def ai_assistant_chat():
    """API endpoint for AI assistant chat"""
    if ai_assistant is None:
        return jsonify({
            'success': False,
            'message': "AI Assistant is not available at the moment.",
            'response': {
                'text': "I'm sorry, but I'm not available right now. Please try again later.",
                'products': []
            }
        }), 503

    # Get user message
    data = request.json
    message = data.get('message', '')
    
    if not message.strip():
        return jsonify({
            'success': False,
            'message': "Please provide a message.",
            'response': {
                'text': "I didn't receive any message. How can I help you?",
                'products': []
            }
        }), 400
    
    # Process message
    try:
        user_id = session.get('user_id')
        response = ai_assistant.process_message(user_id, message)
        
        # Record the interaction in the database
        try:
            new_activity = UserActivity(
                user_id=user_id,
                product_name=message[:100],
                activity_type='ai_assistant'
            )
            db.session.add(new_activity)
            db.session.commit()
        except Exception as db_error:
            print(f"Failed to log user activity, but continuing: {db_error}")
            # Continue even if recording fails
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        print(f"Error processing AI assistant message: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e),
            'response': {
                'text': "I encountered an error processing your message. Could you try again with different wording?",
                'products': []
            }
        }), 500

@app.route("/api/ai_assistant/suggestions", methods=['GET'])
@login_required
def ai_assistant_suggestions():
    """API endpoint for AI assistant suggestions"""
    if ai_assistant is None:
        return jsonify({
            'success': False,
            'suggestions': []
        }), 503
    
    try:
        user_id = session.get('user_id')
        user_preferences = ai_assistant.get_user_preferences(user_id)
        
        # Generate suggestions based on preferences
        suggestions = [
            "Show me new arrivals",
            "What's trending today?",
            "Help me find a gift"
        ]
        
        # Add personalized suggestions based on preferences
        preferred_categories = user_preferences.get('preferred_categories', {})
        preferred_brands = user_preferences.get('preferred_brands', {})
        style_preferences = user_preferences.get('style_preferences', {})
        
        if preferred_categories:
            top_category = max(preferred_categories.items(), key=lambda x: x[1])[0]
            suggestions.append(f"Show me {top_category} products")
        
        if preferred_brands:
            top_brand = max(preferred_brands.items(), key=lambda x: x[1])[0]
            suggestions.append(f"What's new from {top_brand}?")
        
        if 'color' in style_preferences:
            top_color = max(style_preferences['color'].items(), key=lambda x: x[1])[0]
            
            if preferred_categories:
                top_category = max(preferred_categories.items(), key=lambda x: x[1])[0]
                suggestions.append(f"Find {top_color} {top_category}")
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    except Exception as e:
        print(f"Error generating AI assistant suggestions: {e}")
        # Return default suggestions on error
        return jsonify({
            'success': False,
            'suggestions': [
                "Show me popular products",
                "What's on sale?",
                "Help me find a product"
            ]
        })
    
# Add this route to app.py
@app.route("/ai_assistant_visualization")
def ai_assistant_visualization():
    """Show visualization of how the AI Assistant works"""
    return render_template('ai_assistant_visualization.html')



@app.route("/ab_testing_dashboard")
@login_required
def ab_testing_dashboard():
    """Admin dashboard for A/B testing"""
    if session['user_id'] != 1:  # Only admin can access
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    if ab_testing_manager is None:
        flash('A/B Testing functionality is currently unavailable', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    # Get all tests
    all_tests = ab_testing_manager.get_all_tests()
    active_tests = ab_testing_manager.get_active_tests()
    
    # Convert tests to a list for easier rendering
    tests_list = [test.to_dict() for test in all_tests.values()]
    
    # Sort by start_date descending
    tests_list.sort(key=lambda x: x['start_date'], reverse=True)
    
    return render_template('ab_testing_dashboard.html', 
                           tests=tests_list, 
                           active_count=len(active_tests),
                           total_count=len(all_tests))

@app.route("/ab_testing_explanation")
def ab_testing_explanation():
    """Educational page about A/B testing"""
    return render_template('ab_testing_explanation.html')

@app.route("/ab_testing/create", methods=['POST'])
@login_required
def create_ab_test():
    """Create a new A/B test"""
    if session['user_id'] != 1:  # Only admin can access
        return jsonify({'success': False, 'message': 'Permission denied'}), 403
    
    if ab_testing_manager is None:
        return jsonify({'success': False, 'message': 'A/B Testing functionality is unavailable'}), 500
    
    try:
        # Get form data
        name = request.form.get('name')
        description = request.form.get('description')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        variant_count = int(request.form.get('variant_count', 2))
        
        # Validate required fields
        if not all([name, description, start_date_str, end_date_str]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # Create variant configurations
        variants = []
        
        # Control variant (default weights)
        variants.append({
            'id': 'control',
            'name': 'Control',
            'description': 'Default weight configuration',
            'weights': {
                'content_based': 0.3,
                'collaborative': 0.2,
                'neural': 0.5
            }
        })
        
        # Create test variants with different weight configurations
        for i in range(1, variant_count):
            # Generate different weights for each variant
            content_weight = float(request.form.get(f'content_weight_{i}', 0.33))
            collab_weight = float(request.form.get(f'collab_weight_{i}', 0.33))
            neural_weight = float(request.form.get(f'neural_weight_{i}', 0.34))
            
            # Normalize weights to sum to 1
            total = content_weight + collab_weight + neural_weight
            if total > 0:
                content_weight /= total
                collab_weight /= total
                neural_weight /= total
            else:
                # Default to equal weights if all are zero
                content_weight = collab_weight = neural_weight = 1/3
            
            variants.append({
                'id': f'variant_{i}',
                'name': f'Variant {i}',
                'description': f'Alternative weight configuration {i}',
                'weights': {
                    'content_based': content_weight,
                    'collaborative': collab_weight,
                    'neural': neural_weight
                }
            })
        
        # Create the test
        test_id = ab_testing_manager.create_test(
            name=name,
            description=description,
            variants=variants,
            start_date=start_date,
            end_date=end_date,
            status='active'
        )
        
        flash(f'Successfully created A/B test: {name}', 'success')
        return jsonify({'success': True, 'test_id': test_id}), 200
    
    except Exception as e:
        import traceback
        print(f"Error creating A/B test: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route("/ab_testing/view/<test_id>")
@login_required
def view_ab_test(test_id):
    """View details of an A/B test"""
    if session['user_id'] != 1:  # Only admin can access
        flash('You do not have permission to access this page', 'warning')
        return redirect(url_for('index'))
    
    if ab_testing_manager is None:
        flash('A/B Testing functionality is currently unavailable', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    # Get the test
    test = ab_testing_manager.get_test(test_id)
    if not test:
        flash('Test not found', 'warning')
        return redirect(url_for('ab_testing_dashboard'))
    
    # Generate report
    report = ab_testing_manager.generate_report(test_id)
    
    return render_template('ab_test_detail.html', test=test.to_dict(), report=report)

@app.route("/ab_testing/update/<test_id>", methods=['POST'])
@login_required
def update_ab_test(test_id):
    """Update an A/B test"""
    if session['user_id'] != 1:  # Only admin can access
        return jsonify({'success': False, 'message': 'Permission denied'}), 403
    
    if ab_testing_manager is None:
        return jsonify({'success': False, 'message': 'A/B Testing functionality is unavailable'}), 500
    
    try:
        # Get form data
        status = request.form.get('status')
        end_date_str = request.form.get('end_date')
        
        # Parse end date if provided
        end_date = None
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # Update the test
        updates = {}
        if status:
            updates['status'] = status
        if end_date:
            updates['end_date'] = end_date
        
        if updates:
            success = ab_testing_manager.update_test(test_id, **updates)
            if success:
                return jsonify({'success': True}), 200
            else:
                return jsonify({'success': False, 'message': 'Test not found'}), 404
        else:
            return jsonify({'success': False, 'message': 'No updates provided'}), 400
    
    except Exception as e:
        import traceback
        print(f"Error updating A/B test: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route("/ab_testing/delete/<test_id>", methods=['POST'])
@login_required
def delete_ab_test(test_id):
    """Delete an A/B test"""
    if session['user_id'] != 1:  # Only admin can access
        return jsonify({'success': False, 'message': 'Permission denied'}), 403
    
    if ab_testing_manager is None:
        return jsonify({'success': False, 'message': 'A/B Testing functionality is unavailable'}), 500
    
    success = ab_testing_manager.delete_test(test_id)
    if success:
        flash('Test deleted successfully', 'success')
        return jsonify({'success': True}), 200
    else:
        return jsonify({'success': False, 'message': 'Test not found'}), 404

@app.route("/ab_testing/apply_best/<test_id>", methods=['POST'])
@login_required
def apply_best_variant(test_id):
    """Apply the best-performing variant as the default weights"""
    if session['user_id'] != 1:  # Only admin can access
        return jsonify({'success': False, 'message': 'Permission denied'}), 403
    
    if ab_testing_manager is None or recommendation_system is None:
        return jsonify({'success': False, 'message': 'Required functionality is unavailable'}), 500
    
    test = ab_testing_manager.get_test(test_id)
    if not test:
        return jsonify({'success': False, 'message': 'Test not found'}), 404
    
    primary_metric = request.form.get('primary_metric', 'avg_rating')
    best_variant = test.get_best_variant(primary_metric)
    
    if not best_variant or 'weights' not in best_variant:
        return jsonify({'success': False, 'message': 'No valid best variant found'}), 400
    
    # Apply the weights
    weights = best_variant['weights']
    recommendation_system.set_ensemble_weights(
        content_based=weights.get('content_based', 0.3),
        collaborative=weights.get('collaborative', 0.2),
        neural=weights.get('neural', 0.5)
    )
    
    flash(f'Successfully applied weights from {best_variant["name"]}', 'success')
    return jsonify({'success': True, 'variant': best_variant}), 200


# Add these routes to your app.py file
def get_cart_count(user_id):
    """Get the number of items in the user's cart"""
    if not user_id:
        return 0
    return db.session.query(db.func.sum(CartItem.quantity)).filter_by(user_id=user_id).scalar() or 0

# Add this before_request handler to make cart count available to all templates
@app.before_request
def before_request():
    """Run before each request to set up global variables"""
    if 'user_id' in session:
        # Add cart count to all templates
        g.cart_item_count = get_cart_count(session['user_id'])
    else:
        g.cart_item_count = 0

# Add this context processor to make the cart count available to all templates
@app.context_processor
def inject_cart_count():
    """Add cart count to template context"""
    return {'cart_item_count': getattr(g, 'cart_item_count', 0)}

# Cart and Checkout Routes
@app.route('/cart')
@login_required
def view_cart():
    """View shopping cart contents"""
    user_id = session['user_id']
    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    
    # Prepare cart items with product details
    enriched_cart_items = []
    for item in cart_items:
        product_data = train_data[train_data['Name'] == item.product_name]
        if not product_data.empty:
            product = product_data.iloc[0]
            enriched_item = item.to_dict()
            enriched_item['image_url'] = product['ImageURL']
            enriched_item['brand'] = product['Brand']
            enriched_cart_items.append(enriched_item)
    
    # Calculate totals
    subtotal = sum(item['total'] for item in enriched_cart_items)
    
    # Apply shipping cost - free shipping for orders over $50
    shipping_cost = 0 if subtotal >= 50 else 5.99
    
    # Apply tax (e.g. 8%)
    tax = subtotal * 0.08
    
    # Calculate total
    total = subtotal + shipping_cost + tax
    
    # Get recommendations for cross-selling
    recommendations = []
    if cart_items:
        # Get the most recent item in the cart
        recent_item = cart_items[-1].product_name
        try:
            rec_df = recommendation_system.get_recommendations(user_id, recent_item, top_n=4)
            # Add Price column with random values between 20-200
            if not rec_df.empty:
                rec_df['Price'] = np.random.uniform(20, 200, size=len(rec_df))
                recommendations = rec_df.to_dict('records')
        except Exception as e:
            print(f"Error getting recommendations: {e}")
    
    total_items = sum(item['quantity'] for item in enriched_cart_items)
    
    return render_template('cart.html', 
                          cart_items=enriched_cart_items,
                          subtotal=subtotal,
                          shipping_cost=shipping_cost,
                          tax=tax,
                          total=total,
                          total_items=total_items,
                          recommendations=recommendations)

@app.route('/add-to-cart/<path:product_name>', methods=['GET', 'POST'])
@login_required
def add_to_cart(product_name):
    """Add a product to the shopping cart"""
    user_id = session['user_id']
    
    # Get the product from the dataset
    product_data = train_data[train_data['Name'] == product_name]
    if product_data.empty:
        flash('Product not found.', 'danger')
        return redirect(url_for('index'))
    
    # Generate a random price between 20 and 200 dollars
    price = round(np.random.uniform(20, 200), 2)
    
    # Check if the product is already in the cart
    existing_item = CartItem.query.filter_by(user_id=user_id, product_name=product_name).first()
    
    if existing_item:
        # Increment the quantity if already in cart
        existing_item.quantity += 1
        db.session.commit()
        flash(f'Updated quantity of {product_name} in your cart.', 'success')
    else:
        # Add new item to cart
        new_item = CartItem(
            user_id=user_id,
            product_name=product_name,
            price=price,
            quantity=1
        )
        db.session.add(new_item)
        db.session.commit()
        
        # Record user activity
        new_activity = UserActivity(
            user_id=user_id,
            product_name=product_name,
            activity_type='add_to_cart'
        )
        db.session.add(new_activity)
        db.session.commit()
        
        flash(f'Added {product_name} to your cart.', 'success')
    
    # Check if the request wants to redirect to cart or continue shopping
    redirect_to = request.args.get('redirect_to', 'cart')
    if redirect_to == 'cart':
        return redirect(url_for('view_cart'))
    else:
        return redirect(url_for('view_product', product_name=product_name))

@app.route('/remove-from-cart/<int:item_id>')
@login_required
def remove_from_cart(item_id):
    """Remove an item from the shopping cart"""
    user_id = session['user_id']
    
    # Find the cart item
    cart_item = CartItem.query.filter_by(id=item_id, user_id=user_id).first()
    
    if cart_item:
        product_name = cart_item.product_name
        db.session.delete(cart_item)
        db.session.commit()
        flash(f'Removed {product_name} from your cart.', 'success')
    else:
        flash('Item not found in your cart.', 'warning')
    
    return redirect(url_for('view_cart'))

@app.route('/update-cart', methods=['POST'])
@login_required
def update_cart():
    """Update cart quantities"""
    user_id = session['user_id']
    
    # Get all cart items for the user
    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    
    # Update quantities based on form data
    for item in cart_items:
        quantity_key = f'quantity_{item.id}'
        if quantity_key in request.form:
            try:
                new_quantity = int(request.form[quantity_key])
                if 1 <= new_quantity <= 10:  # Limit quantity between 1 and 10
                    item.quantity = new_quantity
            except ValueError:
                pass  # Ignore invalid input
    
    db.session.commit()
    flash('Cart updated successfully.', 'success')
    
    return redirect(url_for('view_cart'))

@app.route('/checkout')
@login_required
def checkout():
    """Show checkout page"""
    user_id = session['user_id']
    
    # Check if cart is empty
    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    if not cart_items:
        flash('Your cart is empty. Add some products before checking out.', 'warning')
        return redirect(url_for('view_cart'))
    
    # Prepare cart items with product details
    enriched_cart_items = []
    for item in cart_items:
        product_data = train_data[train_data['Name'] == item.product_name]
        if not product_data.empty:
            product = product_data.iloc[0]
            enriched_item = item.to_dict()
            enriched_item['image_url'] = product['ImageURL']
            enriched_item['brand'] = product['Brand']
            enriched_cart_items.append(enriched_item)
    
    # Calculate totals
    subtotal = sum(item['total'] for item in enriched_cart_items)
    shipping_cost = 0 if subtotal >= 50 else 5.99
    tax = subtotal * 0.08
    total = subtotal + shipping_cost + tax
    total_items = sum(item['quantity'] for item in enriched_cart_items)
    
    # Get saved addresses
    saved_addresses = Address.query.filter_by(user_id=user_id).all()
    
    return render_template('checkout.html',
                          cart_items=enriched_cart_items,
                          subtotal=subtotal,
                          shipping_cost=shipping_cost,
                          tax=tax,
                          total=total,
                          total_items=total_items,
                          saved_addresses=saved_addresses)

@app.route('/place-order', methods=['POST'])
@login_required
def place_order():
    """Process order placement"""
    user_id = session['user_id']
    
    # Check if cart is empty
    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    if not cart_items:
        flash('Your cart is empty. Add some products before checking out.', 'warning')
        return redirect(url_for('view_cart'))
    
    # Calculate order total
    subtotal = sum(item.price * item.quantity for item in cart_items)
    
    # Get shipping method and cost
    shipping_method = request.form.get('shipping_method', 'standard')
    if shipping_method == 'standard':
        shipping_cost = 0 if subtotal >= 50 else 5.99
    elif shipping_method == 'express':
        shipping_cost = 9.99
    elif shipping_method == 'overnight':
        shipping_cost = 19.99
    else:
        shipping_cost = 0
    
    # Calculate tax
    tax = subtotal * 0.08
    
    # Calculate total
    total_amount = subtotal + shipping_cost + tax
    
    # Process address information
    shipping_address = None
    address_id = request.form.get('address_id')
    
    if address_id and address_id != 'new':
        # Use existing address
        shipping_address = Address.query.filter_by(id=address_id, user_id=user_id).first()
        if not shipping_address:
            flash('The selected address was not found.', 'danger')
            return redirect(url_for('checkout'))
    else:
        # Create new address
        full_name = request.form.get('full_name')
        street_address = request.form.get('street_address')
        city = request.form.get('city')
        state = request.form.get('state')
        postal_code = request.form.get('postal_code')
        country = request.form.get('country')
        
        if not all([full_name, street_address, city, state, postal_code, country]):
            flash('Please fill out all address fields.', 'danger')
            return redirect(url_for('checkout'))
        
        shipping_address = Address(
            user_id=user_id,
            full_name=full_name,
            street_address=street_address,
            city=city,
            state=state,
            postal_code=postal_code,
            country=country
        )
        
        # Save address if requested
        if request.form.get('save_address') == '1':
            shipping_address.is_default = not Address.query.filter_by(user_id=user_id).first()
            db.session.add(shipping_address)
            db.session.commit()
    
    # Process payment information
    payment_method = request.form.get('payment_method', 'credit_card')
    payment_id = None
    card_last_four = None
    card_type = None
    
    if payment_method == 'credit_card':
        # In a real system, you would process payment through a payment gateway
        # For our demo, we'll simulate payment success
        card_number = request.form.get('card_number', '').replace(' ', '')
        card_last_four = card_number[-4:] if len(card_number) >= 4 else '0000'
        
        # Determine card type based on first digit
        if card_number.startswith('4'):
            card_type = 'Visa'
        elif card_number.startswith('5'):
            card_type = 'MasterCard'
        elif card_number.startswith('3'):
            card_type = 'American Express'
        elif card_number.startswith('6'):
            card_type = 'Discover'
        else:
            card_type = 'Credit Card'
        
        # Generate a fake payment ID
        payment_id = f"CC-{datetime.now().strftime('%Y%m%d')}-{random.randint(10000, 99999)}"
    
    elif payment_method == 'paypal':
        # In a real system, you would redirect to PayPal for payment
        # For our demo, we'll simulate payment success
        payment_id = f"PP-{datetime.now().strftime('%Y%m%d')}-{random.randint(10000, 99999)}"
    
    # Create the order
    order = Order(
        user_id=user_id,
        total_amount=total_amount,
        status='paid',  # For demo purposes, set status to paid
        payment_id=payment_id,
        shipping_address=shipping_address.street_address if shipping_address else None
    )
    
    db.session.add(order)
    db.session.commit()
    
    # Create order items
    for cart_item in cart_items:
        order_item = OrderItem(
            order_id=order.id,
            product_name=cart_item.product_name,
            quantity=cart_item.quantity,
            price=cart_item.price
        )
        db.session.add(order_item)
    
    # Clear the cart
    for cart_item in cart_items:
        db.session.delete(cart_item)
    
    db.session.commit()
    
    # Record user activity
    new_activity = UserActivity(
        user_id=user_id,
        product_name=f"Order #{order.id}",
        activity_type='purchase'
    )
    db.session.add(new_activity)
    db.session.commit()
    
    # For A/B testing analytics
    if ab_testing_manager is not None:
        ab_testing_manager.record_metric(user_id, 'purchase')
    
    # Get recommendations for order confirmation page
    recommendations = []
    try:
        rec_df = recommendation_system.get_recommendations(user_id, top_n=4)
        if not rec_df.empty:
            rec_df['Price'] = np.random.uniform(20, 200, size=len(rec_df))
            recommendations = rec_df.to_dict('records')
    except Exception as e:
        print(f"Error getting recommendations: {e}")
    
    # Calculate estimated delivery date based on shipping method
    today = datetime.now()
    if shipping_method == 'overnight':
        estimated_delivery_date = today + timedelta(days=1)
    elif shipping_method == 'express':
        estimated_delivery_date = today + timedelta(days=3)
    else:
        estimated_delivery_date = today + timedelta(days=5)
    
    # Get user email
    user = Signup.query.get(user_id)
    user_email = user.email if user else "customer@example.com"
    
    # Prepare order items with images
    order_items = []
    for item in order.items:
        product_data = train_data[train_data['Name'] == item.product_name]
        if not product_data.empty:
            product = product_data.iloc[0]
            item_dict = item.to_dict()
            item_dict['image_url'] = product['ImageURL']
            order_items.append(item_dict)
    
    return render_template('order_confirmation.html',
                          order=order,
                          order_items=order_items,
                          shipping_address=shipping_address,
                          payment_method=payment_method,
                          card_type=card_type,
                          card_last_four=card_last_four,
                          payment_id=payment_id,
                          subtotal=subtotal,
                          shipping_cost=shipping_cost,
                          tax=tax,
                          estimated_delivery_date=estimated_delivery_date,
                          user_email=user_email,
                          recommendations=recommendations)

@app.route('/order-history')
@login_required
def order_history():
    """Show order history for current user"""
    user_id = session['user_id']
    
    # Get all orders for the user
    orders = Order.query.filter_by(user_id=user_id).order_by(Order.order_date.desc()).all()
    
    # Get the default shipping address
    default_address = Address.query.filter_by(user_id=user_id, is_default=True).first()
    
    # Enrich orders with additional data
    for order in orders:
        # DON'T assign the whole address object to order.shipping_address
        # Instead, add it as a separate attribute that won't be persisted to the database
        order.address = default_address
        
        # Add images to order items
        for item in order.items:
            product_data = train_data[train_data['Name'] == item.product_name]
            if not product_data.empty:
                product = product_data.iloc[0]
                item.image_url = product['ImageURL']
    
    return render_template('order_history.html', orders=orders)

@app.route('/order/<int:order_id>')
@login_required
def view_order(order_id):
    """View details for a specific order"""
    user_id = session['user_id']
    
    # Get the order, ensuring it belongs to the current user
    order = Order.query.filter_by(id=order_id, user_id=user_id).first_or_404()
    
    # Get shipping address
    shipping_address = Address.query.filter_by(user_id=user_id, is_default=True).first()
    
    # Calculate subtotal, shipping cost, and tax
    subtotal = sum(item.price * item.quantity for item in order.items)
    shipping_cost = order.total_amount - subtotal * 1.08  # Approximate from total
    tax = subtotal * 0.08
    
    # Add images to order items
    for item in order.items:
        product_data = train_data[train_data['Name'] == item.product_name]
        if not product_data.empty:
            product = product_data.iloc[0]
            item.image_url = product['ImageURL']
    
    # For demonstration, create simulated tracking and delivery dates
    order_date = order.order_date
    processing_date = order_date + timedelta(days=1)
    shipping_date = processing_date + timedelta(days=1)
    estimated_delivery_date = shipping_date + timedelta(days=3)
    
    # Generate a fake tracking number
    tracking_number = f"TRACK-{order_id}-{random.randint(10000000, 99999999)}"
    
    # Determine payment method from payment ID
    payment_method = 'credit_card' if order.payment_id and order.payment_id.startswith('CC-') else 'paypal'
    
    # For credit cards, determine card type and last four digits
    card_type = "Credit Card"
    card_last_four = "0000"
    
    if payment_method == 'credit_card':
        card_types = ['Visa', 'MasterCard', 'American Express', 'Discover']
        card_type = random.choice(card_types)
        card_last_four = str(random.randint(1000, 9999))
    
    # For PayPal, extract the payment ID
    payment_id = order.payment_id
    
    return render_template('view_order.html',
                          order=order,
                          shipping_address=shipping_address,
                          subtotal=subtotal,
                          shipping_cost=shipping_cost,
                          tax=tax,
                          tracking_number=tracking_number,
                          processing_date=processing_date,
                          shipping_date=shipping_date,
                          estimated_delivery_date=estimated_delivery_date,
                          payment_method=payment_method,
                          card_type=card_type,
                          card_last_four=card_last_four,
                          payment_id=payment_id)


@app.route('/api/autocomplete', methods=['GET'])
def autocomplete():
    """API endpoint for search term autocomplete suggestions"""
    query = request.args.get('query', '').lower()
    max_results = request.args.get('max', 10, type=int)
    
    if not query or len(query) < 2:
        return jsonify({'suggestions': []})
    
    try:
        # Try to get suggestions from cache first for better performance
        suggestions = []
        if autocomplete_cache:
            suggestions = autocomplete_cache.get_suggestions(query, max_results)
        
        # If we have enough suggestions from cache, return them
        if len(suggestions) >= max_results//2:  # Return if we have at least half the requested results
            return jsonify({'suggestions': suggestions[:max_results]})
        
        # Otherwise, fall back to the regular methods
        product_suggestions = get_product_name_suggestions(query, max_results)
        attribute_suggestions = get_attribute_suggestions(query, max_results)
        query_suggestions = get_popular_query_suggestions(query, max_results)
        
        # Combine and deduplicate suggestions
        all_suggestions = []
        seen = set()
        
        # Add product name suggestions first
        for suggestion in product_suggestions:
            if suggestion.lower() not in seen and len(all_suggestions) < max_results:
                all_suggestions.append(suggestion)
                seen.add(suggestion.lower())
        
        # Add popular query suggestions
        for suggestion in query_suggestions:
            if suggestion.lower() not in seen and len(all_suggestions) < max_results:
                all_suggestions.append(suggestion)
                seen.add(suggestion.lower())
        
        # Add attribute suggestions
        for suggestion in attribute_suggestions:
            if suggestion.lower() not in seen and len(all_suggestions) < max_results:
                all_suggestions.append(suggestion)
                seen.add(suggestion.lower())
        
        return jsonify({'suggestions': all_suggestions[:max_results]})
    except Exception as e:
        print(f"Error in autocomplete: {e}")
        return jsonify({'suggestions': []})
    

def get_product_name_suggestions(query, max_results=10):
    """Get autocomplete suggestions from product names"""
    global train_data
    
    # Filter product names that contain the query
    matching_products = train_data[train_data['Name'].str.lower().str.contains(query, na=False)]
    
    # Sort by relevance (exact match at start is best)
    # Then by popularity (review count)
    results = []
    
    # Exact match at start
    start_matches = matching_products[matching_products['Name'].str.lower().str.startswith(query)]
    for _, product in start_matches.sort_values('ReviewCount', ascending=False).head(max_results).iterrows():
        results.append(product['Name'])
    
    # Contains match
    if len(results) < max_results:
        contains_matches = matching_products[~matching_products['Name'].str.lower().str.startswith(query)]
        for _, product in contains_matches.sort_values('ReviewCount', ascending=False).head(max_results - len(results)).iterrows():
            results.append(product['Name'])
    
    return results

def get_attribute_suggestions(query, max_results=5):
    """Get autocomplete suggestions for common product attributes"""
    global train_data
    
    # Common attributes that might be searched
    attributes = []
    
    # Add brands
    brands = train_data['Brand'].dropna().unique()
    for brand in brands:
        if brand.lower().find(query) >= 0:
            attributes.append(f"{brand} products")
    
    # Add categories/tags if available
    if 'Tags' in train_data.columns:
        all_tags = []
        for tags in train_data['Tags'].dropna():
            all_tags.extend([tag.strip() for tag in tags.split(',')])
        
        tag_counter = Counter(all_tags)
        popular_tags = [tag for tag, _ in tag_counter.most_common(50)]
        
        for tag in popular_tags:
            if tag.lower().find(query) >= 0:
                attributes.append(f"{tag}")
    
    # Add some common phrases that people might search for
    common_phrases = [
        "best rated", "top rated", "high quality", "affordable", "cheap", 
        "premium", "luxury", "discount", "sale", "new arrivals"
    ]
    
    for phrase in common_phrases:
        if phrase.find(query) >= 0:
            attributes.append(phrase)
    
    return attributes[:max_results]

def get_popular_query_suggestions(query, max_results=5):
    """Get autocomplete suggestions from popular past queries"""
    try:
        # Try to get real query history from the database
        user_activities = UserActivity.query.filter(
            UserActivity.product_name.ilike(f"%{query}%"),
            UserActivity.activity_type.in_(['nlp_search', 'search'])
        ).with_entities(UserActivity.product_name, db.func.count(UserActivity.id).label('count'))\
         .group_by(UserActivity.product_name)\
         .order_by(db.desc('count'))\
         .limit(max_results).all()
        
        popular_queries = [activity.product_name for activity in user_activities]
        
        # If we don't have enough results, add some common completions
        if len(popular_queries) < max_results:
            # Common query completions
            common_completions = [
                "for women", "for men", "for kids", "for home", 
                "under $50", "under $100", "review", "best",
                "near me", "online", "discount", "sale"
            ]
            
            words = query.split()
            if len(words) > 0:
                for completion in common_completions:
                    suggestion = f"{query} {completion}"
                    if len(popular_queries) < max_results and suggestion not in popular_queries:
                        popular_queries.append(suggestion)
        
        return popular_queries
    
    except Exception as e:
        print(f"Error getting popular queries: {e}")
        return []
    


# Add this route to your app.py file

@app.route("/voice_search_visualization")
def voice_search_visualization():
    """Show visualization of how the voice search system works"""
    return render_template('voice_search_visualization.html')


if __name__ == '__main__':
    app.run(debug=True)