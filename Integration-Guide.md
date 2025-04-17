# E-commerce Recommendation System Integration Guide

## Overview

This document provides instructions for integrating the AI/ML recommendation system into an existing e-commerce platform. The system includes content-based and personalized recommendations, NLP search, image search, and A/B testing capabilities.

## System Architecture

The recommendation system is structured as a modular API with the following components:

```
/recommendation-system
  /modules
    /recommendation        # Recommendation algorithms
    /search                # NLP and image search capabilities
    /testing               # A/B testing framework
    /utils                 # Utility functions
    /assistant             # AI assistant
  /models                  # Trained model files
  /static/uploads          # For image uploads
  app.py                   # Main Flask API
  migrateApi.py            # Migration Flask API 
  /static/swagger.yaml     # API specification
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB recommended)
- Disk space: 1GB for code + dependencies, plus space for models (5-10GB)

### Step 1: Clone the Repository

```bash
git clone [repository_url]
cd recommendation-system
```

### Step 2: Set Up Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Prepare Data

Place your product data CSV in the `/models` directory as `clean_data.csv` with the following columns:
- `Name` (required): Product name
- `Brand`: Product brand
- `Category`: Product category
- `ImageURL`: Product image URL
- `Rating`: Product rating (1-5)
- `ReviewCount`: Number of reviews
- `Description`: Product description
- `Tags`: Product tags/keywords

### Step 4: Start the API Server

```bash
python app.py
```

The API will be available at http://localhost:5000/api

## API Documentation

The recommendation system exposes a RESTful API with the following key endpoints:

### Authentication

All API requests require an API key in the header:

```
Authorization: Bearer YOUR_API_KEY
```

### Recommendations Endpoints

#### Get Content-Based Recommendations

```
GET /api/recommendations/content?item_name=PRODUCT_NAME&top_n=5
```

**Parameters:**
- `item_name`: Name of the product to get recommendations for
- `top_n` (optional): Number of recommendations to return (default: 5)

#### Get User Recommendations

```
GET /api/recommendations/user?user_id=USER_ID&item_name=PRODUCT_NAME&top_n=5
```

**Parameters:**
- `user_id`: User ID to get recommendations for
- `item_name` (optional): Product name to influence recommendations
- `top_n` (optional): Number of recommendations to return (default: 5)

### Search Endpoints

#### NLP Search

```
GET /api/search/nlp?query=SEARCH_QUERY&top_n=10&user_id=USER_ID
```

**Parameters:**
- `query`: Natural language search query
- `top_n` (optional): Number of results to return (default: 10)
- `user_id` (optional): User ID for personalized results

#### Image Search

```
POST /api/search/image
```

**Form Parameters:**
- `image`: Image file (multipart/form-data)
- `top_n` (optional): Number of results to return (default: 10)

#### Multimodal Search (Image + Text)

```
POST /api/search/multimodal
```

**Form Parameters:**
- `image`: Image file (multipart/form-data)
- `text_query`: Text search query
- `top_n` (optional): Number of results to return (default: 10)
- `image_weight` (optional): Weight for image results (0-100, default: 50)

### User Activity Tracking

```
POST /api/activity
```

**Request Body:**
```json
{
  "user_id": 123,
  "product_name": "Product Name",
  "activity_type": "view"  // view, search, click, purchase, etc.
}
```

### A/B Testing

```
GET /api/ab_testing/assign?user_id=USER_ID&test_id=TEST_ID
```

**Parameters:**
- `user_id`: User ID to assign to variants
- `test_id` (optional): Specific test ID to assign

## Integration Methods

There are three ways to integrate this recommendation system:

### 1. API Integration (Recommended)

The simplest approach is to call the API endpoints from your e-commerce platform:

```javascript
// JavaScript example
async function getRecommendations(productName) {
  const response = await fetch(
    `http://your-api-url/api/recommendations/content?item_name=${encodeURIComponent(productName)}`,
    {
      headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
      }
    }
  );
  return response.json();
}
```

```php
// PHP example
function getRecommendations($productName) {
  $url = "http://your-api-url/api/recommendations/content?item_name=" . urlencode($productName);
  $options = [
    'http' => [
      'header' => "Authorization: Bearer YOUR_API_KEY\r\n"
    ]
  ];
  $context = stream_context_create($options);
  $result = file_get_contents($url, false, $context);
  return json_decode($result, true);
}
```

### 2. Library/Package Integration

For tighter integration, you can install the recommendation system as a Python package:

1. Create a setup.py file in the root directory
2. Install the package in your e-commerce project
3. Import and use the modules directly

### 3. Full Migration

If your e-commerce platform is also Python-based, you can fully integrate the code by:

1. Copying the modules directory to your project
2. Importing and using the classes directly

## Integration Examples

### Product Detail Page

Add recommendations to product detail pages:

```html
<div class="product-recommendations">
  <h3>You Might Also Like</h3>
  <div id="recommendations-container"></div>
</div>

<script>
  const productName = "Current Product Name";
  
  // Fetch recommendations when page loads
  fetch(`http://your-api-url/api/recommendations/content?item_name=${encodeURIComponent(productName)}`, {
    headers: {
      'Authorization': 'Bearer YOUR_API_KEY'
    }
  })
  .then(response => response.json())
  .then(data => {
    const container = document.getElementById('recommendations-container');
    
    if (data.status === 'success' && data.recommendations.length > 0) {
      // Display recommendations
      container.innerHTML = data.recommendations.map(product => `
        <div class="recommendation-item">
          <a href="/product/${encodeURIComponent(product.Name)}">
            <img src="${product.ImageURL}" alt="${product.Name}">
            <h4>${product.Name}</h4>
            <div>${product.Brand}</div>
            <div>${product.Rating} ★ (${product.ReviewCount})</div>
          </a>
        </div>
      `).join('');
    } else {
      container.innerHTML = '<p>No recommendations available</p>';
    }
  })
  .catch(error => {
    console.error('Error fetching recommendations:', error);
  });
</script>
```

### Search Box with NLP

Enhance your search box with NLP capabilities:

```html
<form id="search-form">
  <input type="text" id="search-input" placeholder="Search products...">
  <button type="submit">Search</button>
</form>

<div id="search-results"></div>

<script>
  document.getElementById('search-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const query = document.getElementById('search-input').value;
    const resultsContainer = document.getElementById('search-results');
    
    resultsContainer.innerHTML = '<p>Searching...</p>';
    
    // Call NLP search API
    fetch(`http://your-api-url/api/search/nlp?query=${encodeURIComponent(query)}`, {
      headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
      }
    })
    .then(response => response.json())
    .then(data => {
      if (data.status === 'success' && data.results.length > 0) {
        resultsContainer.innerHTML = data.results.map(product => `
          <div class="product-item">
            <a href="/product/${encodeURIComponent(product.Name)}">
              <img src="${product.ImageURL}" alt="${product.Name}">
              <div>
                <h3>${product.Name}</h3>
                <div>${product.Brand}</div>
                <div>${product.Rating} ★ (${product.ReviewCount})</div>
              </div>
            </a>
          </div>
        `).join('');
      } else {
        resultsContainer.innerHTML = '<p>No results found</p>';
      }
    })
    .catch(error => {
      console.error('Error searching products:', error);
      resultsContainer.innerHTML = '<p>Error searching products</p>';
    });
  });
</script>
```

### Track User Activity

Track user interactions to improve personalized recommendations:

```javascript
// Track product view
function trackProductView(userId, productName) {
  if (!userId) return; // Skip if user is not logged in
  
  fetch('http://your-api-url/api/activity', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      user_id: userId,
      product_name: productName,
      activity_type: 'view'
    })
  })
  .catch(error => console.error('Error tracking activity:', error));
}

// Track product purchase
function trackProductPurchase(userId, productName) {
  if (!userId) return;
  
  fetch('http://your-api-url/api/activity', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      user_id: userId,
      product_name: productName,
      activity_type: 'purchase'
    })
  })
  .catch(error => console.error('Error tracking purchase:', error));
}
```

## Testing the Integration

To verify your integration is working correctly:

1. Start the recommendation API server
2. Make test API calls to each endpoint
3. Verify recommendations appear on product pages
4. Check that search results are relevant
5. Confirm user activity is being tracked

## Common Issues and Solutions

### API Connection Issues

**Problem**: Unable to connect to the API
**Solution**: 
- Verify the API server is running
- Check network connectivity and firewall settings
- Ensure the API URL is correct

### CORS Errors

**Problem**: Browser blocks API requests due to CORS policy
**Solution**:
- Configure the API to allow requests from your domain:
  ```python
  from flask_cors import CORS
  CORS(app, resources={r"/api/*": {"origins": "your-domain.com"}})
  ```

### Authentication Errors

**Problem**: API returns 401 Unauthorized
**Solution**:
- Verify the API key is correct
- Ensure the Authorization header is properly formatted

### Slow Response Times

**Problem**: API responses are slow
**Solution**:
- Implement caching for recommendations
- Reduce the number of requested items (top_n parameter)
- Consider deploying the API closer to your application servers

## Customization Options

### Recommendation Weights

You can adjust the weights of different recommendation algorithms via the A/B testing framework:

```python
# Example weight configurations
weights = {
    "content_based": 0.6,  # Higher weight for content similarity
    "collaborative": 0.2,  # Medium weight for collaborative filtering
    "neural": 0.2          # Lower weight for neural network
}
```

### Recommendation Diversity

To increase diversity in recommendations, use the endpoint with a higher top_n value and then filter results:

```javascript
// Request more recommendations than needed
fetch(`http://your-api-url/api/recommendations/user?user_id=123&top_n=20`, {
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY'
  }
})
.then(response => response.json())
.then(data => {
  if (data.status === 'success') {
    // Filter for diversity (e.g., limit to 2 products per brand)
    const diverseRecs = [];
    const brandCounts = {};
    
    for (const product of data.recommendations) {
      if (!brandCounts[product.Brand]) {
        brandCounts[product.Brand] = 0;
      }
      
      if (brandCounts[product.Brand] < 2) {
        diverseRecs.push(product);
        brandCounts[product.Brand]++;
      }
      
      if (diverseRecs.length >= 10) break;
    }
    
    // Display diverse recommendations
    displayRecommendations(diverseRecs);
  }
});
```

## Conclusion

This recommendation system provides powerful AI capabilities to enhance your e-commerce platform. By following this guide, you can integrate personalized recommendations, advanced search, and A/B testing into your existing application.

For additional assistance, please refer to the Swagger documentation (swagger.yaml) or contact the system developer.

---

## Appendix: System Components

### Key Modules

- `enhanced_recommendation_system.py`: Combines multiple recommendation approaches
- `nlp_search.py`: Natural language processing for semantic search
- `image_feature_extractor.py`: Visual similarity search
- `multimodal_search.py`: Combined text and image search
- `ab_testing.py`: A/B testing framework for optimization

### Required Dependencies

- Flask: Web framework for the API
- TensorFlow: For neural recommendation models
- scikit-learn: For traditional ML algorithms
- sentence-transformers: For NLP capabilities
- Pillow: For image processing
- pandas & numpy: For data handling

For the complete list, see `requirements.txt`.