import pandas as pd
import numpy as np
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_product_data(file_path):
    """
    Load product data from either CSV or JSON file
    
    Parameters:
    -----------
    file_path : str
        Path to the product data file (.csv or .json)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing product data
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        # Determine file type by extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            # Load CSV file
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} products from CSV: {file_path}")
            # Reset index to ensure continuous integers starting from 0
            data = data.reset_index(drop=True)
            return data
            
        elif file_extension == '.json':
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Check if the JSON has a 'products' key (common format)
            if 'products' in json_data:
                data = pd.DataFrame(json_data['products'])
            # If it's a list, convert directly to DataFrame
            elif isinstance(json_data, list):
                data = pd.DataFrame(json_data)
            # If it's a dictionary without 'products' key, try to convert directly
            else:
                data = pd.DataFrame([json_data])
            
            # Reset index to ensure continuous integers starting from 0
            data = data.reset_index(drop=True)
            
            logger.info(f"Loaded {len(data)} products from JSON: {file_path}")
            return data
            
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()
    
    
def save_product_data(data, file_path, format='json'):
    """
    Save product data to file
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing product data
    file_path : str
        Path to save the data file
    format : str
        Format to save ('json' or 'csv')
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format.lower() == 'csv':
            data.to_csv(file_path, index=False)
            logger.info(f"Saved {len(data)} products to CSV: {file_path}")
            return True
            
        elif format.lower() == 'json':
            # Convert DataFrame to JSON
            if 'id' in data.columns:
                # Use 'id' column as the record identifier if it exists
                json_data = {"products": data.to_dict(orient='records')}
            else:
                # Standard list of records
                json_data = {"products": data.to_dict(orient='records')}
                
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved {len(data)} products to JSON: {file_path}")
            return True
            
        else:
            logger.error(f"Unsupported format: {format}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        return False


def convert_csv_to_json(csv_path, json_path):
    """
    Convert a CSV file to JSON format
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    json_path : str
        Path to save the JSON file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        data = load_product_data(csv_path)
        return save_product_data(data, json_path, format='json')
    except Exception as e:
        logger.error(f"Error converting CSV to JSON: {e}")
        return False