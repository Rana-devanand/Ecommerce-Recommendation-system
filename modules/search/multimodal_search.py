import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class MultimodalSearch:
    def __init__(self, image_search_extractor, nlp_search):
        """
        Initialize multimodal search with image search and NLP search components
        
        Parameters:
        -----------
        image_search_extractor : ImageFeatureExtractor
            Instance of the image feature extraction class
        nlp_search : NLPSearch
            Instance of the NLP search class
        """
        self.image_search_extractor = image_search_extractor
        self.nlp_search = nlp_search
        self.product_data = self.nlp_search.product_data
        
    def search(self, image_data, text_query, top_n=10, weight_image=0.5, weight_text=0.5):
        """
        Perform multimodal search using both image and text
        
        Parameters:
        -----------
        image_data : bytes
            Binary image data from uploaded file
        text_query : str
            Text query to refine image search
        top_n : int
            Number of results to return
        weight_image : float
            Weight for image search results (0-1)
        weight_text : float
            Weight for text search results (0-1)
            
        Returns:
        --------
        tuple
            (pandas DataFrame of search results, dict of query info)
        """
        # Normalize weights to sum to 1
        total_weight = weight_image + weight_text
        weight_image = weight_image / total_weight
        weight_text = weight_text / total_weight
        
        # Get image search results
        image_results = self.image_search_extractor.find_similar_products(image_data, top_n=top_n*2)
        
        # Process text query to understand attributes and intent
        query_info = self.nlp_search.process_query(text_query)
        
        # Extract specific attributes from the text query
        color_match = re.search(r'in\s+(\w+)\s+color', text_query.lower())
        desired_color = color_match.group(1) if color_match else None
        
        price_match = re.search(r'under\s+\$?(\d+)', text_query.lower())
        max_price = int(price_match.group(1)) if price_match else None
        
        # Get text search results
        text_results, _ = self.nlp_search.enhanced_search(text_query, top_n=top_n*2)
        
        # If either search returned no results, return the other
        if image_results.empty:
            return text_results, query_info
        if text_results.empty:
            return image_results, query_info
        
        # Combine results with weights
        all_products = set(image_results['Name']).union(set(text_results['Name']))
        combined_scores = {}
        
        # Process image results
        for _, row in image_results.iterrows():
            product_name = row['Name']
            combined_scores[product_name] = {
                'image_score': row['similarity'],
                'text_score': 0,
                'data': row
            }
        
        # Process text results
        for _, row in text_results.iterrows():
            product_name = row['Name']
            if product_name in combined_scores:
                combined_scores[product_name]['text_score'] = row['similarity']
            else:
                combined_scores[product_name] = {
                    'image_score': 0,
                    'text_score': row['similarity'],
                    'data': row
                }
        
        # Calculate final scores
        final_scores = []
        for product_name, scores in combined_scores.items():
            final_score = (weight_image * scores['image_score']) + (weight_text * scores['text_score'])
            
            # Apply bonuses for specific attributes from text query
            product_data = scores['data']
            
            # Color bonus
            if desired_color and 'Tags' in product_data and pd.notna(product_data['Tags']):
                if desired_color.lower() in product_data['Tags'].lower():
                    final_score *= 1.2  # 20% boost for matching color
            
            # Price penalty
            if max_price and 'Price' in product_data and pd.notna(product_data['Price']):
                if product_data['Price'] > max_price:
                    final_score *= 0.5  # 50% penalty for exceeding price
            
            product_dict = product_data.to_dict()
            product_dict['similarity'] = final_score
            product_dict['image_similarity'] = scores['image_score']
            product_dict['text_similarity'] = scores['text_score']
            
            final_scores.append(product_dict)
        
        # Sort by final score and return top_n
        results_df = pd.DataFrame(final_scores)
        results_df = results_df.sort_values('similarity', ascending=False).head(top_n)
        
        return results_df, query_info