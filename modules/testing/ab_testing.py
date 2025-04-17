from datetime import datetime, timedelta
import json
import os
import random
import uuid
from collections import defaultdict
import numpy as np
import pandas as pd


class ABTest:
    """
    Class representing an A/B test for recommendation system variants
    """
    def __init__(self, test_id, name, description, variants, start_date, end_date, status="active"):
        """
        Initialize an A/B test
        
        Parameters:
        -----------
        test_id : str
            Unique identifier for the test
        name : str
            Human-readable name for the test
        description : str
            Detailed description of the test
        variants : list of dict
            List of variant configurations, each with:
                - id: unique identifier
                - name: human-readable name
                - weights: dict of weights for recommendation algorithms
        start_date : datetime
            Start date of the test
        end_date : datetime
            End date of the test
        status : str
            Status of the test (active, completed, paused)
        """
        self.test_id = test_id
        self.name = name
        self.description = description
        self.variants = variants
        self.start_date = start_date
        self.end_date = end_date
        self.status = status
        
        # Initialize metrics for each variant
        self.metrics = {variant['id']: {
            'impressions': 0,
            'clicks': 0,
            'purchases': 0,
            'ratings': [],
            'feedback_count': 0,
            'avg_rating': 0.0,
            'ctr': 0.0,
            'conversion_rate': 0.0,
        } for variant in variants}
    
    def to_dict(self):
        """Convert the test object to a dictionary for serialization"""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'variants': self.variants,
            'start_date': self.start_date.isoformat() if isinstance(self.start_date, datetime) else self.start_date,
            'end_date': self.end_date.isoformat() if isinstance(self.end_date, datetime) else self.end_date,
            'status': self.status,
            'metrics': self.metrics
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a test object from a dictionary"""
        start_date = data['start_date']
        end_date = data['end_date']
        
        # Convert string dates to datetime objects
        if isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date)
            except (ValueError, AttributeError):
                try:
                    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    # Try without microseconds
                    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
        
        if isinstance(end_date, str):
            try:
                end_date = datetime.fromisoformat(end_date)
            except (ValueError, AttributeError):
                try:
                    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    # Try without microseconds
                    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
        
        # Create test object
        test = cls(
            test_id=data['test_id'],
            name=data['name'],
            description=data['description'],
            variants=data['variants'],
            start_date=start_date,
            end_date=end_date,
            status=data['status']
        )
        
        # Add metrics if available
        if 'metrics' in data:
            test.metrics = data['metrics']
        
        return test
    
    def update_metrics(self, variant_id, metric_type, value=None):
        """
        Update metrics for a specific variant
        
        Parameters:
        -----------
        variant_id : str
            Identifier of the variant to update
        metric_type : str
            Type of metric to update (impressions, clicks, purchases, ratings)
        value : float, optional
            Value for the metric (only used for ratings)
        """
        if variant_id not in self.metrics:
            return False
        
        if metric_type == 'impression':
            self.metrics[variant_id]['impressions'] += 1
        elif metric_type == 'click':
            self.metrics[variant_id]['clicks'] += 1
            # Update CTR
            if self.metrics[variant_id]['impressions'] > 0:
                self.metrics[variant_id]['ctr'] = self.metrics[variant_id]['clicks'] / self.metrics[variant_id]['impressions']
        elif metric_type == 'purchase':
            self.metrics[variant_id]['purchases'] += 1
            # Update conversion rate
            if self.metrics[variant_id]['clicks'] > 0:
                self.metrics[variant_id]['conversion_rate'] = self.metrics[variant_id]['purchases'] / self.metrics[variant_id]['clicks']
        elif metric_type == 'rating' and value is not None:
            self.metrics[variant_id]['ratings'].append(float(value))
            self.metrics[variant_id]['feedback_count'] += 1
            self.metrics[variant_id]['avg_rating'] = sum(self.metrics[variant_id]['ratings']) / len(self.metrics[variant_id]['ratings'])
        
        return True
    
    def get_best_variant(self, primary_metric='avg_rating'):
        """
        Get the best-performing variant based on the specified metric
        
        Parameters:
        -----------
        primary_metric : str
            Metric to use for comparison (avg_rating, ctr, conversion_rate)
            
        Returns:
        --------
        dict
            The best-performing variant configuration
        """
        # Filter out variants with no data
        valid_variants = {var_id: metrics for var_id, metrics in self.metrics.items() 
                          if metrics['feedback_count'] > 0}
        
        if not valid_variants:
            # If no variants have data, return the first variant
            return next((v for v in self.variants if v['id'] == list(self.metrics.keys())[0]), None)
        
        # Find the best variant based on the primary metric
        best_variant_id = max(valid_variants.items(), key=lambda x: x[1][primary_metric])[0]
        
        # Return the full variant configuration
        return next((v for v in self.variants if v['id'] == best_variant_id), None)
    
    def is_active(self):
        """Check if the test is currently active"""
        if self.status != 'active':
            return False
        
        now = datetime.now()
        
        # Ensure both dates are datetime objects
        start = self.start_date
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except (ValueError, AttributeError):
                try:
                    start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        
        end = self.end_date
        if isinstance(end, str):
            try:
                end = datetime.fromisoformat(end)
            except (ValueError, AttributeError):
                try:
                    end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
        
        return start <= now <= end


class ABTestingManager:
    """
    Manager class for A/B tests in recommendation systems
    """
    def __init__(self, model_dir='models'):
        """
        Initialize the A/B testing manager
        
        Parameters:
        -----------
        model_dir : str
            Directory to store test data
        """
        self.model_dir = model_dir
        self.tests_path = os.path.join(model_dir, 'ab_tests.json')
        self.user_assignments_path = os.path.join(model_dir, 'ab_test_user_assignments.json')
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing tests and user assignments
        self.tests = self._load_tests()
        self.user_assignments = self._load_user_assignments()
    
    def _load_tests(self):
        """Load A/B tests from storage"""
        try:
            with open(self.tests_path, 'r') as f:
                tests_data = json.load(f)
                return {test_id: ABTest.from_dict(test_data) 
                        for test_id, test_data in tests_data.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _load_user_assignments(self):
        """Load user test assignments from storage"""
        try:
            with open(self.user_assignments_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_tests(self):
        """Save A/B tests to storage"""
        tests_data = {test_id: test.to_dict() for test_id, test in self.tests.items()}
        with open(self.tests_path, 'w') as f:
            json.dump(tests_data, f, indent=2)
    
    def _save_user_assignments(self):
        """Save user test assignments to storage"""
        with open(self.user_assignments_path, 'w') as f:
            json.dump(self.user_assignments, f, indent=2)
    
    def create_test(self, name, description, variants, start_date, end_date, status="active"):
        """
        Create a new A/B test
        
        Parameters:
        -----------
        name : str
            Human-readable name for the test
        description : str
            Detailed description of the test
        variants : list of dict
            List of variant configurations
        start_date : datetime
            Start date of the test
        end_date : datetime
            End date of the test
        status : str
            Status of the test (active, completed, paused)
            
        Returns:
        --------
        str
            ID of the created test
        """
        test_id = str(uuid.uuid4())
        
        # Make sure each variant has an ID
        for i, variant in enumerate(variants):
            if 'id' not in variant:
                variant['id'] = f"variant_{i+1}"
        
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variants=variants,
            start_date=start_date,
            end_date=end_date,
            status=status
        )
        
        self.tests[test_id] = test
        self._save_tests()
        
        return test_id
    
    def get_test(self, test_id):
        """Get a test by ID"""
        return self.tests.get(test_id)
    
    def get_active_tests(self):
        """Get all currently active tests"""
        return {test_id: test for test_id, test in self.tests.items() 
                if test.is_active()}
    
    def get_all_tests(self):
        """Get all tests"""
        return self.tests
    
    def update_test(self, test_id, **kwargs):
        """Update a test's properties"""
        if test_id not in self.tests:
            return False
        
        test = self.tests[test_id]
        
        for key, value in kwargs.items():
            if hasattr(test, key):
                setattr(test, key, value)
        
        self._save_tests()
        return True
    
    def delete_test(self, test_id):
        """Delete a test"""
        if test_id in self.tests:
            del self.tests[test_id]
            
            # Also remove user assignments for this test
            for user_id in list(self.user_assignments.keys()):
                if test_id in self.user_assignments[user_id]:
                    del self.user_assignments[user_id][test_id]
            
            self._save_tests()
            self._save_user_assignments()
            return True
        
        return False
    
    def assign_user_to_variant(self, user_id, test_id=None):
        """
        Assign a user to a variant of an A/B test
        
        Parameters:
        -----------
        user_id : int
            User ID to assign
        test_id : str, optional
            Specific test ID to assign the user to.
            If None, assigns to all active tests.
            
        Returns:
        --------
        dict
            Dictionary mapping test IDs to assigned variant IDs
        """
        user_id = str(user_id)  # Ensure user_id is a string for JSON serialization
        
        # Initialize user assignments if not already present
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        
        # Determine which tests to assign
        tests_to_assign = {}
        if test_id:
            if test_id in self.tests and self.tests[test_id].is_active():
                tests_to_assign[test_id] = self.tests[test_id]
        else:
            tests_to_assign = self.get_active_tests()
        
        # Perform assignments
        for tid, test in tests_to_assign.items():
            # Only assign if not already assigned
            if tid not in self.user_assignments[user_id]:
                # Randomly select a variant
                variant = random.choice(test.variants)
                self.user_assignments[user_id][tid] = variant['id']
        
        self._save_user_assignments()
        
        # Return the assignments
        return {tid: var_id for tid, var_id in self.user_assignments[user_id].items() 
                if tid in tests_to_assign}
    
    def get_user_variant(self, user_id, test_id):
        """
        Get the variant assigned to a user for a specific test
        
        Parameters:
        -----------
        user_id : int
            User ID
        test_id : str
            Test ID
            
        Returns:
        --------
        dict
            Variant configuration if assigned, None otherwise
        """
        user_id = str(user_id)
        
        # Check if user is assigned to this test
        if (user_id not in self.user_assignments or 
            test_id not in self.user_assignments[user_id]):
            return None
        
        # Get the assigned variant ID
        variant_id = self.user_assignments[user_id][test_id]
        
        # Return the full variant configuration
        if test_id in self.tests:
            for variant in self.tests[test_id].variants:
                if variant['id'] == variant_id:
                    return variant
        
        return None
    
    def get_user_weights(self, user_id):
        """
        Get the recommendation weights for a user based on their assigned variants
        
        Parameters:
        -----------
        user_id : int
            User ID
            
        Returns:
        --------
        dict
            Dictionary of weights for recommendation algorithms
        """
        user_id = str(user_id)
        
        # Default weights if no active tests or user not assigned
        default_weights = {
            'content_based': 0.3,
            'collaborative': 0.2,
            'neural': 0.5
        }
        
        # If no active tests or user not assigned, return default weights
        active_tests = self.get_active_tests()
        if not active_tests or user_id not in self.user_assignments:
            return default_weights
        
        # Get all user's variants from active tests
        user_variants = []
        for test_id, test in active_tests.items():
            if test_id in self.user_assignments[user_id]:
                variant_id = self.user_assignments[user_id][test_id]
                for variant in test.variants:
                    if variant['id'] == variant_id:
                        user_variants.append(variant)
        
        # If no variants found, return default weights
        if not user_variants:
            return default_weights
        
        # Use the weights from the first active test's variant
        # In a more complex system, you might want to combine weights from multiple tests
        return user_variants[0].get('weights', default_weights)
    
    def record_metric(self, user_id, metric_type, value=None):
        """
        Record a metric for the user's assigned variants
        
        Parameters:
        -----------
        user_id : int
            User ID
        metric_type : str
            Type of metric (impression, click, purchase, rating)
        value : float, optional
            Value for the metric (only used for ratings)
            
        Returns:
        --------
        bool
            True if the metric was recorded, False otherwise
        """
        user_id = str(user_id)
        
        # If user not assigned to any tests, return False
        if user_id not in self.user_assignments:
            return False
        
        # Record metric for all active tests the user is assigned to
        updated = False
        for test_id, variant_id in self.user_assignments[user_id].items():
            if test_id in self.tests and self.tests[test_id].is_active():
                if self.tests[test_id].update_metrics(variant_id, metric_type, value):
                    updated = True
        
        if updated:
            self._save_tests()
        
        return updated
    
    def auto_optimize_weights(self, recommendation_system, primary_metric='avg_rating'):
        """
        Automatically optimize weights based on test results
        
        Parameters:
        -----------
        recommendation_system : EnhancedRecommendationSystem
            Recommendation system object to update weights
        primary_metric : str
            Metric to use for optimization (avg_rating, ctr, conversion_rate)
            
        Returns:
        --------
        dict
            Dictionary of optimized weights
        """
        active_tests = self.get_active_tests()
        
        # If no active tests, return current weights
        if not active_tests:
            return recommendation_system.ensemble_weights
        
        # Find the best variant from each active test
        best_variants = [test.get_best_variant(primary_metric) for test in active_tests.values()]
        best_variants = [v for v in best_variants if v is not None]
        
        # If no best variants found, return current weights
        if not best_variants:
            return recommendation_system.ensemble_weights
        
        # Get weights from the best variant
        # In a more sophisticated system, you might combine weights from multiple tests
        best_weights = best_variants[0].get('weights', {})
        
        # If weights are empty, return current weights
        if not best_weights:
            return recommendation_system.ensemble_weights
        
        # Update recommendation system weights
        recommendation_system.set_ensemble_weights(
            content_based=best_weights.get('content_based', 0.3),
            collaborative=best_weights.get('collaborative', 0.2),
            neural=best_weights.get('neural', 0.5)
        )
        
        return best_weights

    def generate_report(self, test_id):
        """
        Generate a comprehensive report for a test
        
        Parameters:
        -----------
        test_id : str
            Test ID
            
        Returns:
        --------
        dict
            Dictionary containing report data
        """
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        
        # Calculate statistical significance for ratings
        variant_ratings = {var_id: metrics['ratings'] for var_id, metrics in test.metrics.items() 
                           if len(metrics['ratings']) > 0}
        
        significance_results = {}
        if len(variant_ratings) >= 2:
            try:
                from scipy import stats
                
                variant_pairs = [(id1, id2) for i, id1 in enumerate(variant_ratings.keys()) 
                                for id2 in list(variant_ratings.keys())[i+1:]]
                
                for id1, id2 in variant_pairs:
                    ratings1 = variant_ratings[id1]
                    ratings2 = variant_ratings[id2]
                    
                    if len(ratings1) > 0 and len(ratings2) > 0:
                        try:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(ratings1, ratings2, equal_var=False)
                            significance_results[f"{id1}_vs_{id2}"] = {
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'confidence': (1 - p_value) * 100
                            }
                        except Exception as e:
                            # Handle errors in statistical testing
                            significance_results[f"{id1}_vs_{id2}"] = {
                                'error': str(e)
                            }
            except ImportError:
                # Handle case where scipy is not available
                significance_results = {
                    'error': 'Statistical testing requires scipy library, which is not available.'
                }
        
        # Calculate lift for each variant compared to control
        lift_results = {}
        if len(test.variants) > 1:
            # Assume the first variant is the control
            control_id = test.variants[0]['id']
            control_metrics = test.metrics[control_id]
            
            for variant in test.variants[1:]:
                var_id = variant['id']
                var_metrics = test.metrics[var_id]
                
                lift = {}
                
                # Calculate lift for each metric
                if control_metrics['avg_rating'] > 0 and var_metrics['avg_rating'] > 0:
                    lift['avg_rating'] = ((var_metrics['avg_rating'] / control_metrics['avg_rating']) - 1) * 100
                
                if control_metrics['ctr'] > 0 and var_metrics['ctr'] > 0:
                    lift['ctr'] = ((var_metrics['ctr'] / control_metrics['ctr']) - 1) * 100
                
                if control_metrics['conversion_rate'] > 0 and var_metrics['conversion_rate'] > 0:
                    lift['conversion_rate'] = ((var_metrics['conversion_rate'] / control_metrics['conversion_rate']) - 1) * 100
                
                lift_results[var_id] = lift
        
        # Create report data
        report = {
            'test': test.to_dict(),
            'significance': significance_results,
            'lift': lift_results,
            'generated_at': datetime.now().isoformat()
        }
        
        return report