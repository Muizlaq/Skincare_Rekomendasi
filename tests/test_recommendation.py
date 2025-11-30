#!/usr/bin/env python3
"""
Test script to debug recommendation flow with k_value
"""

from config import DatabaseConfig
from models import UserPreference
from recommender import SkincareRecommender

def test_recommendation_flow():
    """Test the complete recommendation flow"""
    try:
        # Get a user with preferences
        query = "SELECT user_id FROM user_preferences LIMIT 1"
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if not result:
            print("No user preferences found")
            return
            
        user_id = result[0]['user_id']
        print(f"Testing with user_id: {user_id}")
        
        # Get user preferences
        preferences = UserPreference.get_by_user_id(user_id)
        if not preferences:
            print("Could not get user preferences")
            return
            
        print(f"User preferences k_value: {preferences.get('k_value', 'NOT SET')}")
        
        # Test with different k_values
        test_k_values = [3, 5, 7, 10]
        
        recommender = SkincareRecommender()
        
        for k_val in test_k_values:
            print(f"\n--- Testing with k_value = {k_val} ---")
            
            # Get recommendations with specific k_value
            recommendations = recommender.get_recommendations(preferences, k_value=k_val)
            
            print(f"Number of recommendations returned: {len(recommendations)}")
            
            if recommendations:
                print("First 3 products:")
                for i, rec in enumerate(recommendations[:3]):
                    product = rec['product']
                    print(f"  {i+1}. {product['name']} - {product['brand']}")
            else:
                print("No recommendations returned")
                
        # Test with user's saved k_value
        print(f"\n--- Testing with user's saved k_value = {preferences.get('k_value', 3)} ---")
        user_k_value = preferences.get('k_value', 3)
        recommendations = recommender.get_recommendations(preferences, k_value=user_k_value)
        print(f"Number of recommendations returned: {len(recommendations)}")
        
    except Exception as e:
        print(f"Error in recommendation flow test: {e}")
        import traceback
        traceback.print_exc()

def test_app_route_simulation():
    """Simulate the app route logic"""
    try:
        print("\n=== Simulating App Route Logic ===")
        
        # Get a user with preferences
        query = "SELECT user_id FROM user_preferences LIMIT 1"
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if not result:
            print("No user preferences found")
            return
            
        user_id = result[0]['user_id']
        
        # Simulate the app.py logic
        preferences = UserPreference.get_by_user_id(user_id)
        if not preferences:
            print("Could not get user preferences")
            return
            
        print(f"Raw preferences from database: {preferences}")
        
        # Convert to dict like in app.py
        preferences_dict = dict(preferences)
        
        # Create rentang_harga based on budget range (like in app.py)
        budget_max = preferences_dict.get('budget_max', 1000000)
        if budget_max <= 50000:
            preferences_dict['rentang_harga'] = '0-50000'
        elif budget_max <= 100000:
            preferences_dict['rentang_harga'] = '50000-100000'
        elif budget_max <= 200000:
            preferences_dict['rentang_harga'] = '100000-200000'
        elif budget_max <= 500000:
            preferences_dict['rentang_harga'] = '200000-500000'
        else:
            preferences_dict['rentang_harga'] = '500000+'
            
        # Ensure all required fields are present with defaults (like in app.py)
        required_fields = {
            'frekuensi_pemakaian': preferences_dict.get('frekuensi_pemakaian', 'harian'),
            'bahan_aktif_efektif': preferences_dict.get('bahan_aktif_efektif', 'tidak_tahu'),
            'preferensi_produk': preferences_dict.get('preferensi_produk', 'semua'),
            'kata_kunci': preferences_dict.get('kata_kunci', '')
        }
        
        # Add missing fields to preferences
        for field, default_value in required_fields.items():
            if field not in preferences_dict or preferences_dict[field] is None:
                preferences_dict[field] = default_value
                
        print(f"Processed preferences: {preferences_dict}")
        
        # Get recommendations with user's preferred k_value (like in app.py)
        user_k_value = preferences_dict.get('k_value', 3)
        print(f"Using k_value: {user_k_value}")
        
        recommender = SkincareRecommender()
        recommendations = recommender.get_recommendations(preferences_dict, k_value=user_k_value)
        
        print(f"Final recommendations count: {len(recommendations)}")
        
        if recommendations:
            print("Recommendations:")
            for i, rec in enumerate(recommendations):
                product = rec['product']
                print(f"  {i+1}. {product['name']} - {product['brand']}")
        
    except Exception as e:
        print(f"Error in app route simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Recommendation Flow with K-Value...")
    print()
    
    test_recommendation_flow()
    test_app_route_simulation()
    
    print("\nTest complete!")