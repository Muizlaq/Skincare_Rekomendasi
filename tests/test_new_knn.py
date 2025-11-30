#!/usr/bin/env python3
"""
Test script untuk menguji implementasi KNN yang baru
Memverifikasi bahwa K-value yang berbeda menghasilkan hasil yang berbeda
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import SkincareRecommender
from models import UserPreference
import mysql.connector
from config import Config

def test_knn_with_different_k_values():
    """Test KNN algorithm dengan K-value yang berbeda"""
    print("=== TESTING NEW KNN IMPLEMENTATION ===\n")
    
    # Initialize recommender
    recommender = SkincareRecommender()
    
    # Sample user preferences
    test_preferences = {
        'kondisi_kulit': 'berminyak',  # Changed from 'jenis_kulit' to 'kondisi_kulit'
        'masalah_kulit': 'jerawat',
        'preferensi_produk': 'semua',  # Added required field
        'budget_min': 50000,
        'budget_max': 200000,
        'rentang_harga': '50000-200000'
    }
    
    print("Test Preferences:")
    for key, value in test_preferences.items():
        print(f"  {key}: {value}")
    print()
    
    # Test dengan K-value yang berbeda
    k_values_to_test = [3, 5, 7, 10]
    
    results = {}
    
    for k in k_values_to_test:
        print(f"--- Testing dengan K = {k} ---")
        
        # Get recommendations dengan K-value tertentu
        recommendations = recommender.get_recommendations(
            preferences=test_preferences,
            max_recommendations=10,
            k_value=k
        )
        
        print(f"Total recommendations: {len(recommendations)}")
        
        # Show top 3 (yang akan ditampilkan di dashboard)
        print("Top 3 recommendations (untuk dashboard):")
        for i, rec in enumerate(recommendations[:3], 1):
            product = rec['product']
            knn_score = rec['knn_score']
            content_sim = rec['content_similarity']
            
            print(f"  {i}. {product['name']}")  # Changed from 'nama_produk' to 'name'
            print(f"     KNN Score: {knn_score:.4f}")
            print(f"     Content Similarity: {content_sim:.4f}")
            print(f"     Price: Rp {product['price']:,}")  # Changed from 'harga' to 'price'
            print(f"     Explanation: {rec['explanation']}")
            print()
        
        # Store results untuk comparison
        results[k] = {
            'top_3_products': [rec['product']['name'] for rec in recommendations[:3]],  # Changed field name
            'top_3_scores': [rec['knn_score'] for rec in recommendations[:3]],
            'all_recommendations': recommendations
        }
        
        print("-" * 50)
        print()
    
    # Compare results
    print("=== COMPARISON ANALYSIS ===")
    print("Checking if different K values produce different results...\n")
    
    # Compare top 3 products for each K
    k_list = list(results.keys())
    different_results = False
    
    for i in range(len(k_list)):
        for j in range(i+1, len(k_list)):
            k1, k2 = k_list[i], k_list[j]
            
            top3_k1 = results[k1]['top_3_products']
            top3_k2 = results[k2]['top_3_products']
            
            if top3_k1 != top3_k2:
                different_results = True
                print(f"✅ K={k1} vs K={k2}: Different top 3 products")
                print(f"   K={k1}: {top3_k1}")
                print(f"   K={k2}: {top3_k2}")
            else:
                print(f"⚠️  K={k1} vs K={k2}: Same top 3 products")
            print()
    
    # Compare scores
    print("Score comparison:")
    for k in k_values_to_test:
        scores = results[k]['top_3_scores']
        print(f"K={k}: Top 3 scores = {[f'{s:.4f}' for s in scores]}")
    
    print()
    if different_results:
        print("✅ SUCCESS: KNN algorithm produces different results with different K values!")
    else:
        print("❌ WARNING: KNN algorithm produces same results regardless of K value")
    
    return results

def test_database_integration():
    """Test integrasi dengan database user preferences"""
    print("\n=== TESTING DATABASE INTEGRATION ===")
    
    try:
        # Connect to database
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )
        cursor = connection.cursor(dictionary=True)
        
        # Get a user with preferences
        cursor.execute("SELECT * FROM user_preferences WHERE k_value IS NOT NULL LIMIT 1")
        user_pref = cursor.fetchone()
        
        if user_pref:
            print(f"Testing with user ID: {user_pref['user_id']}")
            print(f"User's K-value: {user_pref['k_value']}")
            
            # Convert to preferences dict
            preferences = {
                'kondisi_kulit': user_pref['kondisi_kulit'],  # Use kondisi_kulit directly
                'masalah_kulit': user_pref['masalah_kulit'],
                'preferensi_produk': user_pref['preferensi_produk'],  # Use from database
                'rentang_harga': user_pref['rentang_harga'],
                'efektivitas_bahan_aktif': user_pref['efektivitas_bahan_aktif'],
                'frekuensi_penggunaan': user_pref['frekuensi_penggunaan'],
                'usia': user_pref['usia']
            }
            
            # Test dengan K-value dari database
            recommender = SkincareRecommender()
            recommendations = recommender.get_recommendations(
                preferences=preferences,
                max_recommendations=10,
                k_value=user_pref['k_value']
            )
            
            print(f"Generated {len(recommendations)} recommendations")
            print("Top 3 for dashboard:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec['product']['name']} (Score: {rec['knn_score']:.4f})")  # Changed field name
            
            print("✅ Database integration working correctly!")
        else:
            print("❌ No user preferences found in database")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    # Test KNN dengan K-value berbeda
    test_results = test_knn_with_different_k_values()
    
    # Test integrasi database
    test_database_integration()
    
    print("\n=== TEST COMPLETED ===")