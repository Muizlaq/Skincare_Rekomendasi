#!/usr/bin/env python3
"""
Test script untuk menganalisis sensitivitas algoritma KNN terhadap nilai K
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import SkincareRecommender
from config import Config
import mysql.connector

def test_k_sensitivity():
    """Test apakah algoritma KNN menghasilkan hasil yang berbeda untuk K yang berbeda"""
    
    print("=== Test Sensitivitas Algoritma KNN ===\n")
    
    # Initialize recommender
    recommender = SkincareRecommender()
    
    # Test preferences (contoh user dengan kulit berminyak)
    test_preferences = {
        'kondisi_kulit': 'berminyak',
        'usia': 25,
        'masalah_kulit': 'jerawat',
        'rentang_harga': 'menengah',
        'efektivitas_bahan_aktif': 'tinggi',
        'preferensi_produk': 'serum',
        'frekuensi_penggunaan': 'harian',
        'kata_kunci_preferensi': 'anti aging brightening acne'
    }
    
    # Test dengan K values yang berbeda
    k_values = [3, 5, 7, 10]
    results = {}
    
    for k in k_values:
        print(f"Testing dengan K = {k}")
        recommendations = recommender.get_recommendations(test_preferences, k_value=k, max_recommendations=10)
        
        # Simpan top 5 produk dan skornya
        top_5 = []
        for i, rec in enumerate(recommendations[:5]):
            product_name = rec['product']['name']
            knn_score = rec['knn_score']
            content_sim = rec['content_similarity']
            top_5.append({
                'rank': i+1,
                'name': product_name,
                'knn_score': knn_score,
                'content_similarity': content_sim
            })
        
        results[k] = top_5
        
        print(f"Top 5 untuk K={k}:")
        for item in top_5:
            print(f"  {item['rank']}. {item['name'][:50]}... - KNN: {item['knn_score']:.4f}, Content: {item['content_similarity']:.4f}")
        print()
    
    # Analisis perbedaan
    print("=== ANALISIS PERBEDAAN ===")
    
    # Bandingkan K=3 vs K=7
    k3_products = [item['name'] for item in results[3]]
    k7_products = [item['name'] for item in results[7]]
    
    print(f"Produk yang sama di top 5 antara K=3 dan K=7: {len(set(k3_products) & set(k7_products))}/5")
    print(f"Produk berbeda: {set(k3_products) ^ set(k7_products)}")
    
    # Analisis perubahan ranking
    print("\n=== PERUBAHAN RANKING ===")
    for k in k_values:
        print(f"\nK = {k}:")
        for item in results[k]:
            print(f"  Rank {item['rank']}: {item['name'][:40]}... (KNN: {item['knn_score']:.4f})")
    
    # Cek apakah ada perbedaan signifikan dalam skor
    print("\n=== ANALISIS SKOR ===")
    for i in range(5):  # Top 5 products
        print(f"\nRank {i+1} products across different K values:")
        for k in k_values:
            if i < len(results[k]):
                item = results[k][i]
                print(f"  K={k}: {item['name'][:30]}... - KNN: {item['knn_score']:.4f}")
    
    return results

def test_with_database():
    """Test dengan data user dari database"""
    print("\n=== TEST DENGAN DATA DATABASE ===")
    
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
        
        # Get a real user preference
        cursor.execute("SELECT * FROM user_preferences WHERE user_id = 2")
        user_pref = cursor.fetchone()
        
        if user_pref:
            print(f"Testing dengan user ID {user_pref['user_id']}")
            
            preferences = {
                'kondisi_kulit': user_pref['kondisi_kulit'],
                'usia': user_pref['usia'],
                'masalah_kulit': user_pref['masalah_kulit'],
                'rentang_harga': user_pref['rentang_harga'],
                'efektivitas_bahan_aktif': user_pref['efektivitas_bahan_aktif'],
                'preferensi_produk': user_pref['preferensi_produk'],
                'frekuensi_penggunaan': user_pref['frekuensi_penggunaan'],
                'kata_kunci_preferensi': user_pref['kata_kunci_preferensi'] or ''
            }
            
            recommender = SkincareRecommender()
            
            # Test K=3 vs K=7
            print("\nPerbandingan K=3 vs K=7:")
            
            for k in [3, 7]:
                print(f"\n--- K = {k} ---")
                recommendations = recommender.get_recommendations(preferences, k_value=k, max_recommendations=5)
                
                for i, rec in enumerate(recommendations):
                    print(f"{i+1}. {rec['product']['name'][:50]}...")
                    print(f"   KNN Score: {rec['knn_score']:.4f}")
                    print(f"   Content Similarity: {rec['content_similarity']:.4f}")
                    print(f"   Price: Rp {rec['product']['price']:,}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    # Test sensitivitas K
    results = test_k_sensitivity()
    
    # Test dengan database
    test_with_database()
    
    print("\n=== KESIMPULAN ===")
    print("Jika hasil untuk K=3 dan K=7 identik, maka algoritma KNN perlu diperbaiki")
    print("untuk lebih sensitif terhadap perubahan nilai K.")