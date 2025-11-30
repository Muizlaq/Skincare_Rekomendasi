import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models import Product
from config import Config
import re
import math

class SkincareRecommender:
    """Skincare recommendation system using Content-Based Filtering and KNN"""
    
    def __init__(self):
        self.products_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
    
    def load_products(self):
        """Load products from database"""
        products = Product.get_all()
        if not products:
            return False
        
        self.products_df = pd.DataFrame(products)
        self._preprocess_data()
        self._build_content_features()
        return True
    
    def _preprocess_data(self):
        """Preprocess product data"""
        # Clean and normalize text data
        self.products_df['deskripsi_clean'] = self.products_df['description'].apply(self._clean_text)
        self.products_df['nama_clean'] = self.products_df['name'].apply(self._clean_text)
        
        # Combine text features for Content-Based Filtering
        self.products_df['combined_text'] = (
            self.products_df['nama_clean'] + ' ' + 
            self.products_df['brand'] + ' ' + 
            self.products_df['deskripsi_clean']
        )
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _normalize_feature(self, feature):
        """Normalize numerical feature using Min-Max scaling"""
        min_val = feature.min()
        max_val = feature.max()
        
        if max_val == min_val:
            return feature * 0  # All values are the same
        
        return (feature - min_val) / (max_val - min_val)
    
    def _build_content_features(self):
        """Build TF-IDF features for content-based filtering"""
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Indonesian stopwords not available in sklearn
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform the combined text
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['combined_text'])
    
    def _create_user_profile(self, preferences):
        """Create user profile vector from preferences"""
        # Create user query text based on preferences
        user_text_parts = []
        
        # Add skin condition keywords
        skin_condition_keywords = {
            'berminyak': 'oil control minyak sebum',
            'kering': 'moisturizer pelembab hydrating',
            'kombinasi': 'balance seimbang combination',
            'sensitif': 'gentle sensitive hypoallergenic',
            'normal': 'daily maintenance normal'
        }
        user_text_parts.append(skin_condition_keywords.get(preferences['kondisi_kulit'], ''))
        
        # Add skin problem keywords
        skin_problem_keywords = {
            'jerawat': 'acne anti jerawat salicylic',
            'komedo': 'blackhead whitehead pore',
            'kusam': 'brightening whitening vitamin c',
            'kerutan': 'anti aging retinol wrinkle',
            'flek_hitam': 'dark spot niacinamide',
            'pori_besar': 'pore minimizer tightening'
        }
        user_text_parts.append(skin_problem_keywords.get(preferences['masalah_kulit'], ''))
        
        # Add product preference keywords
        if preferences['preferensi_produk'] != 'semua':
            user_text_parts.append(preferences['preferensi_produk'])
        
        # Add user keywords if provided
        if preferences.get('kata_kunci_preferensi'):
            user_text_parts.append(preferences['kata_kunci_preferensi'])
        
        # Add search keywords if provided
        if preferences.get('kata_kunci'):
            user_text_parts.append(preferences['kata_kunci'])
        
        # Combine all text parts
        user_query = ' '.join(user_text_parts)
        user_query = self._clean_text(user_query)
        
        # Transform user query using existing TF-IDF vectorizer
        user_tfidf = self.tfidf_vectorizer.transform([user_query])
        
        return user_tfidf
    
    def _parse_budget_range(self, rentang_harga):
        """Convert rentang_harga enum to numeric min/max values"""
        if not rentang_harga:
            return 0, 1000000  # Default range if no budget specified
        
        # Map rentang_harga enum values to numeric ranges
        budget_ranges = {
            '0-50000': (0, 50000),
            '50000-100000': (50000, 100000),
            '100000-200000': (100000, 200000),
            '200000-500000': (200000, 500000),
            '500000+': (500000, 1000000)
        }
        
        return budget_ranges.get(rentang_harga, (0, 1000000))

    def get_recommendations(self, preferences, max_recommendations=10, k_value=None, knn_only=False, top_k_only=False):
        """Get product recommendations using proper KNN algorithm"""
        # Load products if not already loaded
        if self.products_df is None:
            if not self.load_products():
                return []

        # Use provided k_value or default from config
        k = k_value if k_value is not None else Config.KNN_K_VALUE
        
        # Apply budget filtering FIRST before any calculations
        # Use rentang_harga from preferences instead of budget_min/budget_max
        rentang_harga = preferences.get('rentang_harga', '0-50000')
        budget_min, budget_max = self._parse_budget_range(rentang_harga)
        
        # Filter products by budget range
        filtered_products_df = self.products_df[
            (self.products_df['price'] >= budget_min) & 
            (self.products_df['price'] <= budget_max)
        ].copy()
        
        # If no products match the budget, return empty list
        if filtered_products_df.empty:
            print(f"No products found in budget range {budget_min}-{budget_max} for rentang_harga: {rentang_harga}")
            return []
        
        print(f"Found {len(filtered_products_df)} products in budget range {budget_min}-{budget_max} for rentang_harga: {rentang_harga}")
        
        # Create user profile based on preferences
        user_features = self._create_user_profile(preferences)
        
        # Get content-based similarities using cosine similarity for filtered products only
        # We need to get the TF-IDF matrix for filtered products
        filtered_indices = filtered_products_df.index.tolist()
        filtered_tfidf_matrix = self.tfidf_matrix[filtered_indices]
        
        content_similarities = cosine_similarity(user_features, filtered_tfidf_matrix).flatten()
        
        # Step 1: Calculate distances for filtered products only
        product_distances = []
        
        for i, (idx, row) in enumerate(filtered_products_df.iterrows()):
            # Get content similarity score for this product
            content_score = content_similarities[i]
            
            # Calculate distance (1 - similarity for distance metric)
            distance = 1 - content_score
            
            # Store product with its distance and similarity
            product_distances.append({
                'index': idx,
                'filtered_index': i,  # Index in filtered dataset
                'distance': distance,
                'product': row.to_dict(),
                'content_similarity': content_score
            })
        
        # Step 2: Sort by distance to find nearest neighbors
        product_distances.sort(key=lambda x: x['distance'])
        
        # Step 3: Get K nearest neighbors (limit to available products)
        k_actual = min(k, len(product_distances))
        k_nearest_neighbors = product_distances[:k_actual]
        
        # Step 4: Use KNN algorithm to calculate recommendation scores for filtered products
        final_recommendations = []

        if top_k_only:
            # Return exactly K nearest neighbors as recommendations
            for neighbor in k_nearest_neighbors:
                fi = neighbor['filtered_index']
                row = filtered_products_df.iloc[fi]
                knn_score = self._calculate_knn_score_filtered(fi, k_nearest_neighbors, content_similarities, k, knn_only)
                recommendation = {
                    'product': row.to_dict(),
                    'content_similarity': content_similarities[fi],
                    'knn_score': knn_score,
                    'explanation': self._generate_explanation(content_similarities[fi], preferences, knn_score)
                }
                final_recommendations.append(recommendation)
        else:
            for i, (idx, row) in enumerate(filtered_products_df.iterrows()):
                # Calculate KNN score based on K nearest neighbors with explicit K-sensitivity
                knn_score = self._calculate_knn_score_filtered(i, k_nearest_neighbors, content_similarities, k, knn_only)
                
                # Create recommendation entry
                recommendation = {
                    'product': row.to_dict(),
                    'content_similarity': content_similarities[i],
                    'knn_score': knn_score,
                    'explanation': self._generate_explanation(content_similarities[i], preferences, knn_score)
                }
                final_recommendations.append(recommendation)
        
        # Step 5: Sort by KNN score (descending - higher score = better recommendation)
        final_recommendations.sort(key=lambda x: x['knn_score'], reverse=True)
        
        # Step 6: Return top max_recommendations (default 10), or exactly K if top_k_only
        if top_k_only:
            return final_recommendations[:k_actual]
        return final_recommendations[:max_recommendations]
    
    def _calculate_knn_score_filtered(self, product_index, k_nearest_neighbors, content_similarities, k, knn_only=False):
        """Calculate KNN-based score with explicit K-sensitivity on filtered dataset"""
        if not k_nearest_neighbors:
            return content_similarities[product_index]

        base_similarity = content_similarities[product_index]
        neighbor_indices = [n['filtered_index'] for n in k_nearest_neighbors]
        k_len = len(k_nearest_neighbors)

        # Average similarity of the K neighbors (consensus)
        neighbor_sims = [content_similarities[idx] for idx in neighbor_indices]
        avg_neighbor_sim = float(np.mean(neighbor_sims)) if neighbor_sims else 0.0

        if product_index in neighbor_indices:
            position = neighbor_indices.index(product_index)

            # Aggressive boosting depending on K size and position
            if k_len <= 3:
                position_multiplier = [3.0, 2.0, 1.5][position] if position < 3 else 1.0
                exclusivity_bonus = 1.5
            elif k_len <= 5:
                position_multiplier = [2.5, 1.8, 1.4, 1.2, 1.0][position] if position < 5 else 1.0
                exclusivity_bonus = 1.2
            else:
                position_multiplier = max(1.0, 2.0 - (position * 0.15))
                exclusivity_bonus = 1.0

            # Consensus bonus stronger for smaller K
            consensus_weight = 1.0 / max(1, k_len)
            final_score = base_similarity * position_multiplier * exclusivity_bonus
            final_score += avg_neighbor_sim * consensus_weight * 0.3
        else:
            # Penalties when not inside K neighbors, harsher for smaller K
            if k_len <= 3:
                exclusion_penalty = 0.4
                distance_penalty = 0.2
            elif k_len <= 5:
                exclusion_penalty = 0.6
                distance_penalty = 0.15
            else:
                exclusion_penalty = 0.75
                distance_penalty = 0.1

            max_neighbor_sim = max(neighbor_sims) if neighbor_sims else 0.0
            penalty = (1.0 - max_neighbor_sim) * distance_penalty
            final_score = base_similarity * exclusion_penalty - penalty

        # Blend with content similarity where K influences the KNN weight
        # Smaller K -> stronger KNN influence; Larger K -> more content-driven
        if knn_only:
            knn_weight = 1.0
        else:
            if k_len <= 3:
                knn_weight = 0.6
            elif k_len <= 5:
                knn_weight = 0.45
            else:
                knn_weight = 0.3

        content_weight = 1.0 - knn_weight
        blended = content_weight * base_similarity + knn_weight * final_score
        return blended

    def _calculate_knn_score(self, product_idx, k_neighbors, all_similarities):
        """Calculate KNN-based recommendation score with aggressive K-sensitivity"""
        # Get the K nearest neighbor indices and their similarities
        neighbor_indices = [neighbor['index'] for neighbor in k_neighbors]
        neighbor_similarities = [neighbor['content_similarity'] for neighbor in k_neighbors]
        k = len(k_neighbors)
        
        # Base similarity score
        base_score = all_similarities[product_idx]
        
        if product_idx in neighbor_indices:
            # This product is one of the K nearest neighbors
            position = neighbor_indices.index(product_idx)
            
            # AGGRESSIVE K-SENSITIVITY: Smaller K gives exponentially higher rewards
            # K=3: Very exclusive, top positions get massive boost
            # K=7+: More inclusive, smaller boost
            
            if k <= 3:
                # Very small K: Extremely aggressive boosting
                position_multiplier = [3.0, 2.0, 1.5][position] if position < 3 else 1.0
                k_exclusivity_bonus = 1.5  # High bonus for being in top 3
            elif k <= 5:
                # Small K: Strong boosting
                position_multiplier = [2.5, 1.8, 1.4, 1.2, 1.0][position] if position < 5 else 1.0
                k_exclusivity_bonus = 1.2
            else:
                # Larger K: Moderate boosting
                position_multiplier = max(1.0, 2.0 - (position * 0.15))
                k_exclusivity_bonus = 1.0
            
            # Calculate final score with aggressive K-dependent boost
            final_score = base_score * position_multiplier * k_exclusivity_bonus
            
            # Add neighbor consensus bonus (stronger for smaller K)
            consensus_weight = 1.0 / k  # Smaller K = higher weight
            if len(neighbor_similarities) > 1:
                avg_neighbor_sim = np.mean(neighbor_similarities)
                consensus_bonus = avg_neighbor_sim * consensus_weight * 0.3
                final_score += consensus_bonus
            
            return final_score
            
        else:
            # Product is NOT in K nearest neighbors
            # AGGRESSIVE PENALTY: Smaller K gives much harsher penalties
            
            if k <= 3:
                # Very small K: Harsh penalty for not being in top 3
                exclusion_penalty = 0.4  # Reduce score to 40% of original
                distance_penalty = 0.2   # Additional penalty based on distance from neighbors
            elif k <= 5:
                # Small K: Moderate penalty
                exclusion_penalty = 0.6
                distance_penalty = 0.15
            else:
                # Larger K: Lighter penalty
                exclusion_penalty = 0.75
                distance_penalty = 0.1
            
            # Calculate how far this product is from the K neighbors
            max_neighbor_sim = max(neighbor_similarities)
            min_neighbor_sim = min(neighbor_similarities)
            
            if base_score < min_neighbor_sim:
                # Product is clearly worse than all K neighbors
                distance_factor = (min_neighbor_sim - base_score) * distance_penalty
                final_penalty = exclusion_penalty - distance_factor
            else:
                # Product is somewhere in the range of K neighbors
                final_penalty = exclusion_penalty
            
            # Ensure penalty doesn't make score negative
            final_penalty = max(0.1, final_penalty)
            
            return base_score * final_penalty
    
    def _generate_explanation(self, content_score, preferences, knn_score=None):
        """Generate explanation for recommendation based on content similarity and KNN score"""
        explanation_parts = []
        
        # Use KNN score if available, otherwise use content score
        score_to_use = knn_score if knn_score is not None else content_score
        
        # KNN-based explanation
        if score_to_use > 0.8:
            explanation_parts.append("Sangat direkomendasikan berdasarkan analisis KNN")
        elif score_to_use > 0.6:
            explanation_parts.append("Sangat cocok dengan preferensi Anda")
        elif score_to_use > 0.4:
            explanation_parts.append("Cukup sesuai dengan preferensi Anda")
        elif score_to_use > 0.2:
            explanation_parts.append("Memiliki beberapa kesamaan dengan preferensi Anda")
        else:
            explanation_parts.append("Produk alternatif yang mungkin menarik")
        
        # Add specific preference matches
        if 'jenis_kulit' in preferences and preferences['jenis_kulit']:
            explanation_parts.append(f"untuk kulit {preferences['jenis_kulit']}")
        
        if 'masalah_kulit' in preferences and preferences['masalah_kulit']:
            explanation_parts.append(f"mengatasi {preferences['masalah_kulit']}")
        
        return " - ".join(explanation_parts)