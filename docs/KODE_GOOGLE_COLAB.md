# 📊 Kode Google Colab - Evaluasi Algoritma Rekomendasi Skincare

Dokumen ini berisi semua kode yang diperlukan untuk evaluasi algoritma rekomendasi skincare di Google Colab. Salin setiap bagian kode ke dalam cell terpisah di Google Colab.

---

## 🔧 Cell 1: Setup Environment dan Import Libraries

```python
# Install required libraries
!pip install scikit-learn pandas numpy matplotlib seaborn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("✅ Environment setup complete!")
print("📚 Libraries imported successfully!")
```

---

## 📊 Cell 2: Generate Synthetic Product Dataset

```python
# Generate synthetic skincare products dataset
def generate_product_dataset(n_products=50):
    """Generate synthetic skincare products with realistic attributes"""
    
    # Define product categories and attributes
    categories = ['Cleanser', 'Moisturizer', 'Serum', 'Sunscreen', 'Toner', 'Mask']
    skin_types = ['Normal', 'Oily', 'Dry', 'Combination', 'Sensitive']
    concerns = ['Acne', 'Anti-aging', 'Brightening', 'Hydration', 'Oil Control', 'Sensitivity']
    ingredients = ['Hyaluronic Acid', 'Niacinamide', 'Vitamin C', 'Retinol', 'Salicylic Acid', 
                  'Ceramides', 'Peptides', 'AHA', 'BHA', 'Zinc Oxide']
    
    products = []
    
    for i in range(n_products):
        # Random product attributes
        category = np.random.choice(categories)
        suitable_skin = np.random.choice(skin_types, size=np.random.randint(1, 3), replace=False)
        main_concerns = np.random.choice(concerns, size=np.random.randint(1, 3), replace=False)
        key_ingredients = np.random.choice(ingredients, size=np.random.randint(2, 5), replace=False)
        
        product = {
            'product_id': f'P{i+1:03d}',
            'name': f'{category} {i+1}',
            'category': category,
            'skin_type': ', '.join(suitable_skin),
            'concerns': ', '.join(main_concerns),
            'ingredients': ', '.join(key_ingredients),
            'price': np.random.randint(50, 500) * 1000,  # Price in IDR
            'rating': round(np.random.uniform(3.5, 5.0), 1)
        }
        products.append(product)
    
    return pd.DataFrame(products)

# Generate dataset
products_df = generate_product_dataset(50)
print("✅ Product dataset generated!")
print(f"📦 Total products: {len(products_df)}")
print("\n🔍 Sample products:")
print(products_df.head())
```

---

## 👥 Cell 3: Generate Synthetic User Dataset

```python
# Generate synthetic user dataset
def generate_user_dataset(n_users=20):
    """Generate synthetic users with skincare preferences"""
    
    skin_types = ['Normal', 'Oily', 'Dry', 'Combination', 'Sensitive']
    concerns = ['Acne', 'Anti-aging', 'Brightening', 'Hydration', 'Oil Control', 'Sensitivity']
    
    users = []
    
    for i in range(n_users):
        # Random user preferences
        user_skin_type = np.random.choice(skin_types)
        user_concerns = np.random.choice(concerns, size=np.random.randint(1, 4), replace=False)
        
        user = {
            'user_id': f'U{i+1:03d}',
            'name': f'User {i+1}',
            'skin_type': user_skin_type,
            'concerns': ', '.join(user_concerns),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+']),
            'budget_range': np.random.choice(['Low', 'Medium', 'High'])
        }
        users.append(user)
    
    return pd.DataFrame(users)

# Generate user dataset
users_df = generate_user_dataset(20)
print("✅ User dataset generated!")
print(f"👥 Total users: {len(users_df)}")
print("\n🔍 Sample users:")
print(users_df.head())
```

---

## 🤖 Cell 4: Implement Skincare Recommendation System

```python
class SkincareRecommendationSystem:
    """Hybrid recommendation system combining Content-Based Filtering and K-Nearest Neighbors"""
    
    def __init__(self, products_df, users_df):
        self.products_df = products_df.copy()
        self.users_df = users_df.copy()
        self.tfidf_vectorizer = None
        self.product_features = None
        self.user_features = None
        self.user_similarity_matrix = None
        
    def prepare_features(self):
        """Prepare TF-IDF features for products and users"""
        
        # Combine product features into text
        self.products_df['combined_features'] = (
            self.products_df['category'] + ' ' +
            self.products_df['skin_type'] + ' ' +
            self.products_df['concerns'] + ' ' +
            self.products_df['ingredients']
        )
        
        # Combine user preferences into text
        self.users_df['combined_preferences'] = (
            self.users_df['skin_type'] + ' ' +
            self.users_df['concerns']
        )
        
        # Create TF-IDF features
        all_text = list(self.products_df['combined_features']) + list(self.users_df['combined_preferences'])
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        all_features = self.tfidf_vectorizer.fit_transform(all_text)
        
        # Split features
        n_products = len(self.products_df)
        self.product_features = all_features[:n_products]
        self.user_features = all_features[n_products:]
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.user_features)
        
        print("✅ Features prepared successfully!")
        
    def content_based_filtering(self, user_idx, top_k=10):
        """Content-Based Filtering using TF-IDF and cosine similarity"""
        
        if self.product_features is None:
            self.prepare_features()
        
        # Calculate similarity between user and all products
        user_vector = self.user_features[user_idx]
        similarities = cosine_similarity(user_vector, self.product_features).flatten()
        
        # Get top-k products
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'product_id': self.products_df.iloc[idx]['product_id'],
                'name': self.products_df.iloc[idx]['name'],
                'cbf_score': similarities[idx],
                'category': self.products_df.iloc[idx]['category']
            })
        
        return recommendations
    
    def find_k_nearest_neighbors(self, user_idx, k=5):
        """Find K nearest neighbors for a user"""
        
        if self.user_similarity_matrix is None:
            self.prepare_features()
        
        # Get similarities with other users (exclude self)
        similarities = self.user_similarity_matrix[user_idx].copy()
        similarities[user_idx] = -1  # Exclude self
        
        # Get top-k similar users
        neighbor_indices = similarities.argsort()[-k:][::-1]
        neighbor_scores = similarities[neighbor_indices]
        
        return neighbor_indices, neighbor_scores
    
    def aggressive_knn_scoring(self, cbf_recommendations, user_idx, k=5):
        """Apply aggressive KNN scoring with position multipliers and bonuses"""
        
        # Find K nearest neighbors
        neighbor_indices, neighbor_scores = self.find_k_nearest_neighbors(user_idx, k)
        
        # Create enhanced recommendations
        enhanced_recommendations = []
        
        for i, rec in enumerate(cbf_recommendations):
            base_score = rec['cbf_score']
            
            # Position multiplier (higher for top positions)
            position_multiplier = 1.0 + (0.1 * (len(cbf_recommendations) - i) / len(cbf_recommendations))
            
            # KNN boost calculation
            knn_boost = 0.0
            consensus_count = 0
            
            # Check if neighbors would also recommend this product
            for neighbor_idx, neighbor_sim in zip(neighbor_indices, neighbor_scores):
                neighbor_recs = self.content_based_filtering(neighbor_idx, top_k=10)
                neighbor_product_ids = [r['product_id'] for r in neighbor_recs]
                
                if rec['product_id'] in neighbor_product_ids:
                    # Position in neighbor's recommendations
                    neighbor_position = neighbor_product_ids.index(rec['product_id'])
                    position_bonus = 1.0 - (neighbor_position / 10.0)  # Higher bonus for better positions
                    
                    knn_boost += neighbor_sim * position_bonus
                    consensus_count += 1
            
            # Exclusivity bonus (if recommended by multiple neighbors)
            exclusivity_bonus = 1.0 + (0.2 * consensus_count / len(neighbor_indices))
            
            # Calculate final score
            final_score = base_score * position_multiplier * exclusivity_bonus + (knn_boost * 0.3)
            
            enhanced_rec = rec.copy()
            enhanced_rec.update({
                'position_multiplier': position_multiplier,
                'knn_boost': knn_boost,
                'exclusivity_bonus': exclusivity_bonus,
                'consensus_count': consensus_count,
                'final_score': final_score
            })
            
            enhanced_recommendations.append(enhanced_rec)
        
        # Sort by final score
        enhanced_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return enhanced_recommendations
    
    def hybrid_recommend(self, user_idx, k=5, top_n=10):
        """Generate hybrid recommendations using CBF + Aggressive KNN"""
        
        # Get CBF recommendations
        cbf_recs = self.content_based_filtering(user_idx, top_k=20)
        
        # Apply aggressive KNN scoring
        hybrid_recs = self.aggressive_knn_scoring(cbf_recs, user_idx, k)
        
        # Return top-N recommendations
        return hybrid_recs[:top_n]

# Initialize recommendation system
rec_system = SkincareRecommendationSystem(products_df, users_df)
rec_system.prepare_features()

print("✅ Recommendation system initialized!")
print("🎯 Ready for generating recommendations!")
```

---

## 🧪 Cell 5: Test Recommendation System

```python
# Test the recommendation system with a sample user
test_user_idx = 0
test_user = users_df.iloc[test_user_idx]

print(f"🧪 Testing recommendations for: {test_user['name']}")
print(f"👤 User Profile:")
print(f"   - Skin Type: {test_user['skin_type']}")
print(f"   - Concerns: {test_user['concerns']}")
print(f"   - Age Group: {test_user['age_group']}")

print("\n" + "="*60)

# Test different K values
k_values = [3, 5, 7]

for k in k_values:
    print(f"\n🎯 Recommendations with K={k}:")
    recommendations = rec_system.hybrid_recommend(test_user_idx, k=k, top_n=5)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} ({rec['category']})")
        print(f"   CBF Score: {rec['cbf_score']:.4f}")
        print(f"   Final Score: {rec['final_score']:.4f}")
        print(f"   Consensus: {rec['consensus_count']}/{k} neighbors")
        print()

print("✅ Recommendation system test completed!")
```

---

## 🎯 Cell 6: Generate Ground Truth for Evaluation

```python
def generate_ground_truth(users_df, products_df, min_relevant=3, max_relevant=8):
    """Generate ground truth data for evaluation"""
    
    ground_truth = {}
    
    for user_idx, user in users_df.iterrows():
        relevant_products = []
        
        # Find products that match user's skin type and concerns
        user_skin_type = user['skin_type']
        user_concerns = set(user['concerns'].split(', '))
        
        for prod_idx, product in products_df.iterrows():
            # Check skin type compatibility
            product_skin_types = set(product['skin_type'].split(', '))
            skin_type_match = user_skin_type in product_skin_types
            
            # Check concern overlap
            product_concerns = set(product['concerns'].split(', '))
            concern_overlap = len(user_concerns.intersection(product_concerns))
            
            # Consider relevant if skin type matches OR has concern overlap
            if skin_type_match or concern_overlap > 0:
                relevance_score = 0.5 if skin_type_match else 0.0
                relevance_score += 0.3 * (concern_overlap / len(user_concerns))
                
                # Add some randomness for realism
                relevance_score += np.random.uniform(-0.1, 0.1)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    relevant_products.append({
                        'product_id': product['product_id'],
                        'relevance_score': relevance_score
                    })
        
        # Sort by relevance and robustly set limit
        relevant_products.sort(key=lambda x: x['relevance_score'], reverse=True)
        limit = min(max_relevant, len(relevant_products))
        if limit == 0:
            ground_truth[user['user_id']] = []
        else:
            # If available relevant items fewer than min_relevant, take all; otherwise sample within range
            n_relevant = limit if limit < min_relevant else np.random.randint(min_relevant, limit + 1)
            ground_truth[user['user_id']] = [p['product_id'] for p in relevant_products[:n_relevant]]
    
    return ground_truth

# Generate ground truth
ground_truth = generate_ground_truth(users_df, products_df)

print("✅ Ground truth generated!")
print(f"📊 Ground truth statistics:")
for user_id, relevant_items in ground_truth.items():
    print(f"   {user_id}: {len(relevant_items)} relevant products")

print(f"\n📈 Average relevant products per user: {np.mean([len(items) for items in ground_truth.values()]):.1f}")
```

---

## 📏 Cell 7: Implement Evaluation Metrics

```python
class RecommendationEvaluator:
    """Comprehensive evaluation metrics for recommendation systems"""
    
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
    
    def precision_at_k(self, recommendations, user_id, k):
        """Calculate Precision@K"""
        if user_id not in self.ground_truth:
            return 0.0
        
        relevant_items = set(self.ground_truth[user_id])
        recommended_items = set([rec['product_id'] for rec in recommendations[:k]])
        
        if len(recommended_items) == 0:
            return 0.0
        
        return len(relevant_items.intersection(recommended_items)) / len(recommended_items)
    
    def recall_at_k(self, recommendations, user_id, k):
        """Calculate Recall@K"""
        if user_id not in self.ground_truth:
            return 0.0
        
        relevant_items = set(self.ground_truth[user_id])
        recommended_items = set([rec['product_id'] for rec in recommendations[:k]])
        
        if len(relevant_items) == 0:
            return 0.0
        
        return len(relevant_items.intersection(recommended_items)) / len(relevant_items)
    
    def f1_score_at_k(self, recommendations, user_id, k):
        """Calculate F1-Score@K"""
        precision = self.precision_at_k(recommendations, user_id, k)
        recall = self.recall_at_k(recommendations, user_id, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommendations, user_id, k):
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        if user_id not in self.ground_truth:
            return 0.0
        
        relevant_items = set(self.ground_truth[user_id])
        
        # Calculate DCG
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            if rec['product_id'] in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def average_precision(self, recommendations, user_id):
        """Calculate Average Precision"""
        if user_id not in self.ground_truth:
            return 0.0
        
        relevant_items = set(self.ground_truth[user_id])
        
        if len(relevant_items) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, rec in enumerate(recommendations):
            if rec['product_id'] in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def evaluate_user(self, recommendations, user_id, k_values=[3, 5, 7, 10]):
        """Evaluate all metrics for a single user"""
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(recommendations, user_id, k)
            metrics[f'recall@{k}'] = self.recall_at_k(recommendations, user_id, k)
            metrics[f'f1@{k}'] = self.f1_score_at_k(recommendations, user_id, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(recommendations, user_id, k)
        
        metrics['map'] = self.average_precision(recommendations, user_id)
        
        return metrics
    
    def evaluate_system(self, rec_system, k_neighbor_values=[3, 5, 7], k_eval_values=[3, 5, 7, 10]):
        """Evaluate the entire recommendation system"""
        results = []
        
        for k_neighbors in k_neighbor_values:
            print(f"🔄 Evaluating with K_neighbors = {k_neighbors}...")
            
            user_metrics = []
            
            for user_idx, user in rec_system.users_df.iterrows():
                user_id = user['user_id']
                
                # Generate recommendations
                recommendations = rec_system.hybrid_recommend(
                    user_idx, k=k_neighbors, top_n=15
                )
                
                # Evaluate metrics
                metrics = self.evaluate_user(recommendations, user_id, k_eval_values)
                metrics['user_id'] = user_id
                metrics['k_neighbors'] = k_neighbors
                
                user_metrics.append(metrics)
            
            results.extend(user_metrics)
        
        return pd.DataFrame(results)

# Initialize evaluator
evaluator = RecommendationEvaluator(ground_truth)
print("✅ Evaluation metrics implemented!")
print("📏 Available metrics: Precision@K, Recall@K, F1-Score@K, NDCG@K, MAP")
```

---

## ⚡ Cell 8: Run Full Evaluation

```python
# Run comprehensive evaluation
print("🚀 Starting comprehensive evaluation...")
print("⏳ This may take a few moments...")

# Run evaluation
evaluation_results = evaluator.evaluate_system(
    rec_system, 
    k_neighbor_values=[3, 5, 7, 10],
    k_eval_values=[3, 5, 7, 10]
)

print("✅ Evaluation completed!")
print(f"📊 Total evaluations: {len(evaluation_results)}")

# Calculate summary statistics
summary_stats = evaluation_results.groupby('k_neighbors').agg({
    'precision@3': ['mean', 'std'],
    'precision@5': ['mean', 'std'],
    'precision@7': ['mean', 'std'],
    'precision@10': ['mean', 'std'],
    'recall@3': ['mean', 'std'],
    'recall@5': ['mean', 'std'],
    'recall@7': ['mean', 'std'],
    'recall@10': ['mean', 'std'],
    'f1@3': ['mean', 'std'],
    'f1@5': ['mean', 'std'],
    'f1@7': ['mean', 'std'],
    'f1@10': ['mean', 'std'],
    'ndcg@3': ['mean', 'std'],
    'ndcg@5': ['mean', 'std'],
    'ndcg@7': ['mean', 'std'],
    'ndcg@10': ['mean', 'std'],
    'map': ['mean', 'std']
}).round(4)

print("\n📈 Summary Statistics:")
print(summary_stats)

# Display best performing K values
print("\n🏆 Best Performing K Values:")
# Exclude non-numeric columns for mean calculation
numeric_columns = evaluation_results.select_dtypes(include=[np.number]).columns
mean_metrics = evaluation_results.groupby('k_neighbors')[numeric_columns].mean()

for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map']:
    best_k = mean_metrics[metric].idxmax()
    best_score = mean_metrics[metric].max()
    print(f"   {metric}: K={best_k} (Score: {best_score:.4f})")
```

---



```python
# Create comprehensive visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('📊 Skincare Recommendation System - Performance Analysis', fontsize=16, fontweight='bold')

# 1. Precision@K across different K neighbors
ax1 = axes[0, 0]
for k_eval in [3, 5, 7, 10]:
    metric_data = evaluation_results.groupby('k_neighbors')[f'precision@{k_eval}'].mean()
    ax1.plot(metric_data.index, metric_data.values, marker='o', label=f'Precision@{k_eval}')
ax1.set_title('Precision@K vs K Neighbors')
ax1.set_xlabel('K Neighbors')
ax1.set_ylabel('Precision')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Recall@K across different K neighbors
ax2 = axes[0, 1]
for k_eval in [3, 5, 7, 10]:
    metric_data = evaluation_results.groupby('k_neighbors')[f'recall@{k_eval}'].mean()
    ax2.plot(metric_data.index, metric_data.values, marker='s', label=f'Recall@{k_eval}')
ax2.set_title('Recall@K vs K Neighbors')
ax2.set_xlabel('K Neighbors')
ax2.set_ylabel('Recall')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. F1-Score@K across different K neighbors
ax3 = axes[0, 2]
for k_eval in [3, 5, 7, 10]:
    metric_data = evaluation_results.groupby('k_neighbors')[f'f1@{k_eval}'].mean()
    ax3.plot(metric_data.index, metric_data.values, marker='^', label=f'F1@{k_eval}')
ax3.set_title('F1-Score@K vs K Neighbors')
ax3.set_xlabel('K Neighbors')
ax3.set_ylabel('F1-Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. NDCG@K across different K neighbors
ax4 = axes[1, 0]
for k_eval in [3, 5, 7, 10]:
    metric_data = evaluation_results.groupby('k_neighbors')[f'ndcg@{k_eval}'].mean()
    ax4.plot(metric_data.index, metric_data.values, marker='d', label=f'NDCG@{k_eval}')
ax4.set_title('NDCG@K vs K Neighbors')
ax4.set_xlabel('K Neighbors')
ax4.set_ylabel('NDCG')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. MAP across different K neighbors
ax5 = axes[1, 1]
map_data = evaluation_results.groupby('k_neighbors')['map'].mean()
ax5.plot(map_data.index, map_data.values, marker='*', linewidth=2, markersize=10, color='red')
ax5.set_title('Mean Average Precision (MAP)')
ax5.set_xlabel('K Neighbors')
ax5.set_ylabel('MAP')
ax5.grid(True, alpha=0.3)

# 6. Performance Heatmap
ax6 = axes[1, 2]
heatmap_data = evaluation_results.groupby('k_neighbors')[['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map']].mean()
sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6)
ax6.set_title('Performance Heatmap')
ax6.set_xlabel('K Neighbors')

plt.tight_layout()
plt.show()

# Performance ranking analysis
print("\n🏅 Performance Ranking Analysis:")
print("="*50)

# Exclude non-numeric columns for mean calculation
numeric_columns = evaluation_results.select_dtypes(include=[np.number]).columns
mean_performance = evaluation_results.groupby('k_neighbors')[numeric_columns].mean()

# Calculate overall performance score (weighted average)
weights = {
    'precision@5': 0.25,
    'recall@5': 0.25,
    'f1@5': 0.25,
    'ndcg@5': 0.15,
    'map': 0.10
}

overall_scores = {}
for k in mean_performance.index:
    score = sum(mean_performance.loc[k, metric] * weight for metric, weight in weights.items())
    overall_scores[k] = score

# Sort by overall performance
sorted_performance = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

print("📊 Overall Performance Ranking:")
for rank, (k, score) in enumerate(sorted_performance, 1):
    print(f"   {rank}. K={k}: Overall Score = {score:.4f}")
    
    # Show detailed metrics for this K
    metrics = mean_performance.loc[k]
    print(f"      - Precision@5: {metrics['precision@5']:.4f}")
    print(f"      - Recall@5: {metrics['recall@5']:.4f}")
    print(f"      - F1@5: {metrics['f1@5']:.4f}")
    print(f"      - NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"      - MAP: {metrics['map']:.4f}")
    print()

# K-sensitivity analysis (robust to NaN/std edge cases)
print("🔍 K-Sensitivity Analysis:")
print("="*30)

# Exclude non-numeric columns for std calculation
k_variance = evaluation_results.groupby('k_neighbors')[numeric_columns].std()
most_stable_metrics = {}

for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map']:
    series = k_variance[metric].replace([np.inf, -np.inf], np.nan)
    if series.dropna().empty:
        # Fallback: choose K with highest mean performance for this metric
        best_k = mean_performance[metric].idxmax()
        most_stable_metrics[metric] = (best_k, np.nan)
        print(f"📈 {metric}: Insufficient data for stability; fallback K={best_k}")
    else:
        min_variance_k = series.dropna().idxmin()
        min_variance = series.loc[min_variance_k]
        most_stable_metrics[metric] = (min_variance_k, min_variance)
        print(f"📈 {metric}: Most stable at K={min_variance_k} (std={min_variance:.4f})")

print("\n✅ Visualization and analysis completed!")
```

---

## 📊 Cell 9A: CBF & KNN Component Visualizations

```python
# Visualize CBF base vs KNN contributions and user similarity
plt.style.use('default')

# Choose example user and K (fallback to 5 if best_k not defined yet)
example_user_idx = 0
try:
    k_example = best_k
except NameError:
    k_example = 5

# Generate recommendations
cbf_recs = rec_system.content_based_filtering(example_user_idx, top_k=15)
hybrid_recs = rec_system.aggressive_knn_scoring(cbf_recs, example_user_idx, k=k_example)

# Prepare dataframe
rec_df = pd.DataFrame(hybrid_recs)
rec_df['cbf_component'] = rec_df['cbf_score'] * rec_df['position_multiplier'] * rec_df['exclusivity_bonus']
rec_df['knn_component'] = rec_df['knn_boost'] * 0.3
top_df = rec_df.head(10)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'📊 CBF vs KNN Components (User {example_user_idx+1}, K={k_example})', fontsize=16, fontweight='bold')

# 1) Stacked bar: CBF component vs KNN component
ax1 = axes[0, 0]
indices = np.arange(len(top_df))
ax1.bar(indices, top_df['cbf_component'], label='CBF Component', color='#4e79a7')
ax1.bar(indices, top_df['knn_component'], bottom=top_df['cbf_component'], label='KNN Component', color='#f28e2b')
ax1.set_xticks(indices)
ax1.set_xticklabels(top_df['name'], rotation=45, ha='right')
ax1.set_title('Stacked Contribution: CBF vs KNN')
ax1.set_ylabel('Score')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2) Scatter: CBF score vs Final score (bubble size = consensus)
ax2 = axes[0, 1]
sizes = (top_df['consensus_count'] + 1) * 60
sc = ax2.scatter(top_df['cbf_score'], top_df['final_score'], s=sizes, c=top_df['knn_component'], cmap='viridis', alpha=0.8)
ax2.set_title('CBF Score vs Final Score (colored by KNN component)')
ax2.set_xlabel('CBF Score')
ax2.set_ylabel('Final Score')
cb = plt.colorbar(sc, ax=ax2)
cb.set_label('KNN Component')
ax2.grid(True, alpha=0.3)

# 3) Trend: Average KNN boost & consensus across K values
ax3 = axes[1, 0]
k_values = [3, 5, 7, 10]
avg_boosts, avg_consensus = [], []

for k in k_values:
    boosts = []
    consensuses = []
    for uidx in range(len(rec_system.users_df)):
        cbf_u = rec_system.content_based_filtering(uidx, top_k=20)
        hy_u = rec_system.aggressive_knn_scoring(cbf_u, uidx, k)
        # Use top 10 items for aggregation
        boosts += [r['knn_boost'] for r in hy_u[:10]]
        consensuses += [r['consensus_count'] for r in hy_u[:10]]
    avg_boosts.append(np.mean(boosts))
    avg_consensus.append(np.mean(consensuses))

ax3.plot(k_values, avg_boosts, marker='o', label='Avg KNN Boost', color='#e15759')
ax3.plot(k_values, avg_consensus, marker='s', label='Avg Consensus Count', color='#76b7b2')
ax3.set_title('KNN Effect vs K Neighbors')
ax3.set_xlabel('K Neighbors')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4) Heatmap: User similarity (first 12 users)
ax4 = axes[1, 1]
subset = rec_system.user_similarity_matrix[:12, :12]
sns.heatmap(subset, cmap='Blues', annot=False, ax=ax4)
ax4.set_title('User Similarity Heatmap (first 12 users)')
ax4.set_xlabel('User Index')
ax4.set_ylabel('User Index')

plt.tight_layout()
plt.show()

print("\n🧾 Top-10 Recommendations (components):")
display(top_df[['product_id', 'name', 'cbf_score', 'position_multiplier', 'exclusivity_bonus', 'knn_boost', 'consensus_count', 'cbf_component', 'knn_component', 'final_score']])
```

---

## 📝 Cell 10: Final Conclusions and Recommendations

```python
# Generate final conclusions and recommendations
print("🎯 FINAL CONCLUSIONS AND RECOMMENDATIONS")
print("="*60)

# Best K value recommendation
best_k = sorted_performance[0][0]
best_score = sorted_performance[0][1]

print(f"🏆 RECOMMENDED K VALUE: {best_k}")
print(f"   Overall Performance Score: {best_score:.4f}")
print()

# Detailed analysis of the best K
best_k_metrics = mean_performance.loc[best_k]
print(f"📊 Performance Metrics for K={best_k}:")
print(f"   • Precision@5: {best_k_metrics['precision@5']:.4f} ({best_k_metrics['precision@5']*100:.1f}%)")
print(f"   • Recall@5: {best_k_metrics['recall@5']:.4f} ({best_k_metrics['recall@5']*100:.1f}%)")
print(f"   • F1-Score@5: {best_k_metrics['f1@5']:.4f} ({best_k_metrics['f1@5']*100:.1f}%)")
print(f"   • NDCG@5: {best_k_metrics['ndcg@5']:.4f} ({best_k_metrics['ndcg@5']*100:.1f}%)")
print(f"   • MAP: {best_k_metrics['map']:.4f} ({best_k_metrics['map']*100:.1f}%)")
print()

# Algorithm characteristics
print("🔬 ALGORITHM CHARACTERISTICS:")
print("-" * 30)

if best_k <= 5:
    print("✅ SELECTIVE APPROACH:")
    print("   • Focuses on highly similar users")
    print("   • Higher precision, potentially lower recall")
    print("   • Good for niche/specific recommendations")
elif best_k <= 7:
    print("✅ BALANCED APPROACH:")
    print("   • Good balance between precision and recall")
    print("   • Moderate diversity in recommendations")
    print("   • Suitable for general-purpose recommendations")
else:
    print("✅ INCLUSIVE APPROACH:")
    print("   • Considers broader user base")
    print("   • Higher recall, potentially lower precision")
    print("   • Good for discovery and diversity")

print()

# Stability analysis
print("📈 STABILITY ANALYSIS:")
print("-" * 20)
best_k_std = k_variance.loc[best_k]
stability_score = 1 - best_k_std[['precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'map']].mean()

if stability_score > 0.8:
    stability_level = "VERY STABLE"
elif stability_score > 0.6:
    stability_level = "STABLE"
elif stability_score > 0.4:
    stability_level = "MODERATELY STABLE"
else:
    stability_level = "UNSTABLE"

print(f"🎯 Stability Level: {stability_level}")
print(f"   Stability Score: {stability_score:.3f}")
print()

# Implementation recommendations
print("🚀 IMPLEMENTATION RECOMMENDATIONS:")
print("-" * 35)
print("1. 🎯 PRODUCTION SETTINGS:")
print(f"   • Use K={best_k} for optimal performance")
print(f"   • Monitor precision@5 (target: >{best_k_metrics['precision@5']:.3f})")
print(f"   • Monitor recall@5 (target: >{best_k_metrics['recall@5']:.3f})")
print()

print("2. 📊 MONITORING METRICS:")
print("   • Primary: F1-Score@5 (balanced performance)")
print("   • Secondary: NDCG@5 (ranking quality)")
print("   • Tertiary: MAP (overall system performance)")
print()

print("3. 🔧 OPTIMIZATION OPPORTUNITIES:")
if best_k_metrics['precision@5'] < 0.5:
    print("   • Consider improving content-based features")
if best_k_metrics['recall@5'] < 0.4:
    print("   • Consider expanding the neighbor search space")
if best_k_metrics['ndcg@5'] < 0.6:
    print("   • Consider improving ranking algorithms")
print("   • Implement A/B testing for continuous improvement")
print("   • Consider hybrid approaches with collaborative filtering")
print()

print("4. ⚠️ CONSIDERATIONS:")
print("   • Results based on synthetic data - validate with real data")
print("   • Monitor performance degradation over time")
print("   • Implement cold-start strategies for new users/products")
print("   • Consider computational costs for large-scale deployment")

print("\n" + "="*60)
print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
print("📋 All metrics calculated and analyzed")
print("🎯 Recommendations ready for implementation")
print("="*60)
```
```

---

## 📚 Appendix: Quick Reference

### 🔍 Metrik Evaluasi:
- **Precision@K**: Proporsi item relevan dalam K rekomendasi teratas
- **Recall@K**: Proporsi item relevan yang berhasil ditangkap
- **F1-Score@K**: Harmonic mean dari Precision dan Recall
- **NDCG@K**: Kualitas ranking dengan mempertimbangkan posisi
- **MAP**: Mean Average Precision - performa keseluruhan sistem

### 🎯 Interpretasi Nilai K:
- **K=3-5**: Selektif, fokus pada pengguna sangat mirip
- **K=5-7**: Seimbang antara precision dan recall
- **K=7-10**: Inklusif, mempertimbangkan lebih banyak pengguna

### 🚀 Tips Implementasi:
1. Gunakan K yang direkomendasikan dari evaluasi
2. Monitor metrik secara berkala
3. Implementasi A/B testing
4. Siapkan strategi cold-start
5. Pertimbangkan computational cost

---

**Selamat! Anda telah berhasil menyelesaikan evaluasi algoritma rekomendasi skincare! 🎉**