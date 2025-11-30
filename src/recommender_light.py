import re
import math
from models import Product
from config import Config

class SkincareRecommender:
    def __init__(self):
        self.products = None
        self.vocab = None
        self.idf = None
        self.doc_vectors = None

    def load_products(self):
        products = Product.get_all()
        if not products:
            return False
        self.products = [dict(p) for p in products]
        for p in self.products:
            deskripsi = p.get('description', '')
            nama = p.get('name', '')
            brand = p.get('brand', '')
            p['deskripsi_clean'] = self._clean_text(deskripsi)
            p['nama_clean'] = self._clean_text(nama)
            p['combined_text'] = (p['nama_clean'] + ' ' + brand + ' ' + p['deskripsi_clean']).strip()
        self._build_content_features()
        return True

    def _clean_text(self, text):
        if text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return ' '.join(text.split())

    def _tokenize(self, text):
        return text.split() if text else []

    def _build_content_features(self):
        term_freq = {}
        doc_freq = {}
        tokenized_docs = []
        for p in self.products:
            tokens = self._tokenize(p['combined_text'])
            tokenized_docs.append(tokens)
            seen = set()
            for t in tokens:
                term_freq[t] = term_freq.get(t, 0) + 1
                if t not in seen:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                    seen.add(t)
        if not term_freq:
            self.vocab = {}
            self.idf = {}
            self.doc_vectors = []
            return
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        max_features = 1000
        vocab_terms = [t for t, _ in sorted_terms[:max_features]]
        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        n_docs = len(self.products)
        self.idf = {}
        for t in vocab_terms:
            df = doc_freq.get(t, 0)
            self.idf[t] = math.log((n_docs + 1) / (df + 1)) + 1.0
        self.doc_vectors = []
        for tokens in tokenized_docs:
            tf = {}
            total = len(tokens) if tokens else 1
            for t in tokens:
                if t in self.vocab:
                    tf[t] = tf.get(t, 0) + 1
            vec = {}
            for t, count in tf.items():
                vec[self.vocab[t]] = (count / total) * self.idf[t]
            self.doc_vectors.append(vec)

    def _create_user_profile(self, preferences):
        parts = []
        skin_condition_keywords = {
            'berminyak': 'oil control minyak sebum',
            'kering': 'moisturizer pelembab hydrating',
            'kombinasi': 'balance seimbang combination',
            'sensitif': 'gentle sensitive hypoallergenic',
            'normal': 'daily maintenance normal'
        }
        parts.append(skin_condition_keywords.get(preferences.get('kondisi_kulit'), ''))
        skin_problem_keywords = {
            'jerawat': 'acne anti jerawat salicylic',
            'komedo': 'blackhead whitehead pore',
            'kusam': 'brightening whitening vitamin c',
            'kerutan': 'anti aging retinol wrinkle',
            'flek_hitam': 'dark spot niacinamide',
            'pori_besar': 'pore minimizer tightening'
        }
        parts.append(skin_problem_keywords.get(preferences.get('masalah_kulit'), ''))
        if preferences.get('preferensi_produk') and preferences.get('preferensi_produk') != 'semua':
            parts.append(preferences.get('preferensi_produk'))
        if preferences.get('kata_kunci_preferensi'):
            parts.append(preferences.get('kata_kunci_preferensi'))
        if preferences.get('kata_kunci'):
            parts.append(preferences.get('kata_kunci'))
        q = self._clean_text(' '.join(parts))
        tokens = self._tokenize(q)
        tf = {}
        total = len(tokens) if tokens else 1
        for t in tokens:
            if t in self.vocab:
                tf[t] = tf.get(t, 0) + 1
        vec = {}
        for t, count in tf.items():
            vec[self.vocab[t]] = (count / total) * self.idf.get(t, 0.0)
        return vec

    def _parse_budget_range(self, rentang_harga):
        if not rentang_harga:
            return 0, 1000000
        ranges = {
            '0-50000': (0, 50000),
            '50000-100000': (50000, 100000),
            '100000-200000': (100000, 200000),
            '200000-500000': (200000, 500000),
            '500000+': (500000, 1000000)
        }
        return ranges.get(rentang_harga, (0, 1000000))

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            dot += v * b.get(k, 0.0)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def get_recommendations(self, preferences, max_recommendations=10, k_value=None, knn_only=False, top_k_only=False):
        if self.products is None:
            if not self.load_products():
                return []
        k = k_value if k_value is not None else Config.KNN_K_VALUE
        rentang_harga = preferences.get('rentang_harga', '0-50000')
        budget_min, budget_max = self._parse_budget_range(rentang_harga)
        filtered = []
        filtered_indices = []
        for i, p in enumerate(self.products):
            price = p.get('price', 0)
            if price >= budget_min and price <= budget_max:
                filtered.append(p)
                filtered_indices.append(i)
        if not filtered:
            return []
        user_vec = self._create_user_profile(preferences)
        sims = []
        for idx in filtered_indices:
            sims.append(self._cosine(user_vec, self.doc_vectors[idx]))
        distances = []
        for i, idx in enumerate(filtered_indices):
            distances.append({
                'index': idx,
                'filtered_index': i,
                'distance': 1.0 - sims[i],
                'product': filtered[i],
                'content_similarity': sims[i]
            })
        distances.sort(key=lambda x: x['distance'])
        k_actual = min(k, len(distances))
        neighbors = distances[:k_actual]
        final = []
        if top_k_only:
            for n in neighbors:
                fi = n['filtered_index']
                row = filtered[fi]
                score = self._calculate_knn_score_filtered(fi, neighbors, sims, k, knn_only)
                final.append({'product': row, 'content_similarity': sims[fi], 'knn_score': score, 'explanation': self._generate_explanation(sims[fi], preferences, score)})
        else:
            for i in range(len(filtered)):
                score = self._calculate_knn_score_filtered(i, neighbors, sims, k, knn_only)
                final.append({'product': filtered[i], 'content_similarity': sims[i], 'knn_score': score, 'explanation': self._generate_explanation(sims[i], preferences, score)})
        final.sort(key=lambda x: x['knn_score'], reverse=True)
        if top_k_only:
            return final[:k_actual]
        return final[:max_recommendations]

    def _calculate_knn_score_filtered(self, product_index, k_nearest_neighbors, content_similarities, k, knn_only=False):
        if not k_nearest_neighbors:
            return content_similarities[product_index]
        base_similarity = content_similarities[product_index]
        neighbor_indices = [n['filtered_index'] for n in k_nearest_neighbors]
        k_len = len(k_nearest_neighbors)
        neighbor_sims = [content_similarities[idx] for idx in neighbor_indices]
        avg_neighbor_sim = float(sum(neighbor_sims) / len(neighbor_sims)) if neighbor_sims else 0.0
        if product_index in neighbor_indices:
            position = neighbor_indices.index(product_index)
            if k_len <= 3:
                position_multiplier = [3.0, 2.0, 1.5][position] if position < 3 else 1.0
                exclusivity_bonus = 1.5
            elif k_len <= 5:
                position_multiplier = [2.5, 1.8, 1.4, 1.2, 1.0][position] if position < 5 else 1.0
                exclusivity_bonus = 1.2
            else:
                position_multiplier = max(1.0, 2.0 - (position * 0.15))
                exclusivity_bonus = 1.0
            consensus_weight = 1.0 / max(1, k_len)
            final_score = base_similarity * position_multiplier * exclusivity_bonus
            final_score += avg_neighbor_sim * consensus_weight * 0.3
        else:
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

    def _generate_explanation(self, content_score, preferences, knn_score=None):
        parts = []
        s = knn_score if knn_score is not None else content_score
        if s > 0.8:
            parts.append("Sangat direkomendasikan berdasarkan analisis KNN")
        elif s > 0.6:
            parts.append("Sangat cocok dengan preferensi Anda")
        elif s > 0.4:
            parts.append("Cukup sesuai dengan preferensi Anda")
        elif s > 0.2:
            parts.append("Memiliki beberapa kesamaan dengan preferensi Anda")
        else:
            parts.append("Produk alternatif yang mungkin menarik")
        if preferences.get('jenis_kulit'):
            parts.append(f"untuk kulit {preferences['jenis_kulit']}")
        if preferences.get('masalah_kulit'):
            parts.append(f"mengatasi {preferences['masalah_kulit']}")
        return " - ".join(parts)
