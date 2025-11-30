# 🧴 Sistem Rekomendasi Skincare

Sistem rekomendasi produk skincare menggunakan algoritma Content-Based Filtering + KNN untuk memberikan rekomendasi yang dipersonalisasi berdasarkan jenis kulit dan preferensi pengguna.

## 📁 Struktur Proyek

```
skincare_rekomendasi/
├── src/                          # Source code utama
│   └── app.py                   # Aplikasi Flask utama
├── docs/                        # Dokumentasi
│   ├── README_DETAILED.md       # Dokumentasi lengkap
│   └── KODE_GOOGLE_COLAB.md    # Kode untuk Google Colab
├── notebooks/                   # Jupyter notebooks
│   └── Evaluasi_Algoritma_Rekomendasi_Skincare.ipynb
├── data/                        # Data dan dataset
├── tests/                       # Unit tests
├── config/                      # File konfigurasi
└── README.md                    # File ini
```

## 🚀 Fitur Utama

- **Sistem Rekomendasi**: Algoritma Content-Based Filtering + KNN untuk rekomendasi produk skincare yang dipersonalisasi
- **Evaluasi Algoritma**: Implementasi metrik evaluasi (Precision@K, Recall@K, F1-Score@K, NDCG@K, MAP)
- **Interface Web**: Aplikasi Flask dengan UI yang user-friendly
- **Google Colab Ready**: Kode siap pakai untuk eksperimen di Google Colab

## 🛠️ Instalasi

1. Clone repository:
```bash
git clone <repository-url>
cd skincare_rekomendasi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
cd src
python app.py
```

## 📊 Evaluasi Algoritma

Untuk menjalankan evaluasi algoritma, gunakan salah satu cara berikut:

### Option 1: Google Colab
1. Buka file `docs/KODE_GOOGLE_COLAB.md`
2. Copy kode dari setiap cell ke Google Colab
3. Jalankan cell secara berurutan

### Option 2: Jupyter Notebook
1. Buka `notebooks/Evaluasi_Algoritma_Rekomendasi_Skincare.ipynb`
2. Jalankan semua cell

## 📖 Dokumentasi

- **Dokumentasi Lengkap**: `docs/README_DETAILED.md`
- **Kode Google Colab**: `docs/KODE_GOOGLE_COLAB.md`
- **Notebook Evaluasi**: `notebooks/Evaluasi_Algoritma_Rekomendasi_Skincare.ipynb`

## 🔧 Teknologi

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript
- **Evaluasi**: Custom evaluation metrics implementation

## 📈 Metrik Evaluasi

- **Precision@K**: Presisi pada top-K rekomendasi
- **Recall@K**: Recall pada top-K rekomendasi  
- **F1-Score@K**: Harmonic mean dari Precision dan Recall
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision

## 🤝 Kontribusi

Silakan buat pull request atau buka issue untuk kontribusi dan saran perbaikan.

## 📄 Lisensi

MIT License - lihat file LICENSE untuk detail lengkap.