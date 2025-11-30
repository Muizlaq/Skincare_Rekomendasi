# Skincare Recommendation System

Sistem rekomendasi produk skincare berbasis web menggunakan Flask dan algoritma K-Nearest Neighbors (KNN).

## 📋 Fitur Utama

- **Sistem Rekomendasi**: Algoritma Content-Based Filtering + KNN untuk rekomendasi produk skincare yang dipersonalisasi
- **User Management**: Registrasi, login, dan manajemen profil pengguna
- **Admin Panel**: Manajemen produk, user, dan data sistem
- **Database**: MySQL
- **Responsive Design**: Interface yang mobile-friendly

## 🧠 Algoritma Rekomendasi

### Content-Based Filtering (CBF) + K-Nearest Neighbors (KNN)

Sistem ini menggunakan kombinasi **Content-Based Filtering** dan **K-Nearest Neighbors** untuk memberikan rekomendasi yang akurat:

#### 1. Content-Based Filtering
- **TF-IDF Vectorization**: Mengubah teks produk (nama, brand, deskripsi) menjadi vektor numerik
- **Feature Engineering**: Menggabungkan fitur teks untuk analisis konten
- **Cosine Similarity**: Mengukur kesamaan antara preferensi user dan produk

#### 2. K-Nearest Neighbors (KNN)
- **Distance Calculation**: Menghitung jarak antara user profile dan setiap produk
- **Neighbor Selection**: Memilih K produk terdekat berdasarkan similarity score
- **Ranking**: Mengurutkan rekomendasi berdasarkan tingkat kesamaan

#### 3. Cara Kerja Sistem
1. **User Input**: User mengisi preferensi (jenis kulit, masalah kulit, kata kunci)
2. **Text Processing**: Sistem membersihkan dan memproses teks preferensi
3. **Vectorization**: Mengubah preferensi user menjadi TF-IDF vector
4. **Similarity Calculation**: Menghitung cosine similarity dengan semua produk
5. **KNN Selection**: Memilih K produk dengan similarity tertinggi
6. **Result Ranking**: Mengurutkan dan menampilkan rekomendasi dengan penjelasan

#### 4. Keunggulan Algoritma
- **Personalisasi Tinggi**: Rekomendasi berdasarkan konten yang sesuai preferensi
- **Explainable AI**: Setiap rekomendasi disertai penjelasan mengapa cocok
- **Scalable**: Dapat menangani dataset produk yang besar
- **Real-time**: Perhitungan cepat untuk pengalaman user yang responsif

## 🛠️ Teknologi yang Digunakan

- **Backend**: Python Flask
- **Database**: MySQL
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**: 
  - Scikit-learn (TF-IDF Vectorization, Cosine Similarity)
  - Pandas & NumPy (Data Processing)
  - Content-Based Filtering + KNN Algorithm
- **Dependencies**: Lihat `requirements.txt`

## 📦 Persyaratan Sistem

### Software yang Diperlukan:
- **Python 3.8+** (Direkomendasikan Python 3.9 atau lebih baru)
- **XAMPP** (untuk MySQL - opsional)
- **Git** (untuk clone repository)

### Python Packages:
Semua dependencies tercantum dalam `requirements.txt`

## 🚀 Instalasi dan Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd skincare_rekomendasi
```

### 2. Setup Python Virtual Environment (Direkomendasikan)
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Database

### 4. Setup Database MySQL

```bash
# 1. Install dan jalankan XAMPP
# 2. Start MySQL service di XAMPP Control Panel
# 3. Jalankan setup otomatis:
python setup_mysql.py
```

### 5. Inisialisasi Database
```bash
# Untuk MySQL
python -c "from config import DatabaseConfig; DatabaseConfig.init_database()"
```

### 6. Jalankan Aplikasi
```bash
python app.py
```

Aplikasi akan berjalan di: `http://localhost:5000`

## 🔧 Konfigurasi

### File `.env`
```env
# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_DEBUG=True

# Database Configuration
# Database type: mysql (SQLite support removed)
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3307
DB_USER=root
DB_PASSWORD=
DB_NAME=skincare_recommendation

# Recommendation Settings
KNN_K_VALUE=3
MAX_RECOMMENDATIONS=10

# Admin Credentials (for initial setup)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

### Konfigurasi Database
- **MySQL**: Pastikan MySQL berjalan di port yang dikonfigurasi (default: 3307)

## 👤 Akun Default

### Admin
- **Username**: `admin`
- **Password**: `admin123`
- **URL**: `http://localhost:5000/admin/login`

## 📁 Struktur Project

```
skincare_rekomendasi/
├── app.py                 # Main Flask application
├── config.py             # Database configuration
├── models.py             # Database models
├── recommender.py        # KNN recommendation algorithm
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── mysql_schema.sql      # MySQL database schema
├── setup_mysql.py        # MySQL setup automation
├── static/               # CSS, JS, Images
├── templates/            # HTML templates
└── database/             # Database utilities
```

## 🔧 Setup MySQL

### Setup Database MySQL:
```bash
# Setup MySQL terlebih dahulu
python setup_mysql.py
```

## 🖥️ Instalasi di Komputer Lain

### Langkah-langkah:

1. **Persiapan Komputer Target**
   ```bash
   # Install Python 3.8+
   # Download dari: https://www.python.org/downloads/
   
   # Install Git (opsional)
   # Download dari: https://git-scm.com/downloads
   
   # Install XAMPP (jika menggunakan MySQL)
   # Download dari: https://www.apachefriends.org/
   ```

2. **Copy Project**
   ```bash
   # Opsi A: Clone dari repository
   git clone <repository-url>
   
   # Opsi B: Copy folder project langsung
   # Copy seluruh folder skincare_rekomendasi
   ```

3. **Setup Environment**
   ```bash
   cd skincare_rekomendasi
   
   # Buat virtual environment
   python -m venv venv
   venv\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Konfigurasi Database**
   ```bash
   # Untuk SQLite (mudah):
   # Tidak perlu konfigurasi tambahan
   
   # Untuk MySQL:
   # - Start XAMPP MySQL
   # - Jalankan: python setup_mysql.py
   ```

5. **Jalankan Aplikasi**
   ```bash
   python app.py
   ```

## 🐛 Troubleshooting

### Error: Module tidak ditemukan
```bash
pip install -r requirements.txt
```

### Error: Database connection failed
```bash
# Untuk MySQL: Pastikan MySQL service berjalan
netstat -an | findstr :3307
```

### Error: Port sudah digunakan
```bash
# Ubah port di app.py atau kill process yang menggunakan port 5000
netstat -ano | findstr :5000
```

### Error: Permission denied
```bash
# Jalankan sebagai administrator (Windows)
# Atau ubah permission folder (Linux/Mac)
```

## 📊 Penggunaan

### Untuk User:
1. Buka `http://localhost:5000`
2. Register akun baru atau login
3. Isi preferensi skincare di halaman preferences
4. Dapatkan rekomendasi produk yang dipersonalisasi

### Untuk Admin:
1. Buka `http://localhost:5000/admin/login`
2. Login dengan akun admin
3. Kelola produk, user, dan data sistem

## 🔒 Keamanan

### Untuk Production:
1. **Ubah SECRET_KEY** di `.env`
2. **Ubah password admin** default
3. **Set FLASK_DEBUG=False**
4. **Gunakan HTTPS**
5. **Setup firewall** untuk database

## 📝 Pengembangan

### Menambah Fitur Baru:
1. Buat route baru di `app.py`
2. Tambah model di `models.py` jika perlu
3. Buat template HTML di `templates/`
4. Update database schema jika perlu

### Testing:
```bash
# Test database connection
python -c "from config import DatabaseConfig; print('OK' if DatabaseConfig.get_connection() else 'FAIL')"

# Test recommendation algorithm
python -c "from recommender import SkincareRecommender; print('Recommender OK')"
```

## 📞 Support

Jika mengalami masalah:
1. Periksa log error di terminal
2. Pastikan semua dependencies terinstall
3. Periksa konfigurasi database
4. Restart aplikasi

## 📄 Lisensi

Project ini dibuat untuk tujuan edukasi dan pengembangan sistem rekomendasi.

---

**Happy Coding! 🚀**