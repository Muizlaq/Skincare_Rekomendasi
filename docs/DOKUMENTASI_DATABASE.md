# Dokumentasi Database – Sistem Rekomendasi Skincare

## Ringkasan
- Jenis DB: MySQL (`utf8mb4_unicode_ci`).
- Nama DB default: `skincare_recommendation` (lihat `config/config.py`).
- Tabel inti: `admin`, `users`, `products`, `user_preferences`, `user_ratings` (dibuat otomatis).
- Sumber skema utama: `database/schema.sql` (+ migrasi `k_value`), dengan pembuatan tabel rating dari `src/models.py`.

## Konfigurasi Koneksi
- Host: `Config.DB_HOST` (default `localhost`).
- Port: `Config.DB_PORT` (default `3307`).
- User: `Config.DB_USER` (default `root`).
- Password: `Config.DB_PASSWORD` (default kosong).
- Database: `Config.DB_NAME` (default `skincare_recommendation`).
- File: `config/config.py`.

## Tabel dan Struktur

- `admin`
  - Kolom: `id` (PK, AUTO_INCREMENT), `username` (UNIQUE, NOT NULL), `password` (hash), `nama_admin`, `created_at`, `updated_at`.
  - Tujuan: akun admin untuk mengelola data.

- `users`
  - Kolom: `id` (PK), `username` (UNIQUE), `email` (UNIQUE), `password` (hash), `nama_lengkap`, `created_at`, `updated_at`.
  - Digunakan oleh model `User` (`src/models.py`).

- `products`
  - Kolom: `id` (PK), `no_urut`, `nama_produk`, `brand`, `terjual`, `reviews`, `rating_bintang` (DECIMAL), `marketplace`, `link_produk` (TEXT), `harga` (INT), `deskripsi_produk` (TEXT), `created_at`, `updated_at`.
  - Index: `idx_brand`, `idx_harga`, `idx_rating`, FULLTEXT `idx_nama_deskripsi (nama_produk, deskripsi_produk)`, tambahan `idx_products_search` dan `idx_products_price_rating`.
  - Populasi: melalui `database/import_dataset.py` dari `data/Skincare_Dataset.csv`.

- `user_preferences`
  - Kolom (schema awal `database/schema.sql`):
    - `id` (PK), `user_id` (FK → `users.id`),
    - `kondisi_kulit` ENUM('berminyak','kering','kombinasi','sensitif','normal'),
    - `usia` ENUM('18-25','26-35','36-45','46+'),
    - `masalah_kulit` ENUM('jerawat','komedo','kusam','kerutan','flek_hitam','pori_besar'),
    - `rentang_harga` ENUM('0-50000','50000-100000','100000-200000','200000-500000','500000+'),
    - `efektivitas_bahan_aktif` ENUM('rendah','sedang','tinggi'),
    - `preferensi_produk` ENUM('cleanser','moisturizer','serum','sunscreen','toner','semua'),
    - `frekuensi_penggunaan` ENUM('pagi','malam','pagi_malam'),
    - `kata_kunci_preferensi` (TEXT), `created_at`, `updated_at`.
  - Kolom tambahan (migrasi): `k_value` INT DEFAULT 3.
    - Ditambahkan oleh util `src/fix_k_value.py` (cek dan menambah bila belum ada). File `config/add_k_value_column.sql` bersifat legacy (mengarah ke `skincare_db`, jangan gunakan).
  - Index: `idx_user_id (user_id)`.
  - Digunakan oleh `UserPreference` di `src/models.py` (insert/update/select memerlukan `k_value`).

- `user_ratings` (dibuat otomatis)
  - Dibuat oleh `UserRating.ensure_table()` saat pertama kali digunakan (`src/models.py`).
  - Kolom: `id` (PK), `user_id` (FK → `users.id` ON DELETE CASCADE), `product_id` (FK → `products.id` ON DELETE CASCADE), `rating` (INT 1–5), `note` (TEXT), `created_at`, index `idx_user_product (user_id, product_id)`, `idx_product (product_id)`.
  - Operasi: upsert rating per `(user_id, product_id)`, rekap rata‑rata per produk, daftar rating terbaru.

## Relasi Utama
- `user_preferences.user_id` → `users.id` (FK, CASCADE delete).
- `user_ratings.user_id` → `users.id` (FK, CASCADE delete).
- `user_ratings.product_id` → `products.id` (FK, CASCADE delete).

## Alur Data Penting
- Registrasi pengguna → insert ke `users`.
- Pengaturan preferensi → upsert ke `user_preferences` termasuk `k_value`.
- Rekomendasi → `src/recommender.py` membaca preferensi, memfilter produk dari `products`, menghitung skor.
- Rating pengguna → `user_ratings.upsert()`; admin melihat ringkasan di `/admin/ratings`.
- Impor dataset → `database/import_dataset.py` membersihkan CSV lalu insert batch ke `products`.

## Inisialisasi & Migrasi
- Buat database/tabel: jalankan `tests/test_db.py` (akan mengeksekusi `database/schema.sql`).
- Pastikan kolom `k_value`: jalankan `python src/fix_k_value.py`.
- Cek struktur: `python check_schema.py` atau `python src/check_database.py` untuk memverifikasi keberadaan tabel/kolom.

## Perintah Berguna
- Import dataset: `python database/import_dataset.py`.
- Jalankan server: set `PYTHONPATH='src;config'` lalu `python -u src/app.py`.
- Uji sensitivitas K: `PYTHONPATH='src;config' python tests/test_k_sensitivity.py`.
- Debug K di preferensi: `PYTHONPATH='src;config' python src/debug_k_value.py`.

## Catatan Kompatibilitas Skema
- `config/mysql_schema.sql` adalah skema alternatif yang menambahkan tabel `admin_users` dan `recommendations`, serta field dengan penamaan bahasa Inggris (`name`, `category`, `rating`, dll.).
- Aplikasi saat ini menggunakan skema Indonesia dari `database/schema.sql` (mis. `nama_produk`, `rating_bintang`). Pastikan konsisten dengan skema ini untuk operasi sehari‑hari.

## Praktik Baik
- Gunakan FK dan index yang tersedia untuk menjaga integritas dan performa.
- Hindari perubahan skema tanpa memperbarui model (`src/models.py`) dan skrip impor.
- Dokumentasikan perubahan versi skema di `docs/` bila menambah kolom baru.