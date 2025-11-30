-- Database: skincare_recommendation
-- Men's Skincare Product Recommendation System

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS skincare_recommendation 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE skincare_recommendation;

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS user_preferences;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS admin;

-- Table: admin
CREATE TABLE admin (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    nama_admin VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: users
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    nama_lengkap VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: products (based on dataset structure)
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    no_urut INT,
    nama_produk VARCHAR(255) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    terjual VARCHAR(50) DEFAULT '0',
    reviews VARCHAR(50) DEFAULT '0',
    rating_bintang DECIMAL(2,1) DEFAULT 0.0,
    marketplace VARCHAR(50),
    link_produk TEXT,
    harga INT NOT NULL,
    deskripsi_produk TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_brand (brand),
    INDEX idx_harga (harga),
    INDEX idx_rating (rating_bintang),
    FULLTEXT idx_nama_deskripsi (nama_produk, deskripsi_produk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: user_preferences
CREATE TABLE user_preferences (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    kondisi_kulit ENUM('berminyak', 'kering', 'kombinasi', 'sensitif', 'normal') NOT NULL,
    usia ENUM('18-25', '26-35', '36-45', '46+') NOT NULL,
    masalah_kulit ENUM('jerawat', 'komedo', 'kusam', 'kerutan', 'flek_hitam', 'pori_besar') NOT NULL,
    rentang_harga ENUM('0-50000', '50000-100000', '100000-200000', '200000-500000', '500000+') NOT NULL,
    efektivitas_bahan_aktif ENUM('rendah', 'sedang', 'tinggi') NOT NULL,
    preferensi_produk ENUM('cleanser', 'moisturizer', 'serum', 'sunscreen', 'toner', 'semua') NOT NULL,
    frekuensi_penggunaan ENUM('pagi', 'malam', 'pagi_malam') NOT NULL,
    kata_kunci_preferensi TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default admin
INSERT INTO admin (username, password, nama_admin) VALUES 
('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.s5uO.G', 'Administrator');
-- Default password: admin123

-- Create indexes for better performance
CREATE INDEX idx_products_search ON products(nama_produk(100), brand(50));
CREATE INDEX idx_products_price_rating ON products(harga, rating_bintang);

-- Show tables
SHOW TABLES;