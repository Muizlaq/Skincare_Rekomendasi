-- MySQL Schema untuk Skincare Recommendation System
-- Sesuai dengan struktur SQLite yang digunakan aplikasi

-- Buat database jika belum ada
CREATE DATABASE IF NOT EXISTS skincare_recommendation 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE skincare_recommendation;

-- Tabel users
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    age INT,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel user_preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    skin_type VARCHAR(50),
    skin_concerns TEXT,
    budget_range VARCHAR(50),
    preferred_brands TEXT,
    allergies TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel products
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    brand VARCHAR(100),
    category VARCHAR(100),
    price DECIMAL(10,2),
    description TEXT,
    ingredients TEXT,
    skin_type VARCHAR(100),
    rating DECIMAL(3,2),
    image_url VARCHAR(500),
    link_produk VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_brand (brand),
    INDEX idx_category (category),
    INDEX idx_skin_type (skin_type),
    INDEX idx_rating (rating)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel recommendations
CREATE TABLE IF NOT EXISTS recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    score DECIMAL(5,3),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    INDEX idx_user_product (user_id, product_id),
    INDEX idx_score (score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel admin_users
CREATE TABLE IF NOT EXISTS admin_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    full_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'admin',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabel admin (untuk kompatibilitas dengan schema lama)
CREATE TABLE IF NOT EXISTS admin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default admin user
INSERT IGNORE INTO admin_users (username, password_hash, email, full_name, role)
VALUES ('admin', MD5('admin123'), 'admin@skincare.com', 'Administrator', 'admin');

-- Insert default admin untuk kompatibilitas
INSERT IGNORE INTO admin (username, password, email)
VALUES ('admin', MD5('admin123'), 'admin@skincare.com');

-- Insert sample products data
INSERT IGNORE INTO products (id, name, brand, category, price, description, ingredients, skin_type, rating, image_url, link_produk) VALUES
(1, 'Kahf Face Wash Brightening', 'Kahf', 'Face Wash', 25000, 'Sabun cuci muka untuk mencerahkan wajah', 'Niacinamide, Vitamin C', 'Normal,Oily', 4.2, 'static/images/no-image.png', 'https://shopee.co.id/search?keyword=kahf%20facial%20wash%20brightening'),
(2, 'Kahf Sunscreen Moisturizer SPF 30', 'Kahf', 'Sunscreen', 35000, 'Pelembab dengan perlindungan UV SPF 30', 'Zinc Oxide, Titanium Dioxide', 'All', 4.3, 'static/images/no-image.png', 'https://shopee.co.id/search?keyword=kahf%20sunscreen%20moisturizer'),
(3, 'Kahf Face Wash Oil & Acne Care', 'Kahf', 'Face Wash', 25000, 'Sabun cuci muka untuk kulit berminyak dan berjerawat', 'Salicylic Acid, Tea Tree Oil', 'Oily,Acne', 4.1, 'static/images/no-image.png', 'https://shopee.co.id/search?keyword=kahf%20facial%20wash%20oil%20acne'),
(5, 'Nivea Men Facial Foam Oil Clear Acne Defense', 'Nivea Men', 'Face Wash', 18000, 'Sabun cuci muka untuk pria dengan formula anti-acne', 'Salicylic Acid, Magnolia Extract', 'Oily,Acne', 4.0, 'static/images/no-image.png', 'https://shopee.co.id/search?keyword=nivea%20men%20facial%20foam%20acne%20defense'),
(45, 'Bromen Oil Control Facial Wash', 'Bromen', 'Face Wash', 15000, 'Sabun cuci muka untuk mengontrol minyak berlebih', 'Charcoal, Tea Tree Oil', 'Oily', 3.9, 'static/images/no-image.png', 'https://shopee.co.id/search?keyword=bromen%20oil%20control%20facial%20wash');

COMMIT;