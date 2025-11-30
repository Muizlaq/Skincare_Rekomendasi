from werkzeug.security import generate_password_hash, check_password_hash
from config import DatabaseConfig
from datetime import datetime
import re
 

class User:
    """User model for handling user operations"""
    
    @staticmethod
    def validate_registration_data(username, email, password, nama_lengkap, umur=None):
        """Validate registration data"""
        errors = []
        
        # Username validation
        if not username or len(username.strip()) < 3:
            errors.append("Username minimal 3 karakter")
        elif len(username) > 50:
            errors.append("Username maksimal 50 karakter")
        elif not re.match("^[a-zA-Z0-9_]+$", username):
            errors.append("Username hanya boleh mengandung huruf, angka, dan underscore")
        
        # Email validation
        if not email or not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            errors.append("Format email tidak valid")
        elif len(email) > 100:
            errors.append("Email maksimal 100 karakter")
        
        # Password validation
        if not password or len(password) < 6:
            errors.append("Password minimal 6 karakter")
        elif len(password) > 255:
            errors.append("Password maksimal 255 karakter")
        
        # Full name validation
        if not nama_lengkap or len(nama_lengkap.strip()) < 2:
            errors.append("Nama lengkap minimal 2 karakter")
        elif len(nama_lengkap) > 100:
            errors.append("Nama lengkap maksimal 100 karakter")
        
        # Age validation (optional)
        if umur is not None:
            try:
                age = int(umur)
                if age < 13 or age > 120:
                    errors.append("Umur harus antara 13-120 tahun")
            except (ValueError, TypeError):
                errors.append("Umur harus berupa angka")
        
        return errors
    
    @staticmethod
    def create(username, email, password, nama_lengkap, umur=None):
        """Create new user with validation"""
        # Validate input data
        errors = User.validate_registration_data(username, email, password, nama_lengkap, umur)
        if errors:
            return {'success': False, 'errors': errors}
        
        # Check if user already exists
        if User.get_by_username(username):
            return {'success': False, 'errors': ['Username sudah terdaftar']}
        
        if User.get_by_email(email):
            return {'success': False, 'errors': ['Email sudah terdaftar']}
        
        # Create user
        hashed_password = generate_password_hash(password)
        
        # Note: Current database schema doesn't have 'age' column, so we ignore umur parameter
        query = f"""
            INSERT INTO users (username, email, password, nama_lengkap)
            VALUES (%s, %s, %s, %s)
        """
        params = (username.strip(), email.strip(), hashed_password, nama_lengkap.strip())
        
        result = DatabaseConfig.execute_query(query, params)
        
        if result:
            return {'success': True, 'user_id': result}
        else:
            return {'success': False, 'errors': ['Terjadi kesalahan saat menyimpan data']}
    
    @staticmethod
    def get_by_username(username):
        """Get user by username"""
        query = f"SELECT * FROM users WHERE username = %s"
        result = DatabaseConfig.execute_query(query, (username,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_by_email(email):
        """Get user by email"""
        query = f"SELECT * FROM users WHERE email = %s"
        result = DatabaseConfig.execute_query(query, (email,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_by_id(user_id):
        """Get user by ID"""
        query = f"SELECT * FROM users WHERE id = %s"
        result = DatabaseConfig.execute_query(query, (user_id,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def authenticate(username, password):
        """Authenticate user"""
        user = User.get_by_username(username)
        if user and check_password_hash(user['password'], password):
            return user
        return None
    
    @staticmethod
    def get_all():
        """Get all users"""
        query = f"SELECT id, username, email, nama_lengkap, created_at FROM users ORDER BY created_at DESC"
        return DatabaseConfig.execute_query(query, fetch=True) or []
    
    @staticmethod
    def count():
        """Count total users"""
        query = f"SELECT COUNT(*) as total FROM users"
        result = DatabaseConfig.execute_query(query, fetch=True)
        return result[0]['total'] if result else 0
    
    @staticmethod
    def update(user_id, data):
        """Update user data"""
        # Build dynamic update query
        fields = []
        params = []
        
        if 'username' in data and data['username']:
            fields.append("username = %s")
            params.append(data['username'].strip())
        
        if 'email' in data and data['email']:
            fields.append("email = %s")
            params.append(data['email'].strip())
        
        if 'nama_lengkap' in data and data['nama_lengkap']:
            fields.append("nama_lengkap = %s")
            params.append(data['nama_lengkap'].strip())
        
        if 'password' in data and data['password']:
            fields.append("password = %s")
            params.append(generate_password_hash(data['password']))
        
        if not fields:
            return False
        
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"
        
        return DatabaseConfig.execute_query(query, params) is not None
    
    @staticmethod
    def delete(user_id):
        """Delete user"""
        # First delete user preferences
        query_prefs = f"DELETE FROM user_preferences WHERE user_id = %s"
        DatabaseConfig.execute_query(query_prefs, (user_id,))
        
        # Then delete user
        query = f"DELETE FROM users WHERE id = %s"
        return DatabaseConfig.execute_query(query, (user_id,)) is not None
    
    @staticmethod
    def search(search_term, page=1, per_page=20):
        """Search users by username, email, or nama_lengkap"""
        offset = (page - 1) * per_page
        search_pattern = f"%{search_term}%"
        
        query = f"""
            SELECT id, username, email, nama_lengkap, created_at 
            FROM users 
            WHERE username LIKE %s OR email LIKE %s OR nama_lengkap LIKE %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (search_pattern, search_pattern, search_pattern, per_page, offset)
        results = DatabaseConfig.execute_query(query, params, fetch=True) or []
        
        # Get total count for pagination
        count_query = f"""
            SELECT COUNT(*) as total 
            FROM users 
            WHERE username LIKE %s OR email LIKE %s OR nama_lengkap LIKE %s
        """
        count_params = (search_pattern, search_pattern, search_pattern)
        count_result = DatabaseConfig.execute_query(count_query, count_params, fetch=True)
        total = count_result[0]['total'] if count_result else 0
        
        return {
            'users': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }
    
    @staticmethod
    def get_paginated(page=1, per_page=20):
        """Get paginated users"""
        offset = (page - 1) * per_page
        
        query = f"""
            SELECT id, username, email, nama_lengkap, created_at 
            FROM users 
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        results = DatabaseConfig.execute_query(query, (per_page, offset), fetch=True) or []
        
        # Get total count
        total = User.count()
        
        return {
            'users': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }

class Admin:
    """Admin model for handling admin operations"""
    
    @staticmethod
    def create(username, password, nama_admin='Administrator'):
        """Create new admin"""
        hashed_password = generate_password_hash(password)
        query = """
            INSERT INTO admin (username, password, nama_admin)
            VALUES (%s, %s, %s)
        """
        result = DatabaseConfig.execute_query(query, (username, hashed_password, nama_admin))
        return result > 0 if result else False
    
    @staticmethod
    def get_by_username(username):
        """Get admin by username"""
        query = f"SELECT * FROM admin WHERE username = %s"
        result = DatabaseConfig.execute_query(query, (username,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def authenticate(username, password):
        """Authenticate admin"""
        admin = Admin.get_by_username(username)
        if admin and check_password_hash(admin['password'], password):
            return admin
        return None
    
    @staticmethod
    def ensure_default_admin():
        """Ensure default admin exists"""
        if not Admin.get_by_username('admin'):
            Admin.create('admin', 'admin123', 'Administrator')

class Product:
    """Product model for handling product operations"""
    
    @staticmethod
    def create(name, brand, category, price, description, ingredients=None, skin_type=None, rating=0.0, image_url=None, link_produk=None):
        """Create new product"""
        query = f"""
            INSERT INTO products (nama_produk, brand, harga, deskripsi_produk, rating_bintang, link_produk)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        result = DatabaseConfig.execute_query(query, (name, brand, price, description, rating, link_produk))
        return result > 0 if result else False
    
    @staticmethod
    def get_by_id(product_id):
        """Get product by ID"""
        query = f"""
            SELECT id, nama_produk as name, brand, 'skincare' as category, harga as price, 
                   deskripsi_produk as description, '' as ingredients, '' as skin_type, 
                   rating_bintang as rating, '' as image_url,
                   created_at, updated_at, link_produk, marketplace
            FROM products WHERE id = %s
        """
        result = DatabaseConfig.execute_query(query, (product_id,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_all():
        """Get all products"""
        query = f"""
            SELECT id, nama_produk as name, brand, 'skincare' as category, harga as price, 
                   deskripsi_produk as description, '' as ingredients, '' as skin_type, 
                   rating_bintang as rating, '' as image_url,
                   created_at, updated_at, link_produk, marketplace
            FROM products ORDER BY rating_bintang DESC
        """
        return DatabaseConfig.execute_query(query, fetch=True) or []
    
    @staticmethod
    def get_paginated_with_filters(page=1, per_page=20, search='', brand='', sort='rating'):
        """Get paginated products with search and filter functionality"""
        offset = (page - 1) * per_page
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if search:
            where_conditions.append("(nama_produk LIKE %s OR brand LIKE %s)")
            params.extend([f'%{search}%', f'%{search}%'])
        
        if brand:
            where_conditions.append("brand = %s")
            params.append(brand)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Build ORDER BY clause
        order_by = "ORDER BY "
        if sort == 'name':
            order_by += "nama_produk ASC"
        elif sort == 'price_low':
            order_by += "harga ASC"
        elif sort == 'price_high':
            order_by += "harga DESC"
        else:  # default to rating
            order_by += "rating_bintang DESC"
        
        query = f"""
            SELECT id, nama_produk as name, brand, 'skincare' as category, harga as price, 
                   deskripsi_produk as description, '' as ingredients, '' as skin_type, 
                   rating_bintang as rating, '' as image_url,
                   created_at, updated_at
            FROM products 
            {where_clause}
            {order_by}
            LIMIT %s OFFSET %s
        """
        
        params.extend([per_page, offset])
        products = DatabaseConfig.execute_query(query, params, fetch=True) or []
        
        # Get total count with same filters
        count_query = f"SELECT COUNT(*) as total FROM products {where_clause}"
        count_params = params[:-2]  # Remove LIMIT and OFFSET params
        count_result = DatabaseConfig.execute_query(count_query, count_params, fetch=True)
        total = count_result[0]['total'] if count_result else 0
        
        # Get total unique brands count (without filters for accurate brand count)
        brand_count_query = f"SELECT COUNT(DISTINCT brand) as brand_count FROM products"
        brand_count_result = DatabaseConfig.execute_query(brand_count_query, fetch=True)
        total_brands = brand_count_result[0]['brand_count'] if brand_count_result else 0
        
        return {
            'products': products,
            'total': total,
            'total_brands': total_brands,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }

    @staticmethod
    def get_paginated(page=1, per_page=20):
        """Get paginated products"""
        offset = (page - 1) * per_page
        query = f"""
            SELECT id, nama_produk as name, brand, 'skincare' as category, harga as price, 
                   deskripsi_produk as description, '' as ingredients, '' as skin_type, 
                   rating_bintang as rating, '' as image_url,
                   created_at, updated_at
            FROM products 
            ORDER BY rating_bintang DESC 
            LIMIT %s OFFSET %s
        """
        products = DatabaseConfig.execute_query(query, (per_page, offset), fetch=True) or []
        
        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM products"
        count_result = DatabaseConfig.execute_query(count_query, fetch=True)
        total = count_result[0]['total'] if count_result else 0
        
        return {
            'products': products,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }
    
    @staticmethod
    def update(product_id, data):
        """Update product"""
        query = f"""
            UPDATE products 
            SET nama_produk = %s, brand = %s, harga = %s, deskripsi_produk = %s, 
                rating_bintang = %s, link_produk = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        result = DatabaseConfig.execute_query(query, (
            data.get('name'), data.get('brand'), data.get('price'), 
            data.get('description'), data.get('rating'), data.get('link_produk'), product_id
        ))
        return result > 0 if result else False
    
    @staticmethod
    def delete(product_id):
        """Delete product"""
        query = f"DELETE FROM products WHERE id = %s"
        result = DatabaseConfig.execute_query(query, (product_id,))
        return result > 0 if result else False
    
    @staticmethod
    def count():
        """Count total products"""
        query = f"SELECT COUNT(*) as total FROM products"
        result = DatabaseConfig.execute_query(query, fetch=True)
        return result[0]['total'] if result else 0
    
    @staticmethod
    def search_by_price_range(min_price, max_price):
        """Search products by price range"""
        query = f"SELECT * FROM products WHERE harga BETWEEN %s AND %s ORDER BY rating_bintang DESC"
        return DatabaseConfig.execute_query(query, (min_price, max_price), fetch=True) or []
    
    @staticmethod
    def search_by_keywords(keywords):
        """Search products by keywords in name or description"""
        query = f"""
            SELECT * FROM products 
            WHERE nama_produk LIKE %s OR deskripsi_produk LIKE %s 
            ORDER BY rating_bintang DESC
        """
        search_term = f"%{keywords}%"
        return DatabaseConfig.execute_query(query, (search_term, search_term), fetch=True) or []

class UserPreference:
    """User preference model for handling user preferences"""
    
    @staticmethod
    def ensure_table():
        """Create user_preferences table if it does not exist (using mapped table names)."""
        query = f"""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                kondisi_kulit ENUM('berminyak','kering','kombinasi','sensitif','normal') NOT NULL,
                usia ENUM('18-25','26-35','36-45','46+') NOT NULL,
                masalah_kulit ENUM('jerawat','komedo','kusam','kerutan','flek_hitam','pori_besar') NOT NULL,
                rentang_harga ENUM('0-50000','50000-100000','100000-200000','200000-500000','500000+') NOT NULL,
                efektivitas_bahan_aktif ENUM('rendah','sedang','tinggi') NOT NULL,
                preferensi_produk ENUM('cleanser','moisturizer','serum','sunscreen','toner','semua') NOT NULL,
                frekuensi_penggunaan ENUM('pagi','malam','pagi_malam') NOT NULL,
                kata_kunci_preferensi TEXT,
                k_value INT DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user_id (user_id),
                CONSTRAINT fk_up_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        return DatabaseConfig.execute_query(query) is not None
    
    @staticmethod
    def save(data):
        """Save or update user preferences"""
        user_id = data.get('user_id')
        try:
            UserPreference.ensure_table()
        except Exception:
            pass
        
        # Use rentang_harga directly from the form data
        rentang_harga = data.get('rentang_harga', '100000-200000')  # Default range
        
        # Map frekuensi_pemakaian to frekuensi_penggunaan
        frekuensi_map = {
            'harian': 'pagi_malam',
            'mingguan': 'pagi',
            'sesekali': 'malam'
        }
        frekuensi_penggunaan = frekuensi_map.get(data.get('frekuensi_pemakaian', 'harian'), data.get('frekuensi_pemakaian', 'pagi_malam'))
        
        # Map bahan_aktif_efektif to efektivitas_bahan_aktif
        efektivitas_map = {
            'tidak_tahu': 'sedang',
            'rendah': 'rendah',
            'sedang': 'sedang',
            'tinggi': 'tinggi'
        }
        efektivitas_bahan_aktif = efektivitas_map.get(data.get('bahan_aktif_efektif', 'tidak_tahu'), data.get('bahan_aktif_efektif', 'sedang'))
        
        # Set default age range
        usia = '18-25'  # Default age range
        
        # Check if preferences already exist
        existing = UserPreference.get_by_user_id(user_id)
        
        if existing:
            # Update existing preferences
            query = f"""
                UPDATE user_preferences 
                SET kondisi_kulit = %s, usia = %s, masalah_kulit = %s, rentang_harga = %s,
                    efektivitas_bahan_aktif = %s, preferensi_produk = %s, frekuensi_penggunaan = %s,
                    kata_kunci_preferensi = %s, k_value = %s
                WHERE user_id = %s
            """
            params = (
                data.get('kondisi_kulit'), usia, data.get('masalah_kulit'), 
                rentang_harga, efektivitas_bahan_aktif, data.get('preferensi_produk'),
                frekuensi_penggunaan, data.get('kata_kunci', ''), 
                int(data.get('k_value', 3)), user_id
            )
        else:
            # Insert new preferences
            query = f"""
                INSERT INTO user_preferences 
                (user_id, kondisi_kulit, usia, masalah_kulit, rentang_harga, 
                 efektivitas_bahan_aktif, preferensi_produk, frekuensi_penggunaan, kata_kunci_preferensi, k_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id, data.get('kondisi_kulit'), usia, data.get('masalah_kulit'),
                rentang_harga, efektivitas_bahan_aktif, data.get('preferensi_produk'),
                frekuensi_penggunaan, data.get('kata_kunci', ''), 
                int(data.get('k_value', 3))
            )
        
        result = DatabaseConfig.execute_query(query, params)
        return result > 0 if result else False
    
    @staticmethod
    def get_by_user_id(user_id):
        """Get user preferences by user ID"""
        try:
            UserPreference.ensure_table()
        except Exception:
            pass
        query = f"SELECT * FROM user_preferences WHERE user_id = %s"
        result = DatabaseConfig.execute_query(query, (user_id,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_all():
        """Get all user preferences with user info"""
        try:
            UserPreference.ensure_table()
        except Exception:
            pass
        query = f"""
            SELECT up.*, u.username, u.nama_lengkap as full_name 
            FROM user_preferences up 
            JOIN users u ON up.user_id = u.id 
            ORDER BY up.created_at DESC
        """
        return DatabaseConfig.execute_query(query, fetch=True) or []
    
    @staticmethod
    def count():
        """Count total user preferences"""
        try:
            UserPreference.ensure_table()
        except Exception:
            pass
        query = f"SELECT COUNT(*) as total FROM user_preferences"
        result = DatabaseConfig.execute_query(query, fetch=True)
        return result[0]['total'] if result else 0

class UserRating:
    """User ratings for products"""

    @staticmethod
    def ensure_table():
        """Create ratings table if it does not exist (MySQL)."""
        query = (
            f"""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                product_id INT NOT NULL,
                rating INT NOT NULL,
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_user_product (user_id, product_id),
                INDEX idx_product (product_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
        )
        return DatabaseConfig.execute_query(query) is not None

    @staticmethod
    def get_by_user_and_product(user_id, product_id):
        """Get existing rating by user for a product."""
        query = f"SELECT * FROM user_ratings WHERE user_id = %s AND product_id = %s"
        result = DatabaseConfig.execute_query(query, (user_id, product_id), fetch=True)
        return result[0] if result else None

    @staticmethod
    def upsert(user_id, product_id, rating, note=None):
        """Insert or update a rating (1-5)."""
        try:
            rating_int = int(rating)
        except Exception:
            return False
        if rating_int < 1:
            rating_int = 1
        if rating_int > 5:
            rating_int = 5

        existing = UserRating.get_by_user_and_product(user_id, product_id)
        if existing:
            query = f"UPDATE {TABLE_RATINGS} SET rating = %s, note = %s, created_at = CURRENT_TIMESTAMP WHERE id = %s"
            params = (rating_int, note, existing['id'])
        else:
            query = f"INSERT INTO {TABLE_RATINGS} (user_id, product_id, rating, note) VALUES (%s, %s, %s, %s)"
            params = (user_id, product_id, rating_int, note)
        result = DatabaseConfig.execute_query(query, params)
        return result is not None

    @staticmethod
    def count():
        """Count total ratings."""
        query = f"SELECT COUNT(*) as total FROM {TABLE_RATINGS}"
        result = DatabaseConfig.execute_query(query, fetch=True)
        return result[0]['total'] if result else 0

    @staticmethod
    def summary_by_product(limit=50):
        """Return rating count and average grouped by product."""
        query = (
            f"""
            SELECT p.id as product_id, p.nama_produk as name, p.brand,
                   COUNT(r.id) as rating_count,
                   ROUND(AVG(r.rating), 2) as avg_rating
            FROM {TABLE_PRODUCTS} p
            LEFT JOIN {TABLE_RATINGS} r ON r.product_id = p.id
            GROUP BY p.id, p.nama_produk, p.brand
            ORDER BY rating_count DESC, avg_rating DESC
            LIMIT %s
            """
        )
        return DatabaseConfig.execute_query(query, (limit,), fetch=True) or []

    @staticmethod
    def latest(limit=20):
        """Latest ratings with user and product info."""
        query = (
            f"""
            SELECT r.id, r.user_id, u.username, r.product_id, p.nama_produk as name,
                   r.rating, r.created_at
            FROM {TABLE_RATINGS} r
            JOIN {TABLE_USERS} u ON u.id = r.user_id
            JOIN {TABLE_PRODUCTS} p ON p.id = r.product_id
            ORDER BY r.created_at DESC
            LIMIT %s
            """
        )
        return DatabaseConfig.execute_query(query, (limit,), fetch=True) or []