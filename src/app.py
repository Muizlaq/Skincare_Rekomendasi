import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(os.path.dirname(__file__))

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from config import Config
from models import User, Admin, Product, UserPreference, UserRating
from recommender_light import SkincareRecommender
import os

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))
app.config.from_object(Config)

# Initialize recommender
recommender = SkincareRecommender()

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with improved validation"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        nama_lengkap = request.form.get('nama_lengkap', '').strip()
        umur = request.form.get('umur', '').strip()
        
        # Check password confirmation
        if password != confirm_password:
            flash('Password dan konfirmasi password tidak sama!', 'error')
            return render_template('user/register.html')
        
        # Create user with validation
        result = User.create(username, email, password, nama_lengkap, umur if umur else None)
        
        if result['success']:
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
        else:
            # Display validation errors
            for error in result['errors']:
                flash(error, 'error')
    
    return render_template('user/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.authenticate(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_type'] = 'user'
            return redirect(url_for('user_dashboard'))
        else:
            flash('Username atau password salah!', 'error')
    
    return render_template('user/login.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        admin = Admin.authenticate(username, password)
        if admin:
            session['admin_id'] = admin['id']
            session['username'] = admin['username']
            session['user_type'] = 'admin'
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Username atau password admin salah!', 'error')
    
    return render_template('admin/login.html')

@app.route('/logout')
def logout():
    """Logout user/admin"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/user/dashboard')
def user_dashboard():
    """User dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.get_by_id(session['user_id'])
    user_preferences = UserPreference.get_by_user_id(session['user_id'])
    
    # Calculate recommendation count if preferences exist
    recommendation_count = 0
    recent_recommendations = []
    
    if user_preferences:
        try:
            from recommender_light import SkincareRecommender
            recommender = SkincareRecommender()
            
            # Convert user_preferences dict to include rentang_harga
            preferences_dict = dict(user_preferences)
            
            # rentang_harga already exists in database, no need to create it
            # Just ensure it has a default value if missing
            if 'rentang_harga' not in preferences_dict or not preferences_dict['rentang_harga']:
                preferences_dict['rentang_harga'] = '100000-200000'  # Default range
            
            # Ensure all required fields are present with defaults
            required_fields = {
                'frekuensi_pemakaian': preferences_dict.get('frekuensi_pemakaian', 'harian'),
                'bahan_aktif_efektif': preferences_dict.get('bahan_aktif_efektif', 'tidak_tahu'),
                'preferensi_produk': preferences_dict.get('preferensi_produk', 'semua'),
                'kata_kunci': preferences_dict.get('kata_kunci', '')
            }
            
            # Add missing fields to preferences
            for field, default_value in required_fields.items():
                if field not in preferences_dict or preferences_dict[field] is None:
                    preferences_dict[field] = default_value
            
            # Get recommendations with user's preferred k_value
            user_k_value = preferences_dict.get('k_value', 3)
            recommendations = recommender.get_recommendations(preferences_dict, k_value=user_k_value)
            recommendation_count = len(recommendations)
            recent_recommendations = recommendations[:3]  # Get first 3 for display
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            recommendation_count = 0
    
    return render_template('user/dashboard.html', 
                         user=user, 
                         user_preferences=user_preferences,
                         recommendation_count=recommendation_count,
                         recent_recommendations=recent_recommendations)

@app.route('/user/preferences', methods=['GET', 'POST'])
def user_preferences():
    """User preferences form"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        preference_data = {
            'user_id': session['user_id'],
            'kondisi_kulit': request.form['kondisi_kulit'],
            'masalah_kulit': request.form['masalah_kulit'],
            'rentang_harga': request.form['rentang_harga'],
            'frekuensi_pemakaian': request.form['frekuensi_penggunaan'],
            'bahan_aktif_efektif': request.form.get('efektivitas_bahan_aktif', ''),
            'preferensi_produk': request.form.get('preferensi_produk', ''),
            'kata_kunci': request.form.get('kata_kunci', ''),
            'k_value': int(request.form.get('k_value', 3))
        }
        
        # Save or update preferences
        if UserPreference.save(preference_data):
            flash('Preferensi berhasil disimpan!', 'success')
            return redirect(url_for('get_recommendations'))
        else:
            flash('Terjadi kesalahan saat menyimpan preferensi.', 'error')
    
    # Get existing preferences if any
    preferences = UserPreference.get_by_user_id(session['user_id'])
    return render_template('user/preferences.html', user_preferences=preferences)

@app.route('/user/recommendations')
def get_recommendations():
    """Get product recommendations"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    preferences = UserPreference.get_by_user_id(session['user_id'])
    if not preferences:
        flash('Silakan isi preferensi terlebih dahulu.', 'warning')
        return redirect(url_for('user_preferences'))
    
    # Get URL parameters for filtering and sorting
    search_query = request.args.get('search', '').strip()
    sort_by = request.args.get('sort_by', 'score')
    top_k_only = request.args.get('top_k_only', 'false').lower() == 'true'
    min_price = request.args.get('min_price', type=int)
    max_price = request.args.get('max_price', type=int)
    
    # Override budget range if URL parameters are provided
    if min_price is not None or max_price is not None:
        # Convert min/max price to rentang_harga enum
        if max_price is None:
            max_price = 1000000
        if min_price is None:
            min_price = 0
            
        # Determine rentang_harga based on price range
        if max_price <= 50000:
            preferences['rentang_harga'] = '0-50000'
        elif max_price <= 100000:
            preferences['rentang_harga'] = '50000-100000'
        elif max_price <= 200000:
            preferences['rentang_harga'] = '100000-200000'
        elif max_price <= 500000:
            preferences['rentang_harga'] = '200000-500000'
        else:
            preferences['rentang_harga'] = '500000+'
    
    # Ensure rentang_harga exists with default value
    if 'rentang_harga' not in preferences or not preferences['rentang_harga']:
        preferences['rentang_harga'] = '100000-200000'  # Default range
    
    # Add search query to preferences if provided
    if search_query:
        preferences['kata_kunci'] = search_query
    
    # Ensure all required fields are present with defaults
    required_fields = {
        'frekuensi_pemakaian': preferences.get('frekuensi_pemakaian', 'harian'),
        'bahan_aktif_efektif': preferences.get('bahan_aktif_efektif', 'tidak_tahu'),
        'preferensi_produk': preferences.get('preferensi_produk', 'semua'),
        'kata_kunci': preferences.get('kata_kunci', '')
    }
    
    # Add missing fields to preferences
    for field, default_value in required_fields.items():
        if field not in preferences or preferences[field] is None:
            preferences[field] = default_value
    
    # Get recommendations with user's preferred k_value
    user_k_value = preferences.get('k_value', 3)
    # If user selects sort_by=knn, force pure KNN (knn_only=True)
    knn_only = (sort_by == 'knn')
    recommendations = recommender.get_recommendations(preferences, k_value=user_k_value, knn_only=knn_only, top_k_only=top_k_only)
    
    # Apply sorting based on URL parameter
    if sort_by == 'price_low':
        recommendations.sort(key=lambda x: x['product']['price'])
    elif sort_by == 'price_high':
        recommendations.sort(key=lambda x: x['product']['price'], reverse=True)
    elif sort_by == 'rating':
        recommendations.sort(key=lambda x: x['product']['rating'], reverse=True)
    # elif sort_by == 'knn': keep as-is (already sorted by KNN inside recommender)
    # Default is 'score' which is already sorted by the recommender
    
    return render_template('user/recommendations.html', 
                         recommendations=recommendations, 
                         preferences=preferences)

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    # Get statistics
    total_users = User.count()
    total_products = Product.count()
    total_preferences = UserPreference.count()
    total_ratings = 0
    try:
        UserRating.ensure_table()
        total_ratings = UserRating.count()
    except Exception:
        total_ratings = 0
    
    # Get users created today (simplified - using total for now)
    new_users_today = 0  # Could be enhanced with date filtering
    
    # Get recommendation statistics (simplified - using placeholder values)
    total_recommendations = total_preferences * 10  # Estimate based on preferences
    recommendations_today = 0  # Could be enhanced with actual tracking
    
    stats = {
        'total_users': total_users,
        'total_products': total_products,
        'total_preferences': total_preferences,
        'users_with_preferences': total_preferences,  # Same as total_preferences
        'new_users_today': new_users_today,
        'active_products': total_products,  # Same as total_products
        'total_recommendations': total_recommendations,
        'recommendations_today': recommendations_today,
        'total_ratings': total_ratings
    }
    
    return render_template('admin/dashboard.html', stats=stats)

@app.route('/user/rate', methods=['POST'])
def rate_product():
    """Allow logged-in user to rate a recommended product (1-5)."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Harus login terlebih dahulu'}), 401
    try:
        UserRating.ensure_table()
        product_id = int(request.form.get('product_id', 0))
        rating = int(request.form.get('rating', 0))
        note = request.form.get('note', '')
        if product_id <= 0:
            return jsonify({'success': False, 'message': 'Produk tidak valid'}), 400
        if rating < 1 or rating > 5:
            return jsonify({'success': False, 'message': 'Rating harus 1-5'}), 400
        ok = UserRating.upsert(session['user_id'], product_id, rating, note)
        if ok:
            return jsonify({'success': True, 'message': 'Terima kasih atas penilaian Anda!'})
        else:
            return jsonify({'success': False, 'message': 'Gagal menyimpan penilaian'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/admin/ratings')
def admin_ratings():
    """Admin view to summarize ratings."""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    try:
        UserRating.ensure_table()
        summary = UserRating.summary_by_product(limit=100)
        latest = UserRating.latest(limit=30)
    except Exception as e:
        summary = []
        latest = []
        flash(f'Gagal memuat data rating: {str(e)}', 'error')
    return render_template('admin/ratings.html', summary=summary, latest=latest)

@app.route('/admin/users')
def admin_users():
    """Admin users management with search and pagination"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    per_page = 10
    
    if search:
        result = User.search(search, page, per_page)
    else:
        result = User.get_paginated(page, per_page)
    
    return render_template('admin/users.html', 
                         users=result['users'],
                         pagination=result,
                         search=search)

@app.route('/admin/user/create', methods=['GET', 'POST'])
def admin_create_user():
    """Create new user by admin"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        nama_lengkap = request.form.get('nama_lengkap', '').strip()
        
        result = User.create(username, email, password, nama_lengkap)
        
        if result['success']:
            flash('User berhasil dibuat!', 'success')
            return redirect(url_for('admin_users'))
        else:
            for error in result['errors']:
                flash(error, 'error')
    
    return render_template('admin/create_user.html')

@app.route('/admin/user/edit/<int:user_id>', methods=['GET', 'POST'])
def admin_edit_user(user_id):
    """Edit user by admin"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    user = User.get_by_id(user_id)
    if not user:
        flash('User tidak ditemukan!', 'error')
        return redirect(url_for('admin_users'))
    
    if request.method == 'POST':
        data = {}
        
        username = request.form.get('username', '').strip()
        if username and username != user['username']:
            # Check if username already exists
            existing = User.get_by_username(username)
            if existing and existing['id'] != user_id:
                flash('Username sudah digunakan!', 'error')
                return render_template('admin/edit_user.html', user=user)
            data['username'] = username
        
        email = request.form.get('email', '').strip()
        if email and email != user['email']:
            # Check if email already exists
            existing = User.get_by_email(email)
            if existing and existing['id'] != user_id:
                flash('Email sudah digunakan!', 'error')
                return render_template('admin/edit_user.html', user=user)
            data['email'] = email
        
        nama_lengkap = request.form.get('nama_lengkap', '').strip()
        if nama_lengkap:
            data['nama_lengkap'] = nama_lengkap
        
        password = request.form.get('password', '').strip()
        if password:
            data['password'] = password
        
        if data:
            if User.update(user_id, data):
                flash('User berhasil diupdate!', 'success')
                return redirect(url_for('admin_users'))
            else:
                flash('Gagal mengupdate user!', 'error')
        else:
            flash('Tidak ada perubahan data!', 'info')
    
    return render_template('admin/edit_user.html', user=user)

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    """Delete user by admin"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    user = User.get_by_id(user_id)
    if not user:
        flash('User tidak ditemukan!', 'error')
        return redirect(url_for('admin_users'))
    
    if User.delete(user_id):
        flash(f'User {user["username"]} berhasil dihapus!', 'success')
    else:
        flash('Gagal menghapus user!', 'error')
    
    return redirect(url_for('admin_users'))

@app.route('/api/user/<int:user_id>')
def get_user_detail(user_id):
    """Get user detail for AJAX requests"""
    if 'admin_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.get_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Remove password from response
    user_data = dict(user)
    user_data.pop('password', None)
    
    # Format created_at if exists
    if user_data.get('created_at'):
        user_data['created_at'] = user_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify(user_data)

@app.route('/admin/products')
def admin_products():
    """Admin products management"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '')
    brand = request.args.get('brand', '')
    sort = request.args.get('sort', 'rating')
    
    products = Product.get_paginated_with_filters(page, per_page, search, brand, sort)
    return render_template('admin/products.html', products=products)

@app.route('/admin/product/create', methods=['GET', 'POST'])
def admin_create_product():
    """Create new product"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            brand = request.form['brand']
            category = request.form['category']
            price = float(request.form['price'])
            description = request.form['description']
            ingredients = request.form.get('ingredients', '')
            skin_type = request.form.get('skin_type', '')
            rating = float(request.form.get('rating', 0.0))
            image_url = request.form.get('image_url', '')
            link_produk = request.form.get('link_produk', '')
            
            # Create product
            if Product.create(name, brand, category, price, description, ingredients, skin_type, rating, image_url, link_produk):
                flash('Produk berhasil ditambahkan!', 'success')
                return redirect(url_for('admin_products'))
            else:
                flash('Terjadi kesalahan saat menambahkan produk.', 'error')
        except ValueError as e:
            flash('Data tidak valid. Pastikan harga dan rating berupa angka.', 'error')
        except Exception as e:
            flash(f'Terjadi kesalahan: {str(e)}', 'error')
    
    return render_template('admin/create_product.html')

@app.route('/admin/product/edit/<int:product_id>', methods=['GET', 'POST'])
def admin_edit_product(product_id):
    """Edit product"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    product = Product.get_by_id(product_id)
    if not product:
        flash('Produk tidak ditemukan!', 'error')
        return redirect(url_for('admin_products'))
    
    if request.method == 'POST':
        try:
            update_data = {
                'name': request.form['name'],
                'brand': request.form['brand'],
                'category': request.form['category'],
                'price': float(request.form['price']),
                'description': request.form['description'],
                'ingredients': request.form.get('ingredients', ''),
                'skin_type': request.form.get('skin_type', ''),
                'rating': float(request.form.get('rating', 0.0)),
                'image_url': request.form.get('image_url', ''),
                'link_produk': request.form.get('link_produk', '')
            }
            
            if Product.update(product_id, update_data):
                flash('Produk berhasil diupdate!', 'success')
                return redirect(url_for('admin_products'))
            else:
                flash('Terjadi kesalahan saat mengupdate produk.', 'error')
        except ValueError as e:
            flash('Data tidak valid. Pastikan harga dan rating berupa angka.', 'error')
        except Exception as e:
            flash(f'Terjadi kesalahan: {str(e)}', 'error')
    
    return render_template('admin/edit_product.html', product=product)

@app.route('/admin/product/delete/<int:product_id>', methods=['POST'])
def admin_delete_product(product_id):
    """Delete product"""
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    if Product.delete(product_id):
        flash('Produk berhasil dihapus!', 'success')
    else:
        flash('Terjadi kesalahan saat menghapus produk.', 'error')
    
    return redirect(url_for('admin_products'))

@app.route('/api/product/<int:product_id>')
def get_product_detail(product_id):
    """API endpoint to get product details"""
    try:
        product = Product.get_by_id(product_id)
        if product:
            # Convert to dict and ensure all fields are present
            product_data = dict(product)
            
            # Map database fields to expected frontend fields
            product_data['nama_produk'] = product_data.get('name', 'Tidak diketahui')
            product_data['jenis_produk'] = product_data.get('skin_type', 'Tidak diketahui')
            product_data['bahan_utama'] = product_data.get('ingredients', 'Tidak diketahui')
            product_data['kondisi_kulit'] = product_data.get('skin_type', 'Semua jenis')
            product_data['rating'] = product_data.get('rating', 0)
            product_data['harga'] = f"Rp {product_data.get('price', 0):,.0f}"
            product_data['deskripsi'] = product_data.get('description', 'Deskripsi tidak tersedia')
            product_data['link_produk'] = product_data.get('link_produk', '')
            
            return jsonify(product_data)
        else:
            return jsonify({'error': 'Produk tidak ditemukan'}), 404
    except Exception as e:
        return jsonify({'error': 'Gagal memuat detail produk. Silakan coba lagi.'}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
