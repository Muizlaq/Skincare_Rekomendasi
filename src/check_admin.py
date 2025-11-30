import mysql.connector
from config import Config
from models import Admin

def check_admin_table():
    try:
        conn = mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            port=Config.DB_PORT
        )
        cursor = conn.cursor()
        
        # Check if admin table exists
        cursor.execute("SHOW TABLES LIKE 'admin'")
        result = cursor.fetchone()
        print(f"Tabel admin ada: {result is not None}")
        
        if result:
            # Check admin data
            cursor.execute("SELECT * FROM admin")
            admins = cursor.fetchall()
            print(f"Jumlah admin: {len(admins)}")
            for admin in admins:
                print(f"Admin: {admin}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

def create_admin():
    try:
        # Use the model to create admin
        Admin.ensure_default_admin()
        print("Admin default berhasil dibuat/diverifikasi")
        
        # Check if admin exists
        admin = Admin.get_by_username('admin')
        print(f"Admin ditemukan: {admin is not None}")
        
        if admin:
            # Test authentication
            auth_result = Admin.authenticate('admin', 'admin123')
            print(f"Test autentikasi: {auth_result is not None}")
            
    except Exception as e:
        print(f"Error creating admin: {e}")

if __name__ == "__main__":
    print("=== Checking Admin Table ===")
    check_admin_table()
    
    print("\n=== Creating/Verifying Admin ===")
    create_admin()