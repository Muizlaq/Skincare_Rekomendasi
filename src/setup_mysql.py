#!/usr/bin/env python3
"""
Script setup MySQL untuk Skincare Recommendation System
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_xampp_installation():
    """Cek apakah XAMPP terinstall"""
    xampp_paths = [
        "C:\\xampp",
        "C:\\Program Files\\xampp",
        "C:\\Program Files (x86)\\xampp"
    ]
    
    for path in xampp_paths:
        if os.path.exists(path):
            return path
    
    return None

def start_mysql_service():
    """Mencoba start MySQL service"""
    xampp_path = check_xampp_installation()
    
    if not xampp_path:
        print("❌ XAMPP tidak ditemukan!")
        print("Silakan install XAMPP terlebih dahulu dari: https://www.apachefriends.org/")
        return False
    
    print(f"✓ XAMPP ditemukan di: {xampp_path}")
    
    # Coba start MySQL melalui XAMPP
    mysql_exe = os.path.join(xampp_path, "mysql", "bin", "mysqld.exe")
    
    if not os.path.exists(mysql_exe):
        print(f"❌ MySQL executable tidak ditemukan di: {mysql_exe}")
        return False
    
    print("🚀 Mencoba menjalankan MySQL...")
    
    # Cek apakah MySQL sudah berjalan
    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True)
        if ":3307" in result.stdout:
            print("✓ MySQL sudah berjalan di port 3307")
            return True
    except:
        pass
    
    print("⚠️  MySQL belum berjalan. Silakan:")
    print("1. Buka XAMPP Control Panel")
    print("2. Klik 'Start' pada MySQL")
    print("3. Tunggu hingga status menjadi hijau")
    print("4. Jalankan script ini lagi")
    
    return False

def test_mysql_connection():
    """Test koneksi ke MySQL"""
    try:
        import mysql.connector
        from mysql.connector import Error
        
        connection = mysql.connector.connect(
            host='localhost',
            port=3307,
            user='root',
            password=''
        )
        
        if connection.is_connected():
            print("✓ Koneksi ke MySQL berhasil!")
            connection.close()
            return True
        else:
            print("❌ Gagal terhubung ke MySQL")
            return False
            
    except Error as e:
        print(f"❌ Error koneksi MySQL: {e}")
        return False
    except ImportError:
        print("❌ Module mysql-connector-python belum terinstall")
        print("Jalankan: pip install mysql-connector-python")
        return False

def install_mysql_connector():
    """Install mysql-connector-python jika belum ada"""
    try:
        import mysql.connector
        print("✓ mysql-connector-python sudah terinstall")
        return True
    except ImportError:
        print("📦 Menginstall mysql-connector-python...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mysql-connector-python"])
            print("✓ mysql-connector-python berhasil diinstall")
            return True
        except subprocess.CalledProcessError:
            print("❌ Gagal menginstall mysql-connector-python")
            return False

def create_database():
    """Membuat database MySQL"""
    try:
        import mysql.connector
        from mysql.connector import Error
        
        connection = mysql.connector.connect(
            host='localhost',
            port=3307,
            user='root',
            password=''
        )
        
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS skincare_recommendation CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print("✓ Database 'skincare_recommendation' berhasil dibuat")
        
        connection.close()
        return True
        
    except Error as e:
        print(f"❌ Error membuat database: {e}")
        return False

def run_schema():
    """Menjalankan schema MySQL"""
    schema_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "mysql_schema.sql")
    
    if not os.path.exists(schema_file):
        print(f"❌ File {schema_file} tidak ditemukan")
        return False
    
    try:
        import mysql.connector
        from mysql.connector import Error
        
        connection = mysql.connector.connect(
            host='localhost',
            port=3307,
            user='root',
            password='',
            database='skincare_recommendation'
        )
        
        cursor = connection.cursor()
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            sql_commands = f.read().split(';')
            
        for command in sql_commands:
            command = command.strip()
            if command and not command.startswith('--') and not command.upper().startswith('CREATE DATABASE'):
                try:
                    cursor.execute(command)
                except Error as e:
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")
        
        connection.commit()
        connection.close()
        print("✓ Schema MySQL berhasil dijalankan")
        return True
        
    except Error as e:
        print(f"❌ Error menjalankan schema: {e}")
        return False

def update_env_file():
    """Update file .env untuk menggunakan MySQL"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print(f"❌ File {env_file} tidak ditemukan")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Update DB_TYPE ke mysql
        if "DB_TYPE=sqlite" in content:
            content = content.replace("DB_TYPE=sqlite", "DB_TYPE=mysql")
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✓ File .env berhasil diupdate untuk menggunakan MySQL")
            return True
        elif "DB_TYPE=mysql" in content:
            print("✓ File .env sudah dikonfigurasi untuk MySQL")
            return True
        else:
            print("⚠️  DB_TYPE tidak ditemukan di .env, menambahkan konfigurasi...")
            content = content.replace("# Database Configuration", "# Database Configuration\nDB_TYPE=mysql")
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✓ Konfigurasi DB_TYPE ditambahkan ke .env")
            return True
            
    except Exception as e:
        print(f"❌ Error updating .env: {e}")
        return False

def main():
    """Fungsi utama setup MySQL"""
    print("=" * 60)
    print("SETUP MYSQL UNTUK SKINCARE RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    steps = [
        ("Menginstall mysql-connector-python", install_mysql_connector),
        ("Memeriksa MySQL service", start_mysql_service),
        ("Testing koneksi MySQL", test_mysql_connection),
        ("Membuat database", create_database),
        ("Menjalankan schema", run_schema),
        ("Update konfigurasi .env", update_env_file)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"❌ Gagal: {step_name}")
            break
    
    print("\n" + "=" * 60)
    
    if success_count == len(steps):
        print("🎉 SETUP MYSQL BERHASIL!")
        print("=" * 60)
        print("Langkah selanjutnya:")
        print("1. Restart aplikasi Flask")
        print("2. Aplikasi sekarang menggunakan database MySQL")
        print("3. Jika ingin migrasi data dari SQLite, jalankan: py migrate_to_mysql.py")
    else:
        print("❌ SETUP MYSQL GAGAL!")
        print("=" * 60)
        print("Silakan periksa error di atas dan coba lagi.")
        print("Pastikan:")
        print("- XAMPP sudah terinstall")
        print("- MySQL service di XAMPP sudah running")
        print("- Tidak ada firewall yang memblokir port 3307")

if __name__ == "__main__":
    main()