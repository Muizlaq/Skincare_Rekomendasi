#!/usr/bin/env python3
"""
Script untuk membuat database dan tabel yang diperlukan
"""

import mysql.connector
from mysql.connector import Error
from config import Config
import os

def create_database_and_tables():
    """Membuat database dan tabel yang diperlukan"""
    
    # Koneksi tanpa database terlebih dahulu
    try:
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        
        cursor = connection.cursor()
        
        # Buat database jika belum ada
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"Database '{Config.DB_NAME}' berhasil dibuat atau sudah ada.")
        
        # Gunakan database
        cursor.execute(f"USE {Config.DB_NAME}")
        
        # Baca dan eksekusi schema.sql
        schema_path = os.path.join(os.path.dirname(__file__), 'database', 'schema.sql')
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as file:
                schema_content = file.read()
                
            # Split berdasarkan delimiter dan eksekusi satu per satu
            statements = [stmt.strip() for stmt in schema_content.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    try:
                        cursor.execute(statement)
                        print(f"✓ Berhasil eksekusi: {statement[:50]}...")
                    except Error as e:
                        print(f"✗ Error eksekusi statement: {e}")
                        print(f"Statement: {statement[:100]}...")
        
        connection.commit()
        print("✓ Database dan tabel berhasil dibuat!")
        
        # Test koneksi dengan database
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"Tabel yang tersedia: {[table[0] for table in tables]}")
        
        return True
        
    except Error as e:
        print(f"Error: {e}")
        return False
    
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def test_connection():
    """Test koneksi database"""
    try:
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        
        if connection.is_connected():
            print("✓ Koneksi database berhasil!")
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"MySQL Version: {version[0]}")
            return True
            
    except Error as e:
        print(f"✗ Koneksi database gagal: {e}")
        return False
    
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    print("=== Setup Database Skincare Recommendation ===")
    print(f"Host: {Config.DB_HOST}:{Config.DB_PORT}")
    print(f"Database: {Config.DB_NAME}")
    print(f"User: {Config.DB_USER}")
    print()
    
    # Test koneksi MySQL server
    print("1. Testing MySQL server connection...")
    if create_database_and_tables():
        print("\n2. Testing database connection...")
        if test_connection():
            print("\n✓ Setup database berhasil!")
        else:
            print("\n✗ Test koneksi database gagal!")
    else:
        print("\n✗ Setup database gagal!")
        print("\nPastikan MySQL server berjalan di:")
        print(f"- Host: {Config.DB_HOST}")
        print(f"- Port: {Config.DB_PORT}")
        print(f"- User: {Config.DB_USER} memiliki akses untuk membuat database")