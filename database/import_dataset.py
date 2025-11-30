#!/usr/bin/env python3
"""
Dataset Import Script for Skincare Recommendation System
Imports Skincare_Dataset.csv into MySQL database
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import re
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatabaseConfig

class DatasetImporter:
    """Import CSV dataset to MySQL database"""
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.connection = None
    
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.connection = DatabaseConfig.get_connection()
            if self.connection:
                print("✅ Connected to MySQL database successfully")
                return True
            else:
                print("❌ Failed to connect to database")
                return False
        except Error as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def clean_numeric_value(self, value):
        """Clean numeric values from CSV (remove commas, plus signs, etc.)"""
        if pd.isna(value) or value == '':
            return 0
        
        # Convert to string and clean
        value_str = str(value).strip()
        
        # Remove common formatting
        value_str = value_str.replace(',', '')  # Remove commas
        value_str = value_str.replace('+', '')  # Remove plus signs
        value_str = value_str.replace('++', '') # Remove double plus
        value_str = value_str.replace('Rp', '') # Remove currency
        value_str = value_str.replace('.', '')  # Remove dots
        
        # Extract numbers only
        numbers = re.findall(r'\d+', value_str)
        if numbers:
            return int(numbers[0])
        
        return 0
    
    def clean_price(self, price_str):
        """Clean price string and convert to integer"""
        if pd.isna(price_str) or price_str == '':
            return 0
        
        # Remove currency symbols and formatting
        price_clean = str(price_str).replace('Rp', '').replace(',', '').replace('.', '').strip()
        
        # Extract numbers
        numbers = re.findall(r'\d+', price_clean)
        if numbers:
            return int(numbers[0])
        
        return 0
    
    def clean_rating(self, rating_str):
        """Clean rating string and convert to float"""
        if pd.isna(rating_str) or rating_str == '':
            return 0.0
        
        try:
            return float(str(rating_str).strip())
        except ValueError:
            return 0.0
    
    def load_and_clean_data(self):
        """Load CSV data and clean it"""
        try:
            print(f"📂 Loading dataset from: {self.csv_file_path}")
            
            # Read CSV file
            df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            print(f"📊 Loaded {len(df)} records from CSV")
            
            # Display column names
            print("📋 CSV Columns:", list(df.columns))
            
            # Clean column names (remove spaces, standardize)
            df.columns = df.columns.str.strip()
            
            # Map CSV columns to database columns
            column_mapping = {
                'no.': 'no_urut',
                'nama produk': 'nama_produk',
                'merk': 'brand',
                'terjual': 'terjual',
                'reviews': 'reviews',
                'bintang': 'rating_bintang',
                'marketplace': 'marketplace',
                'link': 'link_produk',
                'harga': 'harga',
                'deskripsi produk': 'deskripsi_produk'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Clean data
            print("🧹 Cleaning data...")
            
            # Clean numeric fields
            df['no_urut'] = df['no_urut'].apply(self.clean_numeric_value)
            df['terjual'] = df['terjual'].astype(str)  # Keep as string for display
            df['reviews'] = df['reviews'].astype(str)  # Keep as string for display
            df['rating_bintang'] = df['rating_bintang'].apply(self.clean_rating)
            df['harga'] = df['harga'].apply(self.clean_price)
            
            # Clean text fields
            df['nama_produk'] = df['nama_produk'].fillna('').astype(str).str.strip()
            df['brand'] = df['brand'].fillna('').astype(str).str.strip()
            df['marketplace'] = df['marketplace'].fillna('').astype(str).str.strip()
            df['link_produk'] = df['link_produk'].fillna('').astype(str).str.strip()
            df['deskripsi_produk'] = df['deskripsi_produk'].fillna('').astype(str).str.strip()
            
            # Remove rows with empty essential fields
            df = df[df['nama_produk'] != '']
            df = df[df['brand'] != '']
            df = df[df['harga'] > 0]
            
            print(f"✅ Data cleaned. {len(df)} valid records ready for import")
            return df
            
        except Exception as e:
            print(f"❌ Error loading/cleaning data: {e}")
            return None
    
    def import_data(self, df):
        """Import cleaned data to database"""
        if not self.connection:
            print("❌ No database connection")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Clear existing data
            print("🗑️ Clearing existing products...")
            cursor.execute("DELETE FROM products")
            cursor.execute("ALTER TABLE products AUTO_INCREMENT = 1")
            
            # Prepare insert query
            insert_query = """
                INSERT INTO products (no_urut, nama_produk, brand, terjual, reviews, 
                                    rating_bintang, marketplace, link_produk, harga, deskripsi_produk)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Convert DataFrame to list of tuples
            data_tuples = []
            for _, row in df.iterrows():
                data_tuple = (
                    int(row['no_urut']),
                    row['nama_produk'][:255],  # Limit length
                    row['brand'][:100],        # Limit length
                    row['terjual'][:50],       # Limit length
                    row['reviews'][:50],       # Limit length
                    float(row['rating_bintang']),
                    row['marketplace'][:50],   # Limit length
                    row['link_produk'],
                    int(row['harga']),
                    row['deskripsi_produk']
                )
                data_tuples.append(data_tuple)
            
            # Batch insert
            print(f"📥 Importing {len(data_tuples)} products...")
            cursor.executemany(insert_query, data_tuples)
            
            # Commit changes
            self.connection.commit()
            
            # Verify import
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]
            
            print(f"✅ Successfully imported {count} products to database")
            
            # Show sample data
            cursor.execute("SELECT id, nama_produk, brand, harga FROM products LIMIT 5")
            sample_data = cursor.fetchall()
            
            print("\n📋 Sample imported data:")
            for row in sample_data:
                print(f"  ID: {row[0]}, Product: {row[1][:50]}..., Brand: {row[2]}, Price: Rp{row[3]:,}")
            
            return True
            
        except Error as e:
            print(f"❌ Import error: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def close_connection(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Database connection closed")

def main():
    """Main function to run the import"""
    print("🚀 Starting Skincare Dataset Import")
    print("=" * 50)
    
    # Get CSV file path
    csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Skincare_Dataset.csv')
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Initialize importer
    importer = DatasetImporter(csv_file)
    
    try:
        # Connect to database
        if not importer.connect_database():
            return False
        
        # Load and clean data
        df = importer.load_and_clean_data()
        if df is None:
            return False
        
        # Import data
        success = importer.import_data(df)
        
        if success:
            print("\n🎉 Dataset import completed successfully!")
            print("✅ You can now run the Flask application")
        else:
            print("\n❌ Dataset import failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        importer.close_connection()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)