#!/usr/bin/env python3
"""
Script to check database connection and table structure
"""

from config import DatabaseConfig, Config

def check_database_connection():
    """Check database connection and show current database info"""
    try:
        # Show configuration
        print("Database Configuration:")
        print(f"Host: {Config.DB_HOST}")
        print(f"Port: {Config.DB_PORT}")
        print(f"User: {Config.DB_USER}")
        print(f"Database: {Config.DB_NAME}")
        print("-" * 50)
        
        # Test connection and show current database
        query = "SELECT DATABASE() as current_db"
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if result:
            print(f"Connected to database: {result[0]['current_db']}")
        
        # Show all databases
        query = "SHOW DATABASES"
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if result:
            print("\nAvailable databases:")
            for row in result:
                db_name = list(row.values())[0]
                print(f"  - {db_name}")
        
        return True
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

def check_user_preferences_table():
    """Check user_preferences table structure"""
    try:
        # Check if table exists
        query = """
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'user_preferences'
        """
        
        result = DatabaseConfig.execute_query(query, (Config.DB_NAME,), fetch=True)
        
        if not result:
            print(f"Table 'user_preferences' does not exist in database '{Config.DB_NAME}'")
            return False
        
        print(f"\nTable 'user_preferences' exists in database '{Config.DB_NAME}'")
        
        # Show table structure
        query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'user_preferences'
            ORDER BY ORDINAL_POSITION
        """
        
        result = DatabaseConfig.execute_query(query, (Config.DB_NAME,), fetch=True)
        
        if result:
            print("\nTable structure:")
            print("-" * 60)
            print(f"{'Column Name':<25} {'Type':<15} {'Nullable':<10} {'Default'}")
            print("-" * 60)
            
            k_value_exists = False
            for row in result:
                default_val = row['COLUMN_DEFAULT'] if row['COLUMN_DEFAULT'] else 'NULL'
                print(f"{row['COLUMN_NAME']:<25} {row['DATA_TYPE']:<15} {row['IS_NULLABLE']:<10} {default_val}")
                
                if row['COLUMN_NAME'] == 'k_value':
                    k_value_exists = True
            
            if k_value_exists:
                print("\n✓ k_value column exists")
            else:
                print("\n✗ k_value column is missing")
                
            return k_value_exists
        else:
            print("Could not retrieve table structure")
            return False
            
    except Exception as e:
        print(f"Error checking table structure: {e}")
        return False

def check_all_databases_for_table():
    """Check all databases for user_preferences table"""
    try:
        # Get all databases
        query = "SHOW DATABASES"
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if not result:
            print("Could not retrieve database list")
            return
        
        print("\nChecking all databases for user_preferences table:")
        print("-" * 60)
        
        for row in result:
            db_name = list(row.values())[0]
            
            # Skip system databases
            if db_name in ['information_schema', 'mysql', 'performance_schema', 'sys']:
                continue
            
            # Check if user_preferences table exists in this database
            query = """
                SELECT COUNT(*) as table_count
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'user_preferences'
            """
            
            result_count = DatabaseConfig.execute_query(query, (db_name,), fetch=True)
            
            if result_count and result_count[0]['table_count'] > 0:
                print(f"✓ {db_name} - has user_preferences table")
                
                # Check for k_value column
                query = """
                    SELECT COUNT(*) as column_count
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s 
                    AND TABLE_NAME = 'user_preferences'
                    AND COLUMN_NAME = 'k_value'
                """
                
                result_col = DatabaseConfig.execute_query(query, (db_name,), fetch=True)
                
                if result_col and result_col[0]['column_count'] > 0:
                    print(f"  └─ ✓ has k_value column")
                else:
                    print(f"  └─ ✗ missing k_value column")
            else:
                print(f"✗ {db_name} - no user_preferences table")
                
    except Exception as e:
        print(f"Error checking databases: {e}")

if __name__ == "__main__":
    print("Checking database connection and table structure...")
    print("=" * 60)
    
    # Check connection
    if check_database_connection():
        print("\n" + "=" * 60)
        
        # Check current database table
        check_user_preferences_table()
        
        print("\n" + "=" * 60)
        
        # Check all databases
        check_all_databases_for_table()
    else:
        print("Could not connect to database")