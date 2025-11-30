#!/usr/bin/env python3
"""
Script to add k_value column to user_preferences table
"""

from config import DatabaseConfig

def add_k_value_column():
    """Add k_value column to user_preferences table if it doesn't exist"""
    try:
        # Check if k_value column already exists
        check_query = """
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'skincare_db' 
            AND TABLE_NAME = 'user_preferences' 
            AND COLUMN_NAME = 'k_value'
        """
        
        result = DatabaseConfig.execute_query(check_query, fetch=True)
        
        if result:
            print("Column k_value already exists in user_preferences table")
            return True
        
        # Add k_value column
        alter_query = """
            ALTER TABLE user_preferences 
            ADD COLUMN k_value INT DEFAULT 3
        """
        
        result = DatabaseConfig.execute_query(alter_query)
        
        if result is not None:
            print("Successfully added k_value column to user_preferences table")
            return True
        else:
            print("Failed to add k_value column")
            return False
            
    except Exception as e:
        print(f"Error adding k_value column: {e}")
        return False

def show_table_structure():
    """Show the current structure of user_preferences table"""
    try:
        query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'skincare_db' 
            AND TABLE_NAME = 'user_preferences'
            ORDER BY ORDINAL_POSITION
        """
        
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if result:
            print("\nCurrent structure of user_preferences table:")
            print("-" * 60)
            print(f"{'Column Name':<25} {'Type':<15} {'Nullable':<10} {'Default'}")
            print("-" * 60)
            for row in result:
                default_val = row['COLUMN_DEFAULT'] if row['COLUMN_DEFAULT'] else 'NULL'
                print(f"{row['COLUMN_NAME']:<25} {row['DATA_TYPE']:<15} {row['IS_NULLABLE']:<10} {default_val}")
        else:
            print("Could not retrieve table structure")
            
    except Exception as e:
        print(f"Error showing table structure: {e}")

if __name__ == "__main__":
    print("Checking and fixing k_value column in user_preferences table...")
    
    # Show current structure
    show_table_structure()
    
    # Add k_value column if needed
    success = add_k_value_column()
    
    if success:
        print("\nUpdated table structure:")
        show_table_structure()
    else:
        print("Failed to update table structure")