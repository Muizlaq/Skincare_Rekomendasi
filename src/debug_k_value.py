#!/usr/bin/env python3
"""
Debug script to check k_value functionality
"""

from config import DatabaseConfig

def check_user_preferences():
    """Check all user preferences and their k_value"""
    try:
        # Get all user preferences
        query = """
            SELECT up.*, u.username 
            FROM user_preferences up 
            JOIN users u ON up.user_id = u.id 
            ORDER BY up.created_at DESC
        """
        
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if result:
            print("Current User Preferences:")
            print("=" * 80)
            for pref in result:
                print(f"User: {pref['username']} (ID: {pref['user_id']})")
                print(f"K-Value: {pref.get('k_value', 'NOT SET')}")
                print(f"Kondisi Kulit: {pref['kondisi_kulit']}")
                print(f"Masalah Kulit: {pref['masalah_kulit']}")
                print(f"Created: {pref['created_at']}")
                print("-" * 40)
        else:
            print("No user preferences found")
            
    except Exception as e:
        print(f"Error checking user preferences: {e}")

def check_table_structure():
    """Check user_preferences table structure"""
    try:
        query = """
            DESCRIBE user_preferences
        """
        
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if result:
            print("\nTable Structure:")
            print("=" * 60)
            for column in result:
                print(f"{column['Field']:<25} {column['Type']:<15} {column['Null']:<8} {column['Default']}")
        else:
            print("Could not get table structure")
            
    except Exception as e:
        print(f"Error checking table structure: {e}")

def test_k_value_update():
    """Test updating k_value for a user"""
    try:
        # Get first user
        query = "SELECT user_id FROM user_preferences LIMIT 1"
        result = DatabaseConfig.execute_query(query, fetch=True)
        
        if result:
            user_id = result[0]['user_id']
            
            # Update k_value to 7
            update_query = "UPDATE user_preferences SET k_value = %s WHERE user_id = %s"
            update_result = DatabaseConfig.execute_query(update_query, (7, user_id))
            
            if update_result:
                print(f"\nSuccessfully updated k_value to 7 for user_id {user_id}")
                
                # Verify the update
                verify_query = "SELECT k_value FROM user_preferences WHERE user_id = %s"
                verify_result = DatabaseConfig.execute_query(verify_query, (user_id,), fetch=True)
                
                if verify_result:
                    print(f"Verified k_value: {verify_result[0]['k_value']}")
                else:
                    print("Could not verify update")
            else:
                print("Failed to update k_value")
        else:
            print("No users found to test with")
            
    except Exception as e:
        print(f"Error testing k_value update: {e}")

if __name__ == "__main__":
    print("Debugging K-Value functionality...")
    print()
    
    check_table_structure()
    check_user_preferences()
    test_k_value_update()
    
    print("\nDebug complete!")