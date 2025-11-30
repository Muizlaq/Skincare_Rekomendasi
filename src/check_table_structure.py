from config import DatabaseConfig

try:
    # Check users table structure
    result = DatabaseConfig.execute_query('SHOW COLUMNS FROM users', fetch=True)
    print('Columns in users table:')
    for col in result:
        print(f'{col["Field"]} - {col["Type"]} - Null: {col["Null"]}')
    
    print('\nSample data from users table:')
    sample = DatabaseConfig.execute_query('SELECT * FROM users LIMIT 1', fetch=True)
    if sample:
        print('Sample row keys:', list(sample[0].keys()))
    else:
        print('No data in users table')
        
except Exception as e:
    print(f'Error: {e}')