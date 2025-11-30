import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from config import DatabaseConfig

# Check table structure with different queries
queries = [
    'SHOW COLUMNS FROM user_preferences',
    'SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = "user_preferences"',
    'SELECT * FROM user_preferences LIMIT 1'
]

for i, query in enumerate(queries):
    print(f'\n=== Query {i+1}: {query} ===')
    try:
        result = DatabaseConfig.execute_query(query, fetch=True)
        if result:
            for row in result:
                print(f'{row}')
        else:
            print('No results')
    except Exception as e:
        print(f'Error: {e}')