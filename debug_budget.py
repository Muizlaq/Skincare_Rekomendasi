import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from models import UserPreference
from config import DatabaseConfig

# Get user preferences
query = 'SELECT * FROM user_preferences ORDER BY id DESC LIMIT 1'
result = DatabaseConfig.execute_query(query, fetch=True)
if result:
    print('Latest user preferences:')
    for key, value in result[0].items():
        print(f'{key}: {value}')
    
    # Check budget values specifically
    budget_min = result[0].get('budget_min', 0)
    budget_max = result[0].get('budget_max', 0)
    print(f'\nBudget conversion:')
    print(f'budget_min: {budget_min} -> {budget_min * 1000}')
    print(f'budget_max: {budget_max} -> {budget_max * 1000}')
else:
    print('No user preferences found')

# Also check some products to see their prices
print('\nSample products with prices:')
query2 = 'SELECT nama_produk, harga FROM products ORDER BY harga LIMIT 10'
products = DatabaseConfig.execute_query(query2, fetch=True)
if products:
    for product in products:
        print(f"{product['nama_produk']}: {product['harga']}")