import json
import os
from config import Config

def handler(request):
    try:
        info = {
            'vercel': os.environ.get('VERCEL') == '1',
            'debug': Config.DEBUG,
            'db_host': Config.DB_HOST,
            'db_port': Config.DB_PORT,
            'db_user_set': bool(Config.DB_USER),
            'db_name': Config.DB_NAME,
            'db_use_ssl': getattr(Config, 'DB_USE_SSL', False)
        }
        return json.dumps({'ok': True, 'env': info}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)}), 500, {'Content-Type': 'application/json'}