import json
from config import DatabaseConfig

def handler(request):
    conn = None
    try:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return json.dumps({
                'ok': False,
                'message': 'Tidak dapat terhubung ke database. Cek DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME.'
            }), 500, {'Content-Type': 'application/json'}
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        return json.dumps({'ok': True, 'tables': tables}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)}), 500, {'Content-Type': 'application/json'}
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass