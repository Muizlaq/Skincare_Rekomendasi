import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Flask configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database settings
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = int(os.environ.get('DB_PORT', 3306))
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'skincare_recommendation')
    DB_USE_SSL = os.environ.get('DB_USE_SSL', 'false').lower() == 'true'
    DB_SSL_CA = os.environ.get('DB_SSL_CA')
    
    DB_TYPE = os.environ.get('DB_TYPE', 'mysql')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    DATABASE_URI = os.environ.get('DATABASE_URI') or f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Recommendation settings
    KNN_K_VALUE = int(os.environ.get('KNN_K_VALUE', 3))
    MAX_RECOMMENDATIONS = int(os.environ.get('MAX_RECOMMENDATIONS', 10))
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

class DatabaseConfig:
    """Database connection configuration"""
    
    @staticmethod
    def get_connection():
        t = (Config.DB_TYPE or '').lower()
        if t in ('postgres', 'postgresql'):
            try:
                import psycopg
                from psycopg.rows import dict_row
                dsn = Config.DATABASE_URL
                if not dsn:
                    dsn = (
                        f"dbname={Config.DB_NAME} user={Config.DB_USER} password={Config.DB_PASSWORD} "
                        f"host={Config.DB_HOST} port={Config.DB_PORT} sslmode=require"
                    )
                conn = psycopg.connect(dsn, row_factory=dict_row)
                return conn
            except Exception as e:
                print(f"Error connecting to Postgres: {e}")
                return None
        else:
            import mysql.connector
            from mysql.connector import Error
            try:
                kwargs = {
                    'host': Config.DB_HOST,
                    'port': Config.DB_PORT,
                    'user': Config.DB_USER,
                    'password': Config.DB_PASSWORD,
                    'database': Config.DB_NAME,
                    'charset': 'utf8mb4',
                    'collation': 'utf8mb4_unicode_ci',
                    'autocommit': True
                }
                if Config.DB_USE_SSL:
                    kwargs['ssl_disabled'] = False
                    if Config.DB_SSL_CA:
                        kwargs['ssl_ca'] = Config.DB_SSL_CA
                connection = mysql.connector.connect(**kwargs)
                return connection
            except Error as e:
                print(f"Error connecting to MySQL: {e}")
                return None
    
    @staticmethod
    def execute_query(query, params=None, fetch=False):
        connection = DatabaseConfig.get_connection()
        if not connection:
            return None
        try:
            t = (Config.DB_TYPE or '').lower()
            cursor = None
            if t in ('postgres', 'postgresql'):
                cursor = connection.cursor()
            else:
                cursor = connection.cursor(dictionary=True)
            q_upper = query.strip().upper()
            if not fetch and t in ('postgres', 'postgresql') and q_upper.startswith('INSERT') and 'RETURNING' not in q_upper:
                query = query + ' RETURNING id'
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                connection.commit()
                return row['id'] if row and 'id' in row else 1
            cursor.execute(query, params or ())
            if fetch:
                if 'SELECT' in q_upper:
                    result = cursor.fetchall()
                else:
                    result = cursor.fetchone()
            else:
                if q_upper.startswith('INSERT'):
                    if hasattr(cursor, 'lastrowid'):
                        result = cursor.lastrowid
                    else:
                        result = cursor.rowcount
                else:
                    result = cursor.rowcount
            connection.commit()
            return result
        except Exception as e:
            print(f"Database error: {e}")
            if hasattr(connection, 'rollback'):
                connection.rollback()
            return None
        finally:
            try:
                if cursor:
                    cursor.close()
            except Exception:
                pass
            try:
                if connection:
                    if hasattr(connection, 'is_connected'):
                        if connection.is_connected():
                            connection.close()
                    else:
                        connection.close()
            except Exception:
                pass
    
    @staticmethod
    def execute_many(query, data_list):
        """Execute multiple queries with data list"""
        connection = DatabaseConfig.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            cursor.executemany(query, data_list)
            connection.commit()
            return True
            
        except Exception as e:
            print(f"Database error: {e}")
            if hasattr(connection, 'rollback'):
                connection.rollback()
            return False
        finally:
            if connection:
                if hasattr(connection, 'is_connected'):
                    if connection.is_connected():
                        cursor.close()
                        connection.close()
                else:
                    connection.close()
    
    @staticmethod
    def init_database():
        conn = DatabaseConfig.get_connection()
        if not conn:
            return False
        try:
            t = (Config.DB_TYPE or '').lower()
            cursor = conn.cursor()
            if t in ('postgres', 'postgresql'):
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        nama_lengkap TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            else:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        nama_lengkap VARCHAR(100) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass