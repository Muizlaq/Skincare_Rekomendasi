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
    DB_PORT = int(os.environ.get('DB_PORT', 3307))
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'skincare_recommendation')
    
    # Database type (mysql only)
    DB_TYPE = os.environ.get('DB_TYPE', 'mysql')  # Default to MySQL
    
    # Database connection string
    DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
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
        """Get MySQL database connection"""
        import mysql.connector
        from mysql.connector import Error
        
        try:
            connection = mysql.connector.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci',
                autocommit=True
            )
            return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    @staticmethod
    def execute_query(query, params=None, fetch=False):
        """Execute database query"""
        connection = DatabaseConfig.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if fetch:
                if 'SELECT' in query.upper():
                    result = cursor.fetchall()
                else:
                    result = cursor.fetchone()
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
            if connection:
                if hasattr(connection, 'is_connected'):
                    if connection.is_connected():
                        cursor.close()
                        connection.close()
                else:
                    connection.close()
    
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
        """Initialize MySQL database and tables"""
        # MySQL initialization would go here
        return True