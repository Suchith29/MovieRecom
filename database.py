# import pymysql
# from config import DB_CONFIG

# def get_db_connection():
#     try:
#         conn = pymysql.connect(**DB_CONFIG)
#         return conn
#     except Exception as e:
#         print(f"Database connection failed: {e}")
#         return None

from sqlalchemy import create_engine
import pandas as pd
from config import DB_CONFIG

# Create a SQLAlchemy engine
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

def fetch_movie_data(limit=5000):
    query = f"""
        SELECT id, title, release_date, revenue, runtime, backdrop_path, budget, original_language, 
               overview, tagline, poster_path, genres, production_companies, production_countries, keywords
        FROM movie
        LIMIT {limit}
    """
    df = pd.read_sql(query, engine)  # Use SQLAlchemy engine instead of pymysql
    return df
