import faiss
import pickle
import numpy as np
import pandas as pd
import mariadb
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DB_CONFIG

# Database Connection
DATABASE_URL = f"mariadb+mariadbconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

# Fetching the movie data
def fetch_movie_data():
    """Fetch movie data (without reprocessing)."""
    query = text("""
        SELECT id, title, overview, genres, keywords
        FROM movie
    """)  # Ensure query is properly formatted as a SQLAlchemy text object

    with engine.connect() as connection:
        df = pd.read_sql_query(query, connection)  # Execute with a connection
        
    df['combined_features'] = (
        df['genres'].fillna('') + " " +
        df['keywords'].fillna('') + " " +
        df['overview'].fillna('')
    )
    df['title_lower'] = df['title'].str.lower()
    movie_dict = {title: idx for idx, title in enumerate(df['title_lower'])}
    id_dict = {idx: id for idx, id in enumerate(df['id'])}
    return df, movie_dict, id_dict


# Function to create HNSW index
def create_hnsw_index(data):
    """Create an HNSW index and save it to a file."""
    d = data.shape[1]  # Dimensionality of the data
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200  
    
    print("Adding vectors to HNSW index...")
    index.add(data)
    
    print("Saving the HNSW index to file...")
    faiss.write_index(index, "data/hnsw_index.bin")
    print("✅ HNSW Index saved successfully to data/hnsw_index.bin")

# Main code
if __name__ == "__main__":
    df, movie_dict, id_dict = fetch_movie_data()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    create_hnsw_index(reduced_matrix)

    with open("data/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("data/svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    with open("data/movie_ids.pkl", "wb") as f:
        pickle.dump(id_dict, f)

    print("Model training complete. FAISS index and models saved.")