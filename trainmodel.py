# import os
# import faiss
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sqlalchemy import create_engine
# from config import DB_CONFIG

# # Ensure 'data/' directory exists
# os.makedirs("data", exist_ok=True)

# # Database Connection
# DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# engine = create_engine(DATABASE_URL)

# def fetch_movie_data():
#     """Fetch movie data directly from the database."""
#     query = """
#         SELECT id, title, overview, genres, keywords
#         FROM movie
#         LIMIT 42174
#     """
#     df = pd.read_sql(query, engine)
#     return df

# def preprocess_data(df):
#     """Combine important text-based features for recommendation."""
#     df['combined_features'] = (
#         df['genres'].fillna('') + " " +
#         df['keywords'].fillna('') + " " +
#         df['overview'].fillna('')
#     )
#     return df

# # Load data from database
# movie_df = fetch_movie_data()
# movie_df = preprocess_data(movie_df)

# # Train TF-IDF model
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# tfidf_matrix = vectorizer.fit_transform(movie_df['combined_features'])

# # Convert sparse matrix to dense numpy array
# tfidf_matrix_dense = tfidf_matrix.toarray().astype('float32')

# # Train FAISS index
# index = faiss.IndexFlatL2(tfidf_matrix_dense.shape[1])  # L2 distance
# index.add(tfidf_matrix_dense)

# # Save the FAISS index and TF-IDF vectorizer
# faiss.write_index(index, "data/faiss_index.bin")
# with open("data/tfidf_vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# # Store movie titles in a dictionary (ID → Title) to avoid CSV storage
# movie_titles = {i: title for i, title in enumerate(movie_df['title'])}
# with open("data/movie_titles.pkl", "wb") as f:
#     pickle.dump(movie_titles, f)

# print("Model training complete. FAISS index and vectorizer saved in 'data/' directory.")




# import faiss
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sqlalchemy import create_engine
# from config import DB_CONFIG

# # Database Connection
# DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# engine = create_engine(DATABASE_URL)

# def fetch_movie_features():
#     """Fetch movie features from database."""
#     query = """
#         SELECT id, title, overview, genres, keywords
#         FROM movie
#         LIMIT 42174
#     """
#     df = pd.read_sql(query, engine)
#     df['title'] = df['title'].str.strip()

#     df['combined_features'] = (
#         df['genres'].fillna('') + " " +
#         df['keywords'].fillna('') + " " +
#         df['overview'].fillna('')
#     )
#     return df

# # Load movie dataset
# df = fetch_movie_features()

# # Convert text features to TF-IDF vectors
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# tfidf_matrix = vectorizer.fit_transform(df['combined_features']).toarray().astype('float32')

# # Save TF-IDF Vectorizer
# with open("data/tfidf_vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# # Save movie titles dictionary
# movie_titles = {i: title for i, title in enumerate(df['title'])}
# with open("data/movie_titles.pkl", "wb") as f:
#     pickle.dump(movie_titles, f)

# # **FAISS with Inverted File Index (IVF)**
# d = tfidf_matrix.shape[1]  # Dimension of vectors
# nlist = 100  # Number of clusters (adjustable)

# quantizer = faiss.IndexFlatL2(d)  # Base index
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# # Train the index
# index.train(tfidf_matrix)
# index.add(tfidf_matrix)

# # Save the FAISS index
# faiss.write_index(index, "data/faiss_index.bin")
# print("✅ FAISS Index trained & saved successfully!")






# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sqlalchemy import create_engine
# from config import DB_CONFIG
# import faiss
# from scipy.sparse import csr_matrix

# # Database Connection
# DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# engine = create_engine(DATABASE_URL)

# def fetch_movie_data(limit=42174):
#     """Fetch only necessary features from the database."""
#     query = f"""
#         SELECT id, title, overview, genres, keywords
#         FROM movie
#         LIMIT {limit}
#     """
#     df = pd.read_sql(query, engine)
#     df['combined_features'] = (
#         df['genres'].fillna('') + " " +
#         df['keywords'].fillna('') + " " +
#         df['overview'].fillna('')
#     )
#     return df

# def preprocess_and_reduce_dimensionality(df, n_components=100):
#     """Preprocess and reduce dimensionality using TruncatedSVD on sparse matrix."""
    
#     # Initialize the TF-IDF Vectorizer
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
    
#     # Apply TF-IDF vectorizer directly on the combined features (text data)
#     tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
#     # Perform TruncatedSVD for dimensionality reduction on sparse matrix
#     svd = TruncatedSVD(n_components=n_components, random_state=42)
#     reduced_matrix = svd.fit_transform(tfidf_matrix)
    
#     return reduced_matrix, vectorizer, svd

# def create_faiss_index(reduced_matrix):
#     """Create and save FAISS index from the reduced matrix."""
    
#     # Convert the reduced matrix to float32 (FAISS requirement)
#     reduced_matrix = reduced_matrix.astype(np.float32)
    
#     # Initialize FAISS index for L2 (Euclidean) distance
#     index = faiss.IndexFlatL2(reduced_matrix.shape[1])  # Create index
    
#     # Add the reduced matrix to the FAISS index
#     index.add(reduced_matrix)
    
#     # Save the FAISS index to disk
#     faiss.write_index(index, "data/faiss_index.bin")

# def save_models(vectorizer, svd):
#     """Save the models to disk."""
    
#     # Save the vectorizer (for transforming new queries)
#     with open("data/tfidf_vectorizer.pkl", "wb") as f:
#         pickle.dump(vectorizer, f)
    
#     # Save the SVD model
#     with open("data/svd_model.pkl", "wb") as f:
#         pickle.dump(svd, f)

# if __name__ == "__main__":
#     # Fetch movie data from database
#     df = fetch_movie_data()

#     # Preprocess and reduce dimensionality
#     reduced_matrix, vectorizer, svd = preprocess_and_reduce_dimensionality(df, n_components=100)

#     # Create and save FAISS index
#     create_faiss_index(reduced_matrix)

#     # Save vectorizer and SVD model for later use
#     save_models(vectorizer, svd)

#     # Optionally, save the reduced matrix for debugging/analysis (not necessary for FAISS)
#     np.save("data/reduced_matrix.npy", reduced_matrix)

#     print("Model training complete. FAISS index and models saved.")






import faiss
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DB_CONFIG

# Database Connection
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

# Fetching the movie data
def fetch_movie_data():
    """Fetch movie data (without reprocessing)."""
    query = """
        SELECT id, title, overview, genres, keywords
        FROM movie
        LIMIT 42174
    """
    df = pd.read_sql_query(query, engine)
    df['combined_features'] = (
        df['genres'].fillna('') + " " +
        df['keywords'].fillna('') + " " +
        df['overview'].fillna('')
    )
    df['title_lower'] = df['title'].str.lower()
    return df

# Function to create HNSW index
def create_hnsw_index(data):
    """Create an HNSW index and save it to a file."""
    d = data.shape[1]  # Dimensionality of the data
    # Build the HNSW index (M = 32, efConstruction = 200 for better balance between speed and accuracy)
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200  # Adjust for balance between speed and accuracy
    
    # Add data to the index
    print("Adding vectors to HNSW index...")
    index.add(data)
    
    # Save the index to a file
    print("Saving the HNSW index to file...")
    faiss.write_index(index, "data/hnsw_index.bin")
    print("✅ HNSW Index saved successfully to data/hnsw_index.bin")

# Main code
if __name__ == "__main__":
    # Fetch movie data
    df = fetch_movie_data()

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    # Perform dimensionality reduction using TruncatedSVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    # Create and save HNSW index
    create_hnsw_index(reduced_matrix)

    # Save vectorizer and SVD models for later use
    with open("data/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("data/svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    print("Model training complete. FAISS index and models saved.")


