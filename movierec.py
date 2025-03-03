# import faiss
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sqlalchemy import create_engine
# from config import DB_CONFIG

# # Load FAISS index
# index = faiss.read_index("data/faiss_index.bin")

# # Load TF-IDF vectorizer
# with open("data/tfidf_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Load movie titles dictionary (to avoid CSV storage)
# with open("data/movie_titles.pkl", "rb") as f:
#     movie_titles = pickle.load(f)

# # Database Connection
# DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# engine = create_engine(DATABASE_URL)

# def fetch_movie_features():
#     """Fetch only necessary features from the database."""
#     query = """
#         SELECT id, title, overview, genres, keywords
#         FROM movie
#         LIMIT 42174
#     """
#     df = pd.read_sql(query, engine)
    
#     # Strip spaces and make title comparison case-insensitive
#     df['title'] = df['title'].str.strip()
    
#     df['combined_features'] = (
#         df['genres'].fillna('') + " " +
#         df['keywords'].fillna('') + " " +
#         df['overview'].fillna('')
#     )
#     return df

# def recommend_movies(movie_title, top_n=10):
#     """ Recommend similar movies using FAISS """
#     df = fetch_movie_features()

#     # Case-insensitive check for movie title
#     if movie_title.lower() not in df['title'].str.lower().values:
#         print(f"Movie '{movie_title}' not found in dataset.")
#         return []

#     movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]

#     # Transform query movie into vector
#     query_vector = vectorizer.transform([df.iloc[movie_index]['combined_features']]).toarray().astype('float32')

#     # Search for similar movies
#     distances, indices = index.search(query_vector, top_n + 1)  # +1 to exclude itself

#     recommendations = [movie_titles[i] for i in indices[0][1:]]  # Skip first (self)
#     return recommendations

# # Example: Get recommendations for a movie
# if __name__ == "__main__":
#     movie_name = "major"  # Change this movie title
#     suggestions = recommend_movies(movie_name)
#     print(f"Recommended movies for '{movie_name}':", suggestions)




# import faiss
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sqlalchemy import create_engine
# from config import DB_CONFIG

# # Load FAISS Index
# index = faiss.read_index("data/faiss_index.bin")

# # Load TF-IDF Vectorizer
# with open("data/tfidf_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Load movie titles dictionary
# with open("data/movie_titles.pkl", "rb") as f:
#     movie_titles = pickle.load(f)

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

# def recommend_movies(movie_title, top_n=10):
#     """Recommend similar movies using optimized FAISS."""
#     df = fetch_movie_features()

#     # Case-insensitive search
#     if movie_title.lower() not in df['title'].str.lower().values:
#         print(f"❌ Movie '{movie_title}' not found in dataset.")
#         return []

#     movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]

#     # Convert query movie into vector
#     query_vector = vectorizer.transform([df.iloc[movie_index]['combined_features']]).toarray().astype('float32')

#     # Search for similar movies using IVF
#     distances, indices = index.search(query_vector, top_n + 1)  # +1 to exclude itself

#     recommendations = [movie_titles[i] for i in indices[0][1:]]  # Skip first (self)
#     return recommendations

# # Example: Get recommendations
# if __name__ == "__main__":
#     movie_name = "major"  # Change this movie title
#     suggestions = recommend_movies(movie_name)
#     print(f"🎬 Recommended movies for '{movie_name}':", suggestions)





# import faiss
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from config import DB_CONFIG
# from sqlalchemy import create_engine

# # Load FAISS index
# index = faiss.read_index("data/faiss_index.bin")

# # Load TF-IDF vectorizer
# with open("data/tfidf_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Load the SVD model (TruncatedSVD)
# with open("data/svd_model.pkl", "rb") as f:
#     svd = pickle.load(f)

# # Load movie titles dictionary
# with open("data/movie_titles.pkl", "rb") as f:
#     movie_titles = pickle.load(f)

# # Database Connection
# DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# engine = create_engine(DATABASE_URL)

# def fetch_movie_features():
#     """Fetch only necessary features from the database."""
#     query = """
#         SELECT id, title, overview, genres, keywords
#         FROM movie
#         LIMIT 42174
#     """
#     df = pd.read_sql(query, engine)
#     df['combined_features'] = (
#         df['genres'].fillna('') + " " +
#         df['keywords'].fillna('') + " " +
#         df['overview'].fillna('')
#     )
#     return df

# def recommend_movies(movie_titles_list, top_n=10):
#     """ Recommend similar movies using FAISS """
#     df = fetch_movie_features()

#     # Fetch the index of the movies in the dataset
#     movie_indices = []
#     movie_titles_lower = df['title'].str.lower().tolist()
#     input_movies_lower = set(title.lower() for title in movie_titles_list)
    
#     for movie_title in movie_titles_list:
#         if movie_title.lower() not in movie_titles_lower:
#             print(f"Movie '{movie_title}' not found in dataset.")
#             continue
#         movie_index = movie_titles_lower.index(movie_title.lower())  # Get index
#         movie_indices.append(movie_index)

#     # Transform the combined features of the provided movies into vectors
#     query_vectors = []
#     for movie_index in movie_indices:
#         query_vector = vectorizer.transform([df.iloc[movie_index]['combined_features']]).toarray().astype('float32')
#         reduced_query_vector = svd.transform(query_vector)  # Apply dimensionality reduction (SVD)
#         query_vectors.append(reduced_query_vector)

#     # Combine the vectors of the movies (averaging their vectors)
#     combined_query_vector = np.mean(query_vectors, axis=0)

#     # Search exactly for `top_n + 1` results (FAISS may return input movie)
#     distances, indices = index.search(combined_query_vector, top_n + len(input_movies_lower))  

#     # Get recommended movie titles while ensuring no input movies are included
#     recommended_movies = []

#     for i in indices[0]:  # Loop through FAISS results
#         movie_title = movie_titles[i]
#         if movie_title.lower() not in input_movies_lower:  # Skip input movies
#             recommended_movies.append(movie_title)
#         if len(recommended_movies) == top_n:  # Stop once we have 10 movies
#             break

#     return recommended_movies

# # Example: Get recommendations for a list of movies
# if __name__ == "__main__":
#     movie_list = ["get out", "inception"]  # Replace with the list of movie titles
#     suggestions = recommend_movies(movie_list, top_n=10)
#     print(f"Combined recommended movies for {movie_list}: {suggestions}")

import faiss
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG
from sklearn.feature_extraction.text import TfidfVectorizer

DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

# Load Pre-trained Models
with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

with open("data/movie_titles.pkl", "rb") as f:
    movie_titles = pickle.load(f)

# ✅ Load HNSW Index
faiss_index = faiss.read_index("data/hnsw_index.bin")  # Load trained HNSW index

def fetch_movie_data():
    """Fetch movie data without reprocessing."""
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
    
    movie_dict = {title: idx for idx, title in enumerate(df['title_lower'])}
    
    return df, movie_dict

movie_df, movie_dict = fetch_movie_data()

def recommend_movies(movie_titles_list, top_n=10):
    """Recommend similar movies using the FAISS HNSW index."""
    
    input_movies_lower = set(title.lower() for title in movie_titles_list)
    movie_indices = [movie_dict[title] for title in input_movies_lower if title in movie_dict]

    if not movie_indices:
        print("No valid movies found in dataset.")
        return []

    query_vectors = vectorizer.transform(movie_df.loc[movie_indices, 'combined_features']).toarray()
    reduced_query_vectors = svd.transform(query_vectors).astype(np.float32)

    # Average all query vectors into a single vector
    combined_query_vector = np.mean(reduced_query_vectors, axis=0, keepdims=True)

    # Perform the search using HNSW
    distances, indices = faiss_index.search(combined_query_vector, top_n + len(input_movies_lower))

    recommended_movies = [
        movie_titles[i] for i in indices[0] if movie_titles[i].lower() not in input_movies_lower
    ][:top_n]

    return recommended_movies

# Example: Get recommendations for a movie
if __name__ == "__main__":
    movie_list = ["inception","major"]
    suggestions = recommend_movies(movie_list, top_n=10)
    print(f"🎬 Recommended movies for {movie_list}: {suggestions}")

