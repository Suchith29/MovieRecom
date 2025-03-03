import faiss
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DB_CONFIG
from sqlalchemy import create_engine

# Load FAISS index
index = faiss.read_index("data/faiss_index.bin")

# Load TF-IDF vectorizer
with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the SVD model (TruncatedSVD)
with open("data/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# Load movie titles dictionary
with open("data/movie_titles.pkl", "rb") as f:
    movie_titles = pickle.load(f)

# Database Connection
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

def fetch_movie_features():
    """Fetch only necessary features from the database."""
    query = """
        SELECT id, title, overview, genres, keywords
        FROM movie
        LIMIT 42174
    """
    df = pd.read_sql(query, engine)
    df['combined_features'] = (
        df['genres'].fillna('') + " " +
        df['keywords'].fillna('') + " " +
        df['overview'].fillna('')
    )
    return df

def recommend_movies(movie_titles_list, top_n=10):
    """ Recommend similar movies using FAISS with dynamic weights """
    df = fetch_movie_features()

    all_recommendations = set()  # Set to hold all unique recommendations

    # Fetch the index of the movies in the dataset
    movie_indices = []
    for movie_title in movie_titles_list:
        # Case-insensitive check for movie title
        if movie_title.lower() not in df['title'].str.lower().values:
            print(f"Movie '{movie_title}' not found in dataset.")
            continue
        # Get the index of the movie
        movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
        movie_indices.append(movie_index)

    # Dynamic weight assignment based on the number of movies
    num_movies = len(movie_titles_list)
    
    if num_movies == 1:
        weights = [1.0]  # If only one movie is given, give it 100% weight
    elif num_movies == 2:
        weights = [0.7, 0.3]  # 70% for the first, 30% for the second
    elif num_movies == 3:
        weights = [0.65, 0.2, 0.15]  # 65% for the first, 20% for the second, 15% for the third
    else:
        # For more than 3 movies, first movie gets at least 55%, and remaining weight is distributed.
        first_weight = 0.55
        remaining_weight = 1 - first_weight
        if num_movies > 3:
            weight_per_movie = remaining_weight / (num_movies - 1)
            weights = [first_weight] + [weight_per_movie] * (num_movies - 1)

    # Normalize the weights to ensure the sum is 1 (just in case)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Transform the combined features of the provided movies into vectors
    query_vectors = []
    for movie_index in movie_indices:
        query_vector = vectorizer.transform([df.iloc[movie_index]['combined_features']]).toarray().astype('float32')
        reduced_query_vector = svd.transform(query_vector)  # Apply dimensionality reduction (SVD)
        query_vectors.append(reduced_query_vector)

    # Apply weights to the vectors and calculate weighted average
    weighted_query_vector = np.zeros_like(query_vectors[0])  # Initialize with zeros (same shape as first query vector)
    for i, weight in enumerate(normalized_weights):
        weighted_query_vector += query_vectors[i] * weight  # Apply weight to the vector

    # Search for similar movies using FAISS
    distances, indices = index.search(weighted_query_vector, top_n + 1)  # +1 to exclude itself

    # Append the recommended movie titles and exclude input movies
    for idx in indices[0][1:]:  # Skip the first (self)
        recommended_movie = movie_titles[idx]
        if recommended_movie not in movie_titles_list:  # Exclude input movies
            all_recommendations.add(recommended_movie)

    return list(all_recommendations)[:top_n]  # Convert the set to a list and limit the number of recommendations

# Example: Get recommendations for a list of movies with dynamically assigned weights
if __name__ == "__main__":
    movie_list = ["Interstellar", "Inception", "rrr"]  # Replace with the list of movie titles
    
    # Get the recommendations based on the dynamic weights
    suggestions = recommend_movies(movie_list, top_n=10)
    print(f"Combined recommended movies for {movie_list}: {suggestions}")
