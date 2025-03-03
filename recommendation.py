# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from database import get_db_connection
# from scipy.sparse import csr_matrix

# def fetch_movie_data(limit=5000):
#     conn = get_db_connection()
#     if not conn:
#         return None
#     query = """
#         SELECT id, title, release_date, revenue, runtime, backdrop_path, budget, original_language, 
#                overview, tagline, poster_path, genres, production_companies, production_countries, keywords
#         FROM movie
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# def preprocess_data(df):
#     # Combine genres, keywords, and overview into a single text feature
#     df['combined_features'] = (
#         df['genres'].fillna('') + " " +
#         df['keywords'].fillna('') + " " +
#         df['overview'].fillna('')
#     )
#     return df

# def recommend_movies(movie_title, df, top_n=10):
#     # Use sparse matrix to save memory
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
#     # Use sparse matrix for cosine similarity to optimize memory usage
#     similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

#     # Get the movie index from the title
#     movie_index = df[df['title'] == movie_title].index[0]

#     # Get the most similar movies
#     similar_movies = list(enumerate(similarity_matrix[movie_index]))
    
#     # Sort movies based on similarity score
#     sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
#     # Return the top N similar movie titles
#     recommendations = [df.iloc[i[0]]['title'] for i in sorted_movies]
#     return recommendations

# if __name__ == "__main__":
#     # Fetch and preprocess data
#     movie_df = fetch_movie_data(limit=5000)
#     if movie_df is not None:
#         movie_df = movie_df.head(5000)
#         movie_df = preprocess_data(movie_df)
        
#         # Example: Recommend movies based on "Inception"
#         movie_name = "Inception"  # Change this to test different movies
#         suggestions = recommend_movies(movie_name, movie_df)
        
#         print(f"Recommended movies for '{movie_name}':", suggestions)

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import TruncatedSVD

# # Function to fetch movie data (assuming your DB connection and query are correct)
# def fetch_movie_data():
#     query = "SELECT id, title, genres, keywords, overview FROM movies"
#     df = pd.read_sql(query, conn)  # This will load the data as a pandas DataFrame
#     return df.head(5000)  # Limit to first 5000 records to reduce memory load

# # Function to recommend movies based on similarity
# def recommend_movies(movie_name, movie_df):
#     # Initialize TfidfVectorizer with sparse matrix support
#     tfidf = TfidfVectorizer(stop_words='english', use_idf=True)

#     # Transform the 'overview' column into a sparse TF-IDF matrix
#     tfidf_matrix = tfidf.fit_transform(movie_df['overview'])

#     # Optionally, apply dimensionality reduction (Truncated SVD) to reduce matrix size
#     svd = TruncatedSVD(n_components=100, random_state=42)  # Reduce to 100 components
#     reduced_matrix = svd.fit_transform(tfidf_matrix)

#     # Calculate cosine similarity on the reduced matrix
#     similarity_matrix = cosine_similarity(reduced_matrix, reduced_matrix)

#     # Get the index of the movie_name
#     movie_idx = movie_df[movie_df['title'].str.contains(movie_name, case=False, na=False)].index[0]

#     # Get similar movies by sorting the similarity scores
#     similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
#     similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

#     # Get the top 5 similar movies
#     top_movies = [movie_df['title'].iloc[i[0]] for i in similarity_scores[1:6]]

#     return top_movies

# # Main execution block
# if __name__ == "__main__":
#     # Fetch movie data (make sure to pass the correct connection object)
#     movie_df = fetch_movie_data()

#     # Example movie to get recommendations for
#     movie_name = "Inception"  # Change this as per the movie you want to search

#     # Get movie recommendations
#     suggestions = recommend_movies(movie_name, movie_df)
    
#     # Output the recommendations
#     print("Recommended Movies:")
#     for suggestion in suggestions:
#         print(suggestion)
import pymysql
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from config import DB_CONFIG
from sqlalchemy import create_engine

# Create a SQLAlchemy engine
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

def get_db_connection():
    print('1')
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def fetch_movie_data():
    print(2)
    conn = get_db_connection()
    if not conn:
        return None
    
    query = """
        SELECT id, title, release_date, revenue, runtime, backdrop_path, budget, 
               original_language, overview, tagline, poster_path, genres, 
               production_companies, production_countries, keywords
        FROM movie
        LIMIT 25000
    """
    df = pd.read_sql(query, conn)
    conn.close()

    return df

def preprocess_data(df):
    print('3')
    df['combined_features'] = (
        df['genres'].fillna('') + " " +
        df['keywords'].fillna('') + " " +
        df['overview'].fillna('')
    )
    return df

def recommend_movies(movie_title, df, top_n=10):
    print(f"Looking for movie: {movie_title}")
    

    # Check if movie exists in dataset
    if movie_title not in df['title'].values:
        print(f"Error: Movie '{movie_title}' not found in dataset.")
        print("Available movie titles:", df['title'].head(20))  # Show the first few titles
        return []

    movie_index = df[df['title'] == movie_title].index[0]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

    similar_movies = list(enumerate(similarity_matrix[movie_index].toarray()[0]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommendations = [df.iloc[i[0]]['title'] for i in sorted_movies]
    return recommendations

if __name__ == "__main__":
    print('5')
    movie_df = fetch_movie_data()

    if movie_df is not None:
        print('4')
        movie_df = preprocess_data(movie_df)

        # Example: Get recommendations for "Inception"
        movie_name = "Bedevil"  # Change this for different movies
        suggestions = recommend_movies(movie_name, movie_df)

        print(f"Recommended movies for '{movie_name}':", suggestions)
