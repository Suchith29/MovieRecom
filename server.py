from flask import Flask, request, jsonify
import faiss
import pickle
import numpy as np
import pandas as pd
from config import DB_CONFIG
from trainmodel import fetch_movie_data  # Import fetch_movie_data from trainmodel

# Initialize Flask app
app = Flask(__name__)

# Load Pre-trained Models
with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

with open("data/movie_titles.pkl", "rb") as f:
    movie_titles = pickle.load(f)

# Load FAISS HNSW Index
faiss_index = faiss.read_index("data/hnsw_index.bin")

# Fetch movie data
movie_df, movie_dict = fetch_movie_data()

# Recommendation function
def recommend_movies(movie_titles_list, top_n=10):
    input_movies_lower = set(title.lower() for title in movie_titles_list)
    movie_indices = [movie_dict[title] for title in input_movies_lower if title in movie_dict]

    if not movie_indices:
        return []

    query_vectors = vectorizer.transform(movie_df.loc[movie_indices, 'combined_features']).toarray()
    reduced_query_vectors = svd.transform(query_vectors).astype(np.float32)
    combined_query_vector = np.mean(reduced_query_vectors, axis=0, keepdims=True)

    distances, indices = faiss_index.search(combined_query_vector, top_n + len(input_movies_lower))

    recommended_movies = []
    for i, dist in zip(indices[0], distances[0]):
        if movie_titles[i].lower() not in input_movies_lower:
            similarity_percent = round((1 - dist) * 100, 2)  # Convert distance to percentage similarity
            recommended_movies.append({"title": movie_titles[i], "similarity": similarity_percent})
        if len(recommended_movies) >= top_n:
            break

    return recommended_movies

# API Endpoint for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_list = data.get("movies", [])

    if not movie_list:
        return jsonify({"error": "No movies provided"}), 400

    recommendations = recommend_movies(movie_list, top_n=10)
    return jsonify({"recommended_movies": recommendations})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
