from flask import Flask, request, jsonify
import faiss
import pickle
import numpy as np
import pandas as pd
from config import DB_CONFIG
from trainmodel import fetch_movie_data  # Import fetch_movie_data from trainmodel
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load Pre-trained Models
with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

with open("data/movie_ids.pkl", "rb") as f:
    id_dict = pickle.load(f)

# Load FAISS HNSW Index
faiss_index = faiss.read_index("data/hnsw_index.bin")

# Fetch movie data
movie_df, movie_dict, id_dict = fetch_movie_data()

# Recommendation function
def recommend_movies(movie_ids_list, top_n=10):
    movie_indices = [idx for idx, movie_id in id_dict.items() if movie_id in movie_ids_list]

    if not movie_indices:
        return []

    query_vectors = vectorizer.transform(movie_df.loc[movie_indices, 'combined_features']).toarray()
    reduced_query_vectors = svd.transform(query_vectors).astype(np.float32)
    combined_query_vector = np.mean(reduced_query_vectors, axis=0, keepdims=True)

    distances, indices = faiss_index.search(combined_query_vector, top_n + len(movie_indices))

    recommended_movies = []
    for i, dist in zip(indices[0], distances[0]):
        if id_dict[i] not in movie_ids_list:
            similarity_percent = round((1 - float(dist)) * 100, 2)  # Convert distance to percentage similarity
            recommended_movies.append({"id": id_dict[i], "similarity": similarity_percent})
        if len(recommended_movies) >= top_n:
            break

    return recommended_movies

# API Endpoint for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_ids = data.get("movie_ids", [])

    if not movie_ids:
        return jsonify({"error": "No movie IDs provided"}), 400

    recommendations = recommend_movies(movie_ids, top_n=10)
    return jsonify({"recommended_movies": recommendations})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)