import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
songs = pd.read_csv('songs.csv')
user_interactions = pd.read_csv('user_interactions.csv')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(songs['lyrics'])

# Compute the cosine similarity matrix using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get song recommendations based on song lyrics
def get_lyrics_recommendations(song_id, num_recommendations=5):
    idx = songs.index[songs['song_id'] == song_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # Exclude the input song itself
    song_indices = [i[0] for i in sim_scores]
    return songs.iloc[song_indices]

# Example usage
recommended_songs = get_lyrics_recommendations(1)
print("Recommended songs based on lyrics for song ID 1:")
print(recommended_songs[['song_id', 'song_title', 'artist']])
