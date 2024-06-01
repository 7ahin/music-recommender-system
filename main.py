import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
song_df = pd.read_csv('songs.csv')

# TF-IDF Vectorization on emotion
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(songs['emotion'])

# Compute the cosine similarity matrix using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get song recommendations based on song lyrics
def get_recommendations(song_name):
    song_index = song_df[song_df['song'] == song_name].index[0]
    similar_songs = list(enumerate(cosine_similarity[song_index]))
    similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = [song_df.iloc[i[0]]['song'] for i in similar_songs]
    return recommendations

# Example usage
recommended_songs = get_lyrics_recommendations(1)
print("Recommended songs based on lyrics for song ID 1:")
print(recommended_songs[['song_id', 'song_title', 'artist']])
