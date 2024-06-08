import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
song_df = pd.read_csv('/animesongsemotion.csv')

# TF-IDF Vectorization on emotion
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(song_df['Emotions'])

# Compute the cosine similarity matrix using linear_kernel
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get song recommendations based on song emotion
def get_recommendations(emotion):
    # Ensure the emotion is lowercased for comparison
    emotion = emotion.lower()
    
    # Find the indices of songs with the specified emotion
    emotion_indices = song_df[song_df['Emotions'].str.lower() == emotion].index
    if len(emotion_indices) == 0:
        return "No songs found for the specified emotion."
    
    # Use the first index to get recommendations
    emotion_index = emotion_indices[0]
    similar_songs = list(enumerate(cosine_similarity[emotion_index]))
    similar_songs = sorted(similar_songs, key=lambda x: (x[1], song_df.iloc[x[0]]['Ratings']), reverse=True)[1:6]

    recommendations = [(song_df.iloc[i[0]]['Artist'], song_df.iloc[i[0]]['Song Name '], song_df.iloc[i[0]]['Anime']) for i in similar_songs]
    return recommendations

# Example usage
user_input_emotion = input("Enter one of the emotions (happy, sad, intense, pain, love, motivated): ")
user_input_emotion = user_input_emotion.lower()  # Convert to lowercase for case-insensitivity
recommendations = get_recommendations(user_input_emotion)

if isinstance(recommendations, list):
    print(f"Top 5 Rated Songs for '{user_input_emotion}':")
    for i, (artist, song) in enumerate(recommendations, 1):
        print(f"{i}. {song} by {artist} (from {anime})")
else:
    print(recommendations)
