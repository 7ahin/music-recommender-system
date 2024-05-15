import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from recommeders.knn_recommender import Recommende
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline

# Step 1: Emotion Recognition
# Load the emotion recognition model
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

# Example user input text
user_input = "I feel very happy and energetic today!"

# Perform emotion recognition
emotion_results = emotion_classifier(user_input)

# Extract the emotion with the highest score
top_emotion = max(emotion_results[0], key=lambda x: x['score'])['label']
print(f"Recognized Emotion: {top_emotion}")

# Step 2: Set Up Spotify API
client_id = 'YOUR_SPOTIFY_CLIENT_ID'
client_secret = 'YOUR_SPOTIFY_CLIENT_SECRET'

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Map emotions to Spotify track characteristics or genres
emotion_to_genre = {
    'joy': 'happy',
    'anger': 'metal',
    'sadness': 'sad',
    'fear': 'ambient',
    'disgust': 'punk',
    'surprise': 'pop'
}

# Use the recognized emotion to get the genre
seed_genre = emotion_to_genre.get(top_emotion.lower(), 'pop')  # Default to 'pop' if emotion is not in the map

# Fetch recommendations based on the emotion/genre
recommendations = sp.recommendations(seed_genres=[seed_genre], limit=10)

# Display recommendations
for i, track in enumerate(recommendations['tracks']):
    print(f"{i+1}: {track['name']} by {track['artists'][0]['name']}")

