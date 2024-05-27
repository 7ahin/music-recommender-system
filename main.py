
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline

# Step 1: Load DEAM Features and Annotations

# Load annotations from the DEAM dataset folder
def load_annotations_from_folder(folder_path):
    all_annotations = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            annotations_df = pd.read_csv(file_path)
            all_annotations.append(annotations_df)

    combined_annotations = pd.concat(all_annotations, ignore_index=True)
    return combined_annotations

# Load features from the DEAM dataset folder
def load_features_from_folder(folder_path):
    all_features = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            features_df = pd.read_csv(file_path)
            all_features.append(features_df)

    combined_features = pd.concat(all_features, ignore_index=True)
    return combined_features

# Directory paths relative to the main script
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(current_dir, 'audio_files')
annotations_dir = os.path.join(current_dir, 'annotations')
features_dir = os.path.join(current_dir, 'features')

# Load and preprocess annotations
annotations_df = load_annotations_from_folder(annotations_dir)

# Load features
features_df = load_features_from_folder(features_dir)

# Merge features with annotations
data = pd.merge(features_df, annotations_df, on='file')

# Encode emotions and standardize features
label_encoder = LabelEncoder()
data['emotion_encoded'] = label_encoder.fit_transform(data['emotion'])

feature_columns = ['tempo', 'energy', 'danceability', 'valence']
X = data[feature_columns]
y = data['emotion_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the model and preprocessing objects
joblib.dump(classifier, os.path.join(current_dir, 'models/classifier_model.pkl'))
joblib.dump(label_encoder, os.path.join(current_dir, 'models/label_encoder.pkl'))
joblib.dump(scaler, os.path.join(current_dir, 'models/scaler.pkl'))

# Step 2: Emotion Recognition

# Load the emotion recognition model
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

# Example user input text
user_input = "I feel very happy and energetic today!"

# Perform emotion recognition
emotion_results = emotion_classifier(user_input)

# Extract the emotion with the highest score
top_emotion = max(emotion_results[0], key=lambda x: x['score'])['label']
print(f"Recognized Emotion: {top_emotion}")

# Step 3: Set Up Spotify API

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


