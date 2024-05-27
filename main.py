import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from transformers import pipeline

# Step 1: Load DEAM Features and Annotations

def load_deam_features(folder_path):
    all_features = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            features_df = pd.read_csv(file_path)
            all_features.append(features_df)

    combined_features = pd.concat(all_features, ignore_index=True)
    return combined_features

def load_deam_annotations(folder_path):
    all_annotations = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            annotations_df = pd.read_csv(file_path)
            all_annotations.append(annotations_df)

    combined_annotations = pd.concat(all_annotations, ignore_index=True)
    return combined_annotations

# Directory paths relative to the main script
current_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.join(current_dir, 'deam_features')
annotations_dir = os.path.join(current_dir, 'deam_annotations')

# Load DEAM features and annotations
deam_features = load_deam_features(features_dir)
deam_annotations = load_deam_annotations(annotations_dir)

# Merge features with annotations
deam_data = pd.merge(deam_features, deam_annotations, on='file_id')

# Encode emotions and standardize features
label_encoder = LabelEncoder()
deam_data['emotion_encoded'] = label_encoder.fit_transform(deam_data['emotion'])

feature_columns = ['tempo', 'energy', 'danceability', 'valence']
X = deam_data[feature_columns]
y = deam_data['emotion_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the model and preprocessing objects
model_dir = os.path.join(current_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(classifier, os.path.join(model_dir, 'classifier_model.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

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

# Step 3: Music Recommendations Based on Emotion

def get_music_recommendations(emotion):
    recommendations = {
        'joy': ['Happy_Song_1', 'Happy_Song_2', 'Happy_Song_3'],
        'anger': ['Metal_Song_1', 'Metal_Song_2', 'Metal_Song_3'],
        'sadness': ['Sad_Song_1', 'Sad_Song_2', 'Sad_Song_3'],
        'fear': ['Ambient_Song_1', 'Ambient_Song_2', 'Ambient_Song_3'],
        'disgust': ['Punk_Song_1', 'Punk_Song_2', 'Punk_Song_3'],
        'surprise': ['Pop_Song_1', 'Pop_Song_2', 'Pop_Song_3']
    }
    
    return recommendations.get(emotion.lower(), ['No recommendations found for this emotion'])

# Get music recommendations based on the recognized emotion
recommendations = get_music_recommendations(top_emotion)

# Display recommendations
print("Music Recommendations:")
for i, song in enumerate(recommendations):
    print(f"{i+1}: {song}")


