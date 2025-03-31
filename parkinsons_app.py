import streamlit as st
import librosa
import numpy as np
import pickle
import joblib
import os
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Constants
UPLOAD_PATH = "AudioFiles/uploaded_audio.wav"
UPLOAD_DIR = "AudioFiles"

# Ensure the directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load ML model and scaler
model = joblib.load('models/xgb_clf_new.joblib')
scaler = joblib.load('models/scaler.joblib')

# List of expected features
EXPECTED_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio
        features = {}
        
        # Fundamental frequencies
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        features['MDVP:Fo(Hz)'] = np.mean(pitches) if len(pitches) > 0 else np.nan
        features['MDVP:Fhi(Hz)'] = np.max(pitches) if len(pitches) > 0 else np.nan
        features['MDVP:Flo(Hz)'] = np.min(pitches) if len(pitches) > 0 else np.nan

        # Noise-to-Harmonic Ratio (NHR)
        harmonic, percussive = librosa.effects.hpss(y)
        features['NHR'] = np.mean(percussive) / (np.mean(harmonic) + 1e-6)
        features['HNR'] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        # Nonlinear dynamic features
        features['RPDE'] = entropy(pitches) if len(pitches) > 0 else np.nan
        features['DFA'] = librosa.feature.rms(y=y).mean()  # Approximation for DFA

        # Spread and PPE (variations in pitch)
        features['spread1'] = np.std(pitches) if len(pitches) > 0 else np.nan
        features['spread2'] = np.var(pitches) if len(pitches) > 0 else np.nan
        features['D2'] = np.percentile(pitches, 99) if len(pitches) > 0 else np.nan
        features['PPE'] = np.mean(np.abs(pitches - np.mean(pitches))) if len(pitches) > 0 else np.nan
        
        # Ensure all expected features exist
        for feature in EXPECTED_FEATURES:
            if feature not in features:
                features[feature] = np.nan  # Fill missing features with NaN
        
        # Gender Classification
        mean_pitch = np.mean(pitches) if len(pitches) > 0 else 0
        gender = "Male" if mean_pitch < 165 else "Female"  # Rough threshold for voice classification
        features['Gender'] = gender
        
        return features
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

def save_uploaded_file(uploadedfile):
    try:
        file_path = os.path.join(UPLOAD_DIR, uploadedfile.name)
        with open(file_path, "wb") as f:
            f.write(uploadedfile.getbuffer())
        return file_path
    except Exception as e:
        return None

# Streamlit app layout
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Parkinson's Disease Detection from Voice")
st.markdown("""Analyze voice recordings to detect the likelihood of Parkinson's disease using machine learning.""")

uploaded_file = st.file_uploader("Upload a voice recording (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file:
    audio_path = save_uploaded_file(uploaded_file)
    st.audio(audio_path, format="audio/wav")
    
    # Prediction Section with Feature Display
    if os.path.exists(audio_path) and model and scaler:
        st.markdown("---")
        st.subheader("📊 Prediction Results and Extracted Features")
        
        features = extract_features(audio_path)
        if features is not None:
            try:
                # Separate numerical features and gender
                gender = features.pop('Gender')  # Remove gender from numerical features

                # Convert to DataFrame
                feature_df = pd.DataFrame([features], index=["Values"]).T
                feature_df.columns = ["Feature Value"]

                # Display gender separately
                st.info(f"Detected Gender: **{gender}**")
                
                # Prepare feature data for scaling
                feature_array = np.array([features[f] for f in EXPECTED_FEATURES]).reshape(1, -1)
                scaled_features = scaler.transform(feature_array)
                
                # Predict with model
                prediction = model.predict(scaled_features)[0]
                
                # Display prediction results
                if prediction == 1:
                    st.error("🔴 High likelihood of Parkinson's detected. Please consult a doctor.")
                else:
                    st.success("🟢 Low likelihood of Parkinson's detected. Stay healthy!")
                
                # Display feature data
                st.markdown("### Extracted Feature Values")
                st.dataframe(feature_df.style.format("{:.4f}"))  # Display with 4 decimal places
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.warning("Feature extraction failed.")
    else:
        st.warning("Model or scaler file is missing! Prediction is not available.")

