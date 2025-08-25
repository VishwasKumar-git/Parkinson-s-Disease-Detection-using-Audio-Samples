🧠 Parkinson’s Disease Prediction using AI/ML
📌 Overview

This project aims to predict Parkinson’s Disease from patient voice/audio samples using Machine Learning techniques.
The system analyzes extracted features from audio recordings and classifies whether the person is likely to have Parkinson’s Disease.

The project also includes a Streamlit-based web app where users can upload their audio sample (or input features) and instantly receive predictions.


🚀 Features
🎤 Audio-based Prediction – Uses ML models to detect Parkinson’s disease.
📊 Data Processing & Feature Extraction – Includes preprocessing and feature engineering.
🤖 Machine Learning Model – Trained on medical/audio datasets.
🌐 Streamlit Frontend – Simple and interactive user interface for predictions.
📈 Performance Metrics – Model evaluated using accuracy, precision, recall, and F1-score.


🛠️ Tech Stack
Frontend: Streamlit
Backend / ML: Python, Scikit-learn / TensorFlow / PyTorch (depending on your implementation)
Libraries: NumPy, Pandas, Matplotlib, Librosa (if audio features), Joblib/Pickle (for saving model)
Dataset: UCI Parkinson’s Dataset


⚙️ Installation & Setup

1. Clone the repository
git clone https://github.com/VishwasKumar-git/Parkinson-s-Disease-Detection-using-Audio-Samples.git
cd Parkinson-s-Disease-Detection-using-Audio-Samples

2. Create a virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate   # for Linux/Mac
    venv\Scripts\activate      # for Windows
   
3. Run the Streamlit app
   streamlit run app.py




