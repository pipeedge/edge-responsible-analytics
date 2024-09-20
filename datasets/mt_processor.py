import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import kaggle

# os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

def download_medical_transcriptions_data():
    try:
        print("Downloading Medical Transcriptions data...")
        kaggle.api.dataset_download_files('tboyle10/medicaltranscriptions', path='datasets/', unzip=True)
    except kaggle.rest.ApiException as e:
        print(f"Error downloading dataset: {e}")
        if e.status == 403:
            print("Access denied. Please ensure you have accepted the dataset's terms and conditions on Kaggle.")
        raise

def load_medical_transcriptions_data(data_path):
    # Load CSV file from the Medical Transcriptions dataset
    data = pd.read_csv(os.path.join(data_path, 'mtsamples.csv'))
    return data

def preprocess_medical_transcriptions(data):
    # Example preprocessing: Extract relevant columns and clean text
    data = data[['transcription', 'medical_specialty']]
    data = data.dropna()
    data['transcription'] = data['transcription'].str.replace('\n', ' ')
    
    # Define features and target
    X = data['transcription']
    y = data['medical_specialty']
    
    return X, y

def process_medical_transcriptions_data(data_path):
    # Download the dataset if not already downloaded
    if not os.path.exists(data_path):
        download_medical_transcriptions_data()
    
    # Load Medical Transcriptions data
    data = load_medical_transcriptions_data(data_path)
    
    # Preprocess the data
    X, y = preprocess_medical_transcriptions(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    data_path = "datasets/mt"
    X_train, X_test, y_train, y_test = process_medical_transcriptions_data(data_path)
    
    # Save the processed data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    print("Data processing completed and saved.")