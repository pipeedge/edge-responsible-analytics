import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import kaggle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(ROOT_DIR, 'kaggle_credentials')

def download_medical_transcriptions_data():
    try:
        print("Downloading Medical Transcriptions data...")
        kaggle.api.dataset_download_files('tboyle10/medicaltranscriptions', path='datasets/mt', unzip=True)
    except kaggle.rest.ApiException as e:
        print(f"Error downloading dataset: {e}")
        if e.status == 403:
            print("Access denied. Please ensure you have accepted the dataset's terms and conditions on Kaggle.")
        raise

def load_medical_transcriptions_data(data_path):
    # Load CSV file from the Medical Transcriptions dataset
    data = pd.read_csv(os.path.join(data_path, 'mtsamples.csv'))
    return data

def extract_demographics(description):
    """
    Extract demographic information from the description text.
    Returns gender and age if found, otherwise returns None.
    """
    import re
    
    # Initialize default values
    gender = None
    age = None
    
    # Convert to lowercase for easier matching
    desc_lower = description.lower()
    
    # Extract gender
    if 'female' in desc_lower or 'woman' in desc_lower:
        gender = 'female'
    elif 'male' in desc_lower or 'man' in desc_lower:
        gender = 'male'
    
    # Extract age using regex
    age_pattern = r'\b(\d{1,3})[\s-]*(year|yr|y)[s\s-]*(old)?\b'
    age_match = re.search(age_pattern, desc_lower)
    if age_match:
        try:
            age = int(age_match.group(1))
            # Filter out unrealistic ages
            if age < 0 or age > 120:
                age = None
        except ValueError:
            age = None
            
    return gender, age

def preprocess_medical_transcriptions(data):
    # Example preprocessing: Extract relevant columns and clean text
    data = data[['description', 'transcription', 'medical_specialty']]
    data = data.dropna()
    
    # Extract demographic information
    demographics = data['description'].apply(extract_demographics)
    data['gender'] = demographics.apply(lambda x: x[0])
    data['age'] = demographics.apply(lambda x: x[1])
    
    # Drop rows where we couldn't extract demographic information
    data = data.dropna(subset=['gender', 'age'])
    
    # Convert age to categorical bins for fairness evaluation
    data['age_group'] = pd.cut(data['age'], 
                              bins=[0, 18, 35, 50, 65, 120],
                              labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    data['transcription'] = data['transcription'].str.replace('\n', ' ')
    
    # Add task-specific prefix for T5
    data['transcription'] = 'summarize: ' + data['transcription']
    
    # Define features and target
    X = data['transcription']
    y = data['medical_specialty']
    sensitive_features = data[['gender', 'age_group']]
    
    return X, y, sensitive_features

def process_medical_transcriptions_data(data_path):
    # Download the dataset if not already downloaded
    if not os.path.exists(data_path):
        download_medical_transcriptions_data()
    
    # Load Medical Transcriptions data
    data = load_medical_transcriptions_data(data_path)
    
    # Preprocess the data
    X, y, sensitive_features = preprocess_medical_transcriptions(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, sf_train, sf_test

if __name__ == "__main__":
    data_path = "datasets/mt"
    X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data(data_path)
    
    # Save the processed data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('sf_train.npy', sf_train)
    np.save('sf_test.npy', sf_test)
    
    print("Data processing completed and saved.")