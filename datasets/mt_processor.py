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

def preprocess_medical_transcriptions(data, batch_size=32):
    """
    Preprocess medical transcriptions data in batches to manage memory usage.
    """
    # Extract relevant columns and create a copy to avoid warnings
    data = data[['description', 'transcription', 'medical_specialty']].copy()
    data = data.dropna()
    
    processed_batches = []
    sensitive_batches = []
    specialty_batches = []
    
    # Process data in batches
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        # Create an explicit copy of the batch
        batch = data.iloc[start_idx:end_idx].copy()
        
        # Extract demographic information for current batch
        demographics = batch['description'].apply(extract_demographics)
        # Use loc for proper assignment
        batch.loc[:, 'gender'] = demographics.apply(lambda x: x[0])
        batch.loc[:, 'age'] = demographics.apply(lambda x: x[1])
        
        # Drop rows where we couldn't extract demographic information
        batch = batch.dropna(subset=['gender', 'age'])
        
        if len(batch) == 0:
            continue
            
        # Convert age to categorical bins for fairness evaluation
        batch.loc[:, 'age_group'] = pd.cut(
            batch['age'], 
            bins=[0, 18, 35, 50, 65, 120],
            labels=['0-18', '19-35', '36-50', '51-65', '65+']
        )
        
        # Clean transcription text
        batch.loc[:, 'transcription'] = batch['transcription'].str.replace('\n', ' ')
        batch.loc[:, 'transcription'] = 'summarize: ' + batch['transcription']
        
        # Store processed batch
        processed_batches.append(batch['transcription'])
        specialty_batches.append(batch['medical_specialty'])
        sensitive_batches.append(batch[['gender', 'age_group']])
        
    # Concatenate all processed batches
    X = pd.concat(processed_batches)
    y = pd.concat(specialty_batches)
    sensitive_features = pd.concat(sensitive_batches)
    
    return X, y, sensitive_features

def process_medical_transcriptions_data(data_path, batch_size=32):
    """
    Process medical transcriptions data with batch processing.
    """
    # Download the dataset if not already downloaded
    if not os.path.exists(data_path):
        download_medical_transcriptions_data()
    
    # Load Medical Transcriptions data
    data = load_medical_transcriptions_data(data_path)
    
    # Preprocess the data in batches
    X, y, sensitive_features = preprocess_medical_transcriptions(data, batch_size=batch_size)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42)
    
    # Convert to smaller chunks for saving
    def save_in_chunks(array, filename, chunk_size=1000):
        chunks = [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]
        for i, chunk in enumerate(chunks):
            np.save(f'{filename}_chunk_{i}.npy', chunk)
        return len(chunks)
    
    # Save processed data in chunks
    train_chunks = save_in_chunks(X_train, 'X_train')
    test_chunks = save_in_chunks(X_test, 'X_test')
    
    # Save smaller arrays normally
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('sf_train.npy', sf_train)
    np.save('sf_test.npy', sf_test)
    
    # Save chunk information
    chunk_info = {
        'train_chunks': train_chunks,
        'test_chunks': test_chunks,
        'chunk_size': 1000
    }
    with open('chunk_info.json', 'w') as f:
        json.dump(chunk_info, f)
    
    return X_train, X_test, y_train, y_test, sf_train, sf_test

def load_data_chunks(prefix, chunk_info):
    """
    Load data chunks and concatenate them.
    """
    chunks = []
    for i in range(chunk_info[f'{prefix}_chunks']):
        chunk = np.load(f'{prefix}_chunk_{i}.npy')
        chunks.append(chunk)
    return np.concatenate(chunks)

if __name__ == "__main__":
    data_path = "datasets/mt"
    # Process data with smaller batch size
    X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data(
        data_path, 
        batch_size=50  # Reduced batch size for memory efficiency
    )
    
    print("Data processing completed and saved in chunks.")