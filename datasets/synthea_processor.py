import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import kaggle

# os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.dirname(__file__), 'datasets')

def download_synthea_data():
    # Download the dataset from Kaggle
    kaggle.api.dataset_download_files('synthea/synthea-dataset', path='datasets/', unzip=True)

def load_synthea_data(data_path):
    # Load JSON files from the Synthea dataset
    data = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.json'):
            with open(os.path.join(data_path, file_name), 'r') as f:
                data.append(json.load(f))
    return data

def preprocess_synthea(data):
    # Convert JSON data to DataFrame
    records = []
    for patient in data:
        for encounter in patient['encounters']:
            record = {
                'age': patient['age'],
                'gender': patient['gender'],
                'encounter_type': encounter['type'],
                'condition': encounter['condition'],
                'medication': encounter['medication'],
                'procedure': encounter['procedure'],
                'mortality': patient['mortality']
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Define features, target, and sensitive attribute
    features = ['age', 'encounter_type', 'condition', 'medication', 'procedure']
    target = 'mortality'
    sensitive_attribute = 'gender'
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Normalize numerical columns
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Prepare X, y, and sensitive_features
    X = df[features]
    y = df[target]
    sensitive_features = df[sensitive_attribute]
    
    return X, y, sensitive_features

def process_synthea_data(data_path):
    # Download the dataset if not already downloaded
    if not os.path.exists(data_path):
        download_synthea_data()
    
    # Load Synthea data
    data = load_synthea_data(data_path)
    
    # Preprocess the data
    X, y, sensitive_features = preprocess_synthea(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test

# Example usage
if __name__ == "__main__":
    data_path = "datasets/synthea"
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = process_synthea_data(data_path)
    
    # Save the processed data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('sensitive_train.npy', sensitive_train)
    np.save('sensitive_test.npy', sensitive_test)
    
    print("Data processing completed and saved.")