import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

def load_chestxray8_data(csv_path, images_dir):
    # Load the CSV file containing image labels and metadata
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    df = df[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender']]
    
    # Process labels
    df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|') if pd.notnull(x) else [])
    df['Label'] = df['Finding Labels'].apply(lambda labels: 0 if 'No Finding' in labels else 1)
    
    # Encode gender: Male=0, Female=1, Unknown=2
    df['Patient Gender'] = df['Patient Gender'].apply(lambda x: 0 if x.lower() == 'male' else (1 if x.lower() == 'female' else 2))
    
    return df

def preprocess_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def prepare_dataset(df, images_dir, test_size=0.2):
    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['Label'], random_state=42)
    
    def process_data(df):
        images = []
        labels = []
        sensitive_features = []
        
        for _, row in df.iterrows():
            image_file = row['Image Index']
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                images.append(preprocess_image(image_path))
                labels.append(row['Label'])
                sensitive_features.append(row['Patient Gender'])
        
        return np.array(images), np.array(labels), np.array(sensitive_features)
    
    X_train, y_train, sensitive_train = process_data(train_df)
    X_val, y_val, sensitive_val = process_data(val_df)
    
    return (X_train, y_train, sensitive_train), (X_val, y_val, sensitive_val)

# Example usage
if __name__ == "__main__":
    csv_path = "path/to/Data_Entry_2017.csv"
    images_dir = "path/to/images"
    
    df = load_chestxray8_data(csv_path, images_dir)
    (X_train, y_train, sensitive_train), (X_val, y_val, sensitive_val) = prepare_dataset(df, images_dir)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")