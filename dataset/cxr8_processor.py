import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from dataset import load_dataset

def load_chestxray8_data():
    # Load dataset from Hugging Face
    ds = load_dataset("BahaaEldin0/NIH-Chest-Xray-14-Augmented-70-percent")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        'image': ds['train']['image'],
        'label': ds['train']['label'],
        'age': ds['train']['Patient Age'],
        'gender': ds['train']['Patient Gender'],
        'view_position': ds['train']['View Position'],
        'patient_id': ds['train']['Patient ID']
    })
    
    # Process age into groups for better privacy protection
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 18, 30, 50, 70, 100],
                            labels=['0-18', '19-30', '31-50', '51-70', '70+'])
    
    return df

def preprocess_image(image):
    """
    Preprocess the image data from the dataset.
    """
    # Convert PIL image to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Resize to target size
    img_array = tf.image.resize(img_array, (224, 224))
    # Preprocess for MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def process_data_in_batches(df_subset, batch_size=32):
    """
    Process data in batches to manage memory efficiently.
    """
    total_samples = len(df_subset)
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_df = df_subset.iloc[start_idx:end_idx]
        
        images = []
        labels = []
        sensitive_features = []
        
        for _, row in batch_df.iterrows():
            # Process image
            img_array = preprocess_image(row['image'])
            images.append(img_array)
            
            # Process label (assuming binary classification for simplicity)
            label = 1 if "No Finding" not in row['label'] else 0
            labels.append(label)
            
            # Create sensitive features dictionary
            sensitive = {
                'gender': row['gender'],
                'age_group': row['age_group']
            }
            sensitive_features.append(sensitive)
        
        yield (np.array(images), 
               np.array(labels), 
               pd.DataFrame(sensitive_features))

def prepare_dataset(df, test_size=0.2, batch_size=32):
    """
    Prepare the dataset for training and testing with batch processing.
    """
    # Split the data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    
    return (process_data_in_batches(train_df, batch_size), 
            process_data_in_batches(val_df, batch_size))

def process_cxr8_data(batch_size=32):
    """
    Process CXR8 data with batch processing.
    Returns generators for training and validation data.
    """
    try:
        # Load the dataset
        df = load_chestxray8_data()
        
        # Prepare dataset with batch processing
        train_generator, val_generator = prepare_dataset(df, batch_size=batch_size)
        
        return train_generator, val_generator
        
    except Exception as e:
        print(f"Error processing CXR8 data: {e}")
        raise

if __name__ == "__main__":
    # Test the data loading and processing
    train_gen, val_gen = process_cxr8_data(batch_size=32)
    
    # Test first batch
    X_batch, y_batch, sensitive_batch = next(train_gen)
    
    print(f"Batch shapes:")
    print(f"Images: {X_batch.shape}")
    print(f"Labels: {y_batch.shape}")
    print(f"Sensitive features: {sensitive_batch.shape}")