import os
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import json

def extract_tar_gz_files(download_dir, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    for item in os.listdir(download_dir):
        if item.endswith('.tar.gz'):
            file_path = os.path.join(download_dir, item)
            print(f"Extracting {file_path}...")
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            print(f"Extraction of {file_path} complete.")

def read_labels(csv_path):
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    df = df[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender']]
    
    # Process labels
    # Split multiple labels separated by '|'
    df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|') if pd.notnull(x) else [])
    
    # Create a binary label: 0 for No Findings, 1 for any disease
    df['Label'] = df['Finding Labels'].apply(lambda labels: 0 if not labels else 1)
    
    # Extract sensitive features
    # Convert age to integer
    df['Patient Age'] = df['Patient Age'].apply(lambda x: int(x) if pd.notnull(x) else -1)
    # Encode gender: Male=0, Female=1, Unknown=2
    df['Patient Gender'] = df['Patient Gender'].apply(lambda x: 0 if x.lower() == 'male' else (1 if x.lower() == 'female' else 2))
    
    # Select relevant columns
    df_labels = df[['Image Index', 'Label', 'Patient Age', 'Patient Gender']]
    
    return df_labels

def split_data(df, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['Label'], random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def organize_dataset(df, source_dir, destination_dir, label_columns):
    for _, row in df.iterrows():
        image_file = row['Image Index']
        src_path = os.path.join(source_dir, image_file)
        dest_dir = destination_dir  # Using a single directory for binary classification
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_dir)
        else:
            print(f"Warning: {src_path} does not exist.")
    
    # Save labels and sensitive features as JSON
    labels = df['Label'].tolist()
    ages = df['Patient Age'].tolist()
    genders = df['Patient Gender'].tolist()
    image_files = df['Image Index'].tolist()
    
    data = {
        'image_files': image_files,
        'labels': labels,
        'age': ages,
        'gender': genders
    }
    
    # Save to destination_dir
    with open(os.path.join(destination_dir, 'labels_sensitive.json'), 'w') as f:
        json.dump(data, f)
    
    print(f"Organized dataset saved in {destination_dir}")

def prepare_dataset(download_dir, extract_dir, csv_path, organized_dir, test_size=0.2):
    print("Starting extraction of images...")
    extract_tar_gz_files(download_dir, extract_dir)
    print("Extraction complete.\n")
    
    print("Reading labels and sensitive features from CSV...")
    df_labels = read_labels(csv_path)
    print(f"Total samples: {len(df_labels)}\n")
    
    print("Splitting data into training and validation sets...")
    train_df, val_df = split_data(df_labels, test_size=test_size)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}\n")
    
    print("Organizing training dataset...")
    train_dir = os.path.join(organized_dir, 'train')
    organize_dataset(train_df, extract_dir, train_dir, label_columns)
    print("Training dataset organized.\n")
    
    print("Organizing validation dataset...")
    val_dir = os.path.join(organized_dir, 'val')
    organize_dataset(val_df, extract_dir, val_dir, label_columns)
    print("Validation dataset organized.\n")
    
    print("Dataset preparation complete.")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    download_directory = os.path.join(ROOT_DIR, "downloaded_zips")  # Directory where batch_download_zips.py downloads files
    extracted_directory = os.path.join(ROOT_DIR, "extracted_images")
    csv_file_path = os.path.join(ROOT_DIR, "Data_Entry_2017_v2020.csv")
    organized_dataset_dir = os.path.join(ROOT_DIR, "cxr8")
    
    # Define label columns
    label_columns = ['Label']  # Binary classification
    
    prepare_dataset(
        download_dir=download_directory,
        extract_dir=extracted_directory,
        csv_path=csv_file_path,
        organized_dir=organized_dataset_dir,
        test_size=0.2
    )