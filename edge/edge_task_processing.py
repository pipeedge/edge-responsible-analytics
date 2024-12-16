import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import paho.mqtt.client as mqtt
import json
import threading
import base64
import time
import argparse
import psutil  # Import psutil for memory usage
import gc       # Import gc for garbage collection
import shutil   # Add shutil for directory operations
import tempfile # Add tempfile for temporary file operations
import tarfile  # Add tarfile for tar operations

#import mlflow
import tensorflow as tf
from edge_infer import perform_inference, process_inference_results
from edge_training import train_model
from dataset.chest_xray_processor import process_chest_xray_data
from dataset.mt_processor import process_medical_transcriptions_data
import logging
import pandas as pd
import numpy as np 

# Load configuration
#MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
# MQTT_BROKER = os.getenv('MQTT_BROKER', 'mosquitto-service')
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.200.3.159')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC_UPLOAD = os.getenv('MQTT_TOPIC_UPLOAD', 'models/upload')
MQTT_TOPIC_AGGREGATED = os.getenv('MQTT_TOPIC_AGGREGATED', 'models/aggregated')

# Unique identifier for the end device (can be MAC address or any unique ID)
DEVICE_ID = os.getenv('DEVICE_ID', 'device')

# Initialize MQTT Client
client = mqtt.Client(client_id=DEVICE_ID, protocol=mqtt.MQTTv5)

# Event to signal when a new aggregated model is received
model_update_event = threading.Event()

# Global model variable
model_lock = threading.Lock()
model = None

# Set up MLflow tracking
#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#mlflow.set_experiment("Edge_Responsible_Analytics")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("edge_device.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def process_task(task):
    """
    Process a single task based on its type and data.
    """
    data_type = task['data_type']
    
    if data_type == "chest_xray":
        processed_data = process_chest_xray_data(task['data_path'])
    elif data_type == "cxr8":
        from dataset.cxr8_processor import process_cxr8_data
        train_gen, val_gen = process_cxr8_data(batch_size=task.get('batch_size', 32))
        # For inference, use the validation generator
        if task['type'] == 'inference':
            processed_data = val_gen
        else:
            processed_data = train_gen
    elif data_type == "mt":
        # Process medical transcription data
        X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data(task['data_path'])
        # For inference, use the test set
        if task['data_type'] == 'inference':
            processed_data = X_test
        else:
            processed_data = X_train
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    if task['type'] == 'inference':
        predictions = perform_inference(processed_data, data_type)
        if isinstance(predictions, dict):
            # For CXR8 data that includes sensitive features
            return predictions['predictions']
        return predictions
    elif task['type'] == 'training':
        try:
            # Map data_type to model_type
            model_type = 'MobileNet' if data_type in ['chest_xray', 'cxr8'] else 'tinybert'
            
            # Get training configuration from task
            batch_size = task.get('batch_size', 16)  # Default to 16 for IoT devices
            epochs = 10
            
            # Call updated train_model function
            training_metrics, validation_data = train_model(
                data_path=task['data_path'],
                model_type=model_type,
                batch_size=batch_size,
                epochs=epochs
            )
            
            # Return combined results
            return {
                'status': 'success',
                'metrics': training_metrics,
                'validation_data': validation_data,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    else:
        return "Unknown task type"

# Function to send the trained model
def send_trained_model(model_path, model_type):
    try:
        if model_type == 'MobileNet':
            # Handle single file model
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        else:  # TinyBERT
            # Create a temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                shutil.make_archive(tmp.name[:-7], 'gztar', model_path)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
                model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                os.unlink(tmp.name)  # Clean up temp file
        
        # Send the model
        payload = json.dumps({
            'device_id': DEVICE_ID,
            'model_type': model_type,
            'model_data': model_b64
        })
        client.publish(MQTT_TOPIC_UPLOAD, payload)
        print(f"[{DEVICE_ID}] Sent trained model to {MQTT_TOPIC_UPLOAD}, model size {len(model_b64)}")
    except Exception as e:
        print(f"[{DEVICE_ID}] Failed to send trained model: {e}")

# Callback when a message is received
def on_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_AGGREGATED:
        payload = json.loads(msg.payload.decode('utf-8'))
        aggregated_model_b64 = payload.get('model_data')
        model_type = payload.get('model_type')
        
        if aggregated_model_b64:
            try:
                model_bytes = base64.b64decode(aggregated_model_b64)
                
                if model_type == 'MobileNet':
                    # Handle single file model
                    model_path = 'mobilenet_model.keras'
                    with open(model_path, 'wb') as f:
                        f.write(model_bytes)
                else:  # TinyBERT
                    # Handle directory-based model
                    import tempfile
                    import tarfile
                    
                    model_dir = 'tinybert_model'
                    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                        tmp.write(model_bytes)
                        tmp.flush()
                        
                        # Clear existing model directory if it exists
                        if os.path.exists(model_dir):
                            shutil.rmtree(model_dir)
                        
                        # Extract the new model
                        with tarfile.open(tmp.name, 'r:gz') as tar:
                            tar.extractall(path=os.path.dirname(model_dir))
                        
                        os.unlink(tmp.name)  # Clean up temp file
                
                print(f"[{DEVICE_ID}] Received and saved aggregated model")
                
                # Load the new model
                with model_lock:
                    global model
                    if model_type == 'MobileNet':
                        model = tf.keras.models.load_model(model_path, compile=False)
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    else:  # TinyBERT
                        from load_models import load_bert_model
                        model, _ = load_bert_model()  # Reload with saved weights
                
                print(f"[{DEVICE_ID}] Aggregated model loaded successfully")
                model_update_event.set()
                
            except Exception as e:
                print(f"[{DEVICE_ID}] Failed to process aggregated model: {e}")

# Set up MQTT callbacks
client.on_message = on_message

# Connect to MQTT Broker
def connect_mqtt():
    print(f"Connect to {MQTT_BROKER}, {MQTT_PORT}")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.subscribe(MQTT_TOPIC_AGGREGATED)
    print(f"[{DEVICE_ID}] Subscribed to {MQTT_TOPIC_AGGREGATED}")

# Start MQTT loop in a separate thread
def mqtt_loop():
    client.loop_forever()

# Task processing function
def task_processing(task_type, model_type, data_type):
    global model
    
    # Configure TensorFlow memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Limit GPU memory
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
                )
    except:
        logger.info("No GPU available or couldn't configure GPU memory growth")
    
    # Limit TensorFlow memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU memory growth setting failed: {e}")
    
    # Map model_type to data_type and data paths
    if model_type == 'MobileNet':
        if data_type == 'chest_xray':
            train_data_path = 'dataset/chest_xray/train'
            inference_data_path = 'dataset/chest_xray/val'
        elif data_type == 'cxr8':  # cxr8
            train_data_path = 'dataset/cxr8'
            inference_data_path = 'dataset/cxr8'
    elif model_type == 'tinybert':
        data_type = 'mt'
        train_data_path = 'dataset/mt'
        inference_data_path = 'dataset/mt'
    else:
        logger.error(f"Unsupported model_type: {model_type}")
        return

    # Load the initial model based on model_type
    if model_type == 'MobileNet':
        from load_models import load_mobilenet_model
        with model_lock:
            model = load_mobilenet_model()
    elif model_type == 'tinybert':
        from load_models import load_bert_model
        with model_lock:
            model, tokenizer = load_bert_model()

    # Define tasks with batch processing
    inference_task = {
        'type': 'inference',
        'data_type': data_type,
        'data_path': inference_data_path,
        'batch_size': 16  # Small batch size for memory efficiency
    }
    training_task = {
        'type': 'training',
        'data_type': data_type,
        'data_path': train_data_path,
        'batch_size': 16
    }

    if task_type == 'inference':
        # Perform Inference
        print(f"[{DEVICE_ID}] Starting inference task.")
        inference_result = process_task(inference_task)
        print(f"[{DEVICE_ID}] Inference Result: {np.mean(inference_result)}")

    if task_type == 'training':
        # Perform Training
        print(f"[{DEVICE_ID}] Starting training task.")
        training_result = process_task(training_task)
        print(f"[{DEVICE_ID}] Training Result: {training_result}")

    # Save the trained model
    if model_type == 'MobileNet':
        model_path = 'mobilenet_model.keras'
        with model_lock:
            model.save(model_path, save_format='keras')
    elif model_type == 'tinybert':
        model_path = 'tinybert_model'
        with model_lock:
            if os.path.exists(model_path):
                shutil.rmtree(model_path)  # Clear existing directory
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)  # Save tokenizer along with model
    print(f"[{DEVICE_ID}] Trained model saved to {model_path}")

    # Upload the trained model in a separate thread
    upload_thread = threading.Thread(target=send_trained_model, args=(model_path, model_type))
    upload_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the thread alive
    except KeyboardInterrupt:
        print(f"[{DEVICE_ID}] Shutting down.")
        client.disconnect()

def log_memory_usage():
    """
    Logs the current memory usage of the process in MB.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    logger.info(f"Current memory usage: {mem:.2f} MB")

def clean_up():
    """
    Performs garbage collection to free up memory.
    """
    gc.collect()
    logger.info("Performed garbage collection.")

def memory_monitor(interval=60):
    """
    Thread function to monitor memory usage at specified intervals.
    
    Args:
        interval (int): Time in seconds between each memory check.
    """
    while True:
        log_memory_usage()
        clean_up()
        time.sleep(interval)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edge Task Processing Script")
    parser.add_argument('--model_type', type=str, default='MobileNet',
                        choices=['MobileNet', 'tinybert'],
                        help='Type of model to use (default: MobileNet)')
    parser.add_argument('--task_type', type=str, default='inference',
                        choices=['inference', 'training'],
                        help='Type of task to perform (default: inference)')
    parser.add_argument('--data_type', type=str, default='chest_xray',
                        choices=['chest_xray', 'cxr8', 'mt'],
                        help='Type of data to perform (default: chest_xray)')
    
    args = parser.parse_args()
    model_type = args.model_type
    task_type = args.task_type
    data_type = args.data_type

    logger.info(f"[{DEVICE_ID}] Starting edge task processing with model_type='{model_type}' and task_type='{task_type}'.")

    # Connect to MQTT
    connect_mqtt()
    # Start MQTT loop in background
    thread = threading.Thread(target=mqtt_loop)
    thread.daemon = True
    thread.start()

    # Start memory monitoring thread
    monitor_thread = threading.Thread(target=memory_monitor, args=(300,), daemon=True)
    monitor_thread.start()

    # Start task processing in the main thread
    task_processing(task_type, model_type, data_type)
    try:
        while True:
            time.sleep(1)  # Keep the thread alive
    except KeyboardInterrupt:
        print(f"[{DEVICE_ID}] Shutting down.")
        client.disconnect()

if __name__ == "__main__":
    main()