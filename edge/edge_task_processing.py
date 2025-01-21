import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import logging
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("edge_device.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

import tensorflow as tf
def configure_memory_settings():
    """Configure TensorFlow for memory-efficient training on edge devices"""
    try:
        # Limit TensorFlow memory growth for GPUs if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Limit CPU memory usage through thread settings
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        
        # Remove the unsupported CPU memory limit configuration
        # Instead, use soft device placement
        tf.config.set_soft_device_placement(True)
        
        logger.info("Memory settings configured successfully")
    except Exception as e:
        logger.warning(f"Error configuring memory settings: {e}")

configure_memory_settings()

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
from dataset.chest_xray_processor import process_chest_xray_data
from dataset.mt_processor import process_medical_transcriptions_data
import pandas as pd
import numpy as np
from datetime import datetime

# Load configuration
#MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
MQTT_BROKER = os.getenv('MQTT_BROKER', 'mosquitto-service')
# MQTT_BROKER = os.getenv('MQTT_BROKER', '10.200.3.159')
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

# Add connection monitoring callbacks
def on_connect(client, userdata, flags, rc, properties=None):
    """MQTTv5 on_connect callback"""
    logger.info(f"[{DEVICE_ID}] Connected to MQTT broker with result code: {rc}")
    if rc == 0:
        # Successful connection, subscribe to topics
        client.subscribe(MQTT_TOPIC_AGGREGATED)
        logger.info(f"[{DEVICE_ID}] Successfully subscribed to {MQTT_TOPIC_AGGREGATED}")
    else:
        logger.error(f"[{DEVICE_ID}] Failed to connect, return code: {rc}")

def on_disconnect(client, userdata, rc, properties=None):
    """MQTTv5 on_disconnect callback"""
    logger.warning(f"[{DEVICE_ID}] Disconnected with result code: {rc}")
    if rc != 0:
        logger.info(f"[{DEVICE_ID}] Unexpected disconnection. Attempting to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            logger.exception(f"[{DEVICE_ID}] Failed to reconnect")

def on_message(client, userdata, msg):
    """MQTTv5 on_message callback"""
    if msg.topic == MQTT_TOPIC_AGGREGATED:
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            chunk_data = payload.get('model_data')
            model_type = payload.get('model_type')
            chunk_index = payload.get('chunk_index')
            total_chunks = payload.get('total_chunks')

            if not all([chunk_data, model_type, chunk_index is not None, total_chunks]):
                logger.error(f"[{DEVICE_ID}] Received message with missing fields")
                return

            with model_chunks_lock:
                # Initialize chunk storage for this model type if needed
                if model_type not in model_chunks:
                    model_chunks[model_type] = {
                        'chunks': [''] * total_chunks,
                        'total_chunks': total_chunks
                    }

                # Store this chunk
                model_chunks[model_type]['chunks'][chunk_index] = chunk_data
                received_count = sum(1 for chunk in model_chunks[model_type]['chunks'] if chunk)
                logger.info(f"[{DEVICE_ID}] Received chunk {chunk_index + 1}/{total_chunks} for {model_type} model")

                # Check if we have all chunks
                if received_count == total_chunks:
                    logger.info(f"[{DEVICE_ID}] Received all chunks for {model_type} model, assembling")
                    complete_model = ''.join(model_chunks[model_type]['chunks'])
                    # Clean up chunks
                    del model_chunks[model_type]
                    # Process the complete model
                    process_complete_model(model_type, complete_model)

        except json.JSONDecodeError:
            logger.exception(f"[{DEVICE_ID}] Failed to decode JSON payload")
        except Exception as e:
            logger.exception(f"[{DEVICE_ID}] Unexpected error in on_message: {e}")

# Set up MQTT callbacks
client.on_message = on_message
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Connect to MQTT Broker
def connect_mqtt():
    try:
        logger.info(f"[{DEVICE_ID}] Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        # Enable automatic reconnection
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        logger.exception(f"[{DEVICE_ID}] Failed to connect to MQTT broker")

# Start MQTT loop in a separate thread
def mqtt_loop():
    try:
        logger.info(f"[{DEVICE_ID}] Starting MQTT loop")
        client.loop_forever()
    except Exception as e:
        logger.exception(f"[{DEVICE_ID}] MQTT loop encountered an error")

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
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), "inference_results")
        os.makedirs(results_dir, exist_ok=True)
        
        if isinstance(predictions, dict):
            # For CXR8 data that includes sensitive features
            return predictions['predictions']
        
        # Prepare results dictionary
        results = {
            'status': 'success',
            'predictions': np.mean(predictions),
            'model_type': task.get('model_type'),
            'data_type': data_type,
            'timestamp': datetime.now().isoformat(),
            'device_id': DEVICE_ID,
            'task_config': {
                'batch_size': task.get('batch_size', 32)
            }
        }
        
        # Save results to JSON file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            results_dir,
            f"inference_results_{data_type}_{task['model_type']}_{timestamp}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Inference results saved to {results_file}")

        return predictions
            
    elif task['type'] == 'training':
        try:
            training_metrics = None
            
            if data_type in ['chest_xray', 'cxr8']:
                from edge_training import train_mobilenet_edge
                
                history = train_mobilenet_edge(
                    data_path=task['data_path'],
                    epochs=task.get('epochs', 5),
                    samples_per_class=task.get('samples_per_class', 50)
                )
                
                # History now contains all metrics directly
                training_metrics = {
                    'loss': history['loss'],
                    'accuracy': history['accuracy'],
                    'val_loss': history['val_loss'],
                    'val_accuracy': history['val_accuracy'],
                    'best_accuracy': history['best_accuracy'],
                    'best_loss': history['best_loss']
                }
                
            elif data_type == 'mt':
                # Determine if using T5 or TinyBERT based on task configuration
                model_variant = task.get('model_variant', 'tinybert')
                
                if model_variant == 't5':
                    from edge_training import train_t5_edge
                    training_metrics = train_t5_edge(
                        data_path=task['data_path'],
                        epochs=task.get('epochs', 5),
                        max_samples=task.get('max_samples', 200)
                    )
                    
                else:  # tinybert
                    from edge_training import train_bert_edge
                    training_metrics = train_bert_edge(
                        data_path=task['data_path'],
                        epochs=task.get('epochs', 5),
                        max_samples=task.get('max_samples', 300)
                    )
            
            # Save training results to JSON
            results = {
                'status': 'success',
                'metrics': training_metrics,
                'model_type': task['model_type'],
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'device_id': DEVICE_ID,
                'task_config': {
                    'epochs': task.get('epochs', 5),
                    'max_samples': task.get('max_samples'),
                    'samples_per_class': task.get('samples_per_class'),
                    'model_variant': task.get('model_variant', 'tinybert')
                }
            }
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.getcwd(), "training_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results to JSON file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                results_dir, 
                f"training_results_{data_type}_{task['model_type']}_{timestamp}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training results saved to {results_file}")
            
            return results
            
        except Exception as e:
            error_result = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'device_id': DEVICE_ID,
                'data_type': data_type,
                'model_type': task['model_type']
            }
            
            # Save error results
            results_dir = os.path.join(os.getcwd(), "training_results")
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = os.path.join(
                results_dir,
                f"training_error_{data_type}_{task['model_type']}_{timestamp}.json"
            )
            
            with open(error_file, 'w') as f:
                json.dump(error_result, f, indent=2)
                
            logger.error(f"Training error saved to {error_file}")
            return error_result
    else:
        return "Unknown task type"

# Function to send the trained model
def send_trained_model(model_path, model_type, data_tpye):
    try:
        logger.info(f"[{DEVICE_ID}] Starting to send trained model of type {model_type}")
        if model_type == 'MobileNet':
            # Handle single file model
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            logger.info(f"[{DEVICE_ID}] Successfully read MobileNet model, size: {len(model_bytes)} bytes")
        else:  # TinyBERT
            # Create a temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                shutil.make_archive(tmp.name[:-7], 'gztar', model_path)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
                model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                os.unlink(tmp.name)  # Clean up temp file
                logger.info(f"[{DEVICE_ID}] Successfully read TinyBERT model, size: {len(model_bytes)} bytes")
        
        # Split large models into chunks if needed (MQTT typically has a message size limit)
        chunk_size = 100000  # 100KB chunks
        model_chunks = [model_b64[i:i+chunk_size] for i in range(0, len(model_b64), chunk_size)]
        total_chunks = len(model_chunks)
        
        logger.info(f"[{DEVICE_ID}] Splitting model into {total_chunks} chunks for transmission")
        
        for i, chunk in enumerate(model_chunks):
            payload = json.dumps({
                'device_id': DEVICE_ID,
                'model_type': model_type,
                'model_data': chunk,
                'chunk_index': i,
                'total_chunks': total_chunks,
                'data_type': data_tpye
            })
            
            # Publish with QoS 1 to ensure delivery
            result = client.publish(MQTT_TOPIC_UPLOAD, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"[{DEVICE_ID}] Successfully published chunk {i+1}/{total_chunks}")
            else:
                logger.error(f"[{DEVICE_ID}] Failed to publish chunk {i+1}/{total_chunks}: {mqtt.error_string(result.rc)}")
                # Wait and retry once on failure
                time.sleep(1)
                retry_result = client.publish(MQTT_TOPIC_UPLOAD, payload, qos=1)
                if retry_result.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise Exception(f"Failed to publish chunk after retry: {mqtt.error_string(retry_result.rc)}")
            
            # Small delay between chunks to prevent overwhelming the broker
            time.sleep(0.1)
            
        logger.info(f"[{DEVICE_ID}] Completed sending all {total_chunks} chunks of the model")
    except Exception as e:
        logger.exception(f"[{DEVICE_ID}] Failed to send trained model")

# Task processing function
def task_processing(task_type, model_type, data_type):
    global model
    # Map model_type to data paths
    if model_type == 'MobileNet':
        if data_type == 'chest_xray':
            train_data_path = 'dataset/chest_xray/train'
            inference_data_path = 'dataset/chest_xray/val'
        elif data_type == 'cxr8':
            train_data_path = 'dataset/cxr8'
            inference_data_path = 'dataset/cxr8'
    elif model_type in ['tinybert', 't5']:
        data_type = 'mt'
        train_data_path = 'dataset/mt'
        inference_data_path = 'dataset/mt'
    else:
        logger.error(f"Unsupported model_type: {model_type}")
        return

    # Define tasks with optimized batch sizes for edge devices
    inference_task = {
        'type': 'inference',
        'data_type': data_type,
        'model_type': model_type,
        'data_path': inference_data_path,
        'batch_size': 8  # Reduced batch size for edge devices
    }
    
    training_task = {
        'type': 'training',
        'data_type': data_type,
        'data_path': train_data_path,
        'model_type': model_type,
        'epochs': 5,  # Reduced epochs for edge devices
        'model_variant': model_type.lower(),  # For MT tasks to choose between t5/tinybert
        'samples_per_class': 50 if model_type == 'MobileNet' else None,  # For MobileNet
        'max_samples': 100 if model_type == 't5' else 150  # For transformer models
    }

    if task_type == 'inference':
        # Perform Inference
        print(f"[{DEVICE_ID}] Starting inference task.")
        inference_result = process_task(inference_task)
        print(f"[{DEVICE_ID}] Inference Result: {np.mean(inference_result)}")

    if task_type == 'training':
        # Perform Training with edge optimizations
        print(f"[{DEVICE_ID}] Starting edge-optimized training task.")
        training_result = process_task(training_task)
        print(f"[{DEVICE_ID}] Training Result: {training_result}")
        
        # Clean up memory after training
        gc.collect()

    # Model path based on type
    if model_type == 'MobileNet':
        model_path = 'mobilenet_model.keras'
    elif model_type == 't5':
        model_path = 't5_small'
    else:  # tinybert
        model_path = 'tinybert_model'

    # Upload the trained model in a separate thread
    upload_thread = threading.Thread(target=send_trained_model, args=(model_path, model_type, data_type))
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

# Dictionary to store received model chunks
model_chunks = {}
model_chunks_lock = threading.Lock()

def process_complete_model(model_type, model_data):
    """Process a complete model after receiving all chunks"""
    try:
        model_bytes = base64.b64decode(model_data)
        
        if model_type == 'MobileNet':
            # Handle single file model
            model_path = 'mobilenet_model.keras'
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
        else:  # TinyBERT
            # Handle directory-based model
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
        
        logger.info(f"[{DEVICE_ID}] Received and saved aggregated model")
        
        # Load the new model
        with model_lock:
            global model
            if model_type == 'MobileNet':
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:  # TinyBERT
                from load_models import load_bert_model
                model, _ = load_bert_model()  # Reload with saved weights
        
        logger.info(f"[{DEVICE_ID}] Aggregated model loaded successfully")
        model_update_event.set()
        
    except Exception as e:
        logger.error(f"[{DEVICE_ID}] Failed to process aggregated model: {e}")