import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import paho.mqtt.client as mqtt
import json
import threading
import base64
import time
from flask import Flask, request, jsonify  # {{ edit_1 }}

#import mlflow
from edge_infer import perform_inference
from edge_training import train_model
from datasets.chest_xray_processor import process_chest_xray_data
from datasets.mt_processor import process_medical_transcriptions_data
from load_models import load_mobilenet_model

app = Flask(__name__)

# Load configuration
#MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.200.3.30')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC_UPLOAD = os.getenv('MQTT_TOPIC_UPLOAD', 'models/upload')
MQTT_TOPIC_AGGREGATED = os.getenv('MQTT_TOPIC_AGGREGATED', 'models/aggregated')

# Unique identifier for the end device (can be MAC address or any unique ID)
DEVICE_ID = os.getenv('DEVICE_ID', 'device_1')

# Initialize MQTT Client
client = mqtt.Client(client_id=DEVICE_ID)

# Event to signal when a new aggregated model is received
model_update_event = threading.Event()

# Global model variable
model_lock = threading.Lock()
model = None

# Set up MLflow tracking
#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#mlflow.set_experiment("Edge_Responsible_Analytics")

def process_task(task):
    if task['type'] == 'inference':
        if task['data_type'] == 'chest_xray':
            processed_data = process_chest_xray_data(task['data'])
        elif task['data_type'] == 'mt':
            processed_data = process_medical_transcriptions_data(task['data'])
        else:
            return "Unknown data type"
        return perform_inference(processed_data, task['data_type'])
    elif task['type'] == 'training':
        if task['data_type'] == 'chest_xray':
            training_results = train_model(task['data'], task['data_type'])
        elif task['data_type'] == 'mt':
            training_results = train_model(task['data'], task['data_type'])
        else:
            return "Unknown data type"
        return training_results
    else:
        return "Unknown task type"

# Function to send the trained model
def send_trained_model(model_path, model_type='MobileNet'):
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
    payload = json.dumps({
        'device_id': DEVICE_ID,
        'model_type': model_type,
        'model_data': model_b64
    })
    client.publish(MQTT_TOPIC_UPLOAD, payload)
    print(f"[{DEVICE_ID}] Sent trained model to {MQTT_TOPIC_UPLOAD}, model size {len(model_b64)}")

# Callback when a message is received
def on_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_AGGREGATED:
        payload = json.loads(msg.payload.decode('utf-8'))
        aggregated_model_b64 = payload.get('model_data')
        if aggregated_model_b64:
            aggregated_model_bytes = base64.b64decode(aggregated_model_b64)
            aggregated_model_path = 'aggregated_model.h5'
            with open(aggregated_model_path, 'wb') as f:
                f.write(aggregated_model_bytes)
            print(f"[{DEVICE_ID}] Received aggregated model. Loading {aggregated_model_path}")
            try:
                new_model = tf.keras.models.load_model(aggregated_model_path)
                with model_lock:
                    global model
                    model = new_model
                print(f"[{DEVICE_ID}] Aggregated model loaded successfully.")
                model_update_event.set()  # Signal that model has been updated
            except Exception as e:
                print(f"[{DEVICE_ID}] Failed to load aggregated model: {e}")

# Set up MQTT callbacks
client.on_message = on_message

# Connect to MQTT Broker
def connect_mqtt():
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.subscribe(MQTT_TOPIC_AGGREGATED)
    print(f"[{DEVICE_ID}] Subscribed to {MQTT_TOPIC_AGGREGATED}")

# Start MQTT loop in a separate thread
def mqtt_loop():
    client.loop_forever()

# Task processing function
def task_processing(task_type, model_type):
    global model
    # Load the initial model
    with model_lock:
        model = load_mobilenet_model()

    # Define tasks
    inference_task = {
        'type': 'inference',
        'data_type': 'chest_xray',
        'data': 'datasets/chest_xray/val'
    }
    training_task = {
        'type': 'training',
        'data_type': 'chest_xray',
        'data': 'datasets/chest_xray/train'
    }

    if task_type == 'inference':
        # Perform Inference
        print(f"[{DEVICE_ID}] Starting inference task.")
        inference_result = process_task(inference_task)
        print(f"[{DEVICE_ID}] Inference Result: {inference_result}")

    if task_type == 'training':
        # Perform Training
        print(f"[{DEVICE_ID}] Starting training task.")
        training_result = process_task(training_task)
        print(f"[{DEVICE_ID}] Training Result: {training_result}")

    # Save the trained model
    model_path = 'mobilenet_model.h5'
    with model_lock:
        model.save(model_path)
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


# Route to upload model
@app.route('/upload_model', methods=['POST'])
def upload_model():
    data = request.json
    model_path = data.get('model_path')
    model_type = data.get('model_type')
    send_trained_model(model_path, model_type)  # Call existing function
    return jsonify({"status": "Model uploaded successfully"}), 200


# New Route to trigger task processing
@app.route('/start_task', methods=['POST'])
def start_task():
    data = request.json
    task_type = data.get('task_type', 'inference')
    model_type = data.get('model_type', 'MobileNet')

    task_thread = threading.Thread(target=task_processing, args=(task_type, model_type))
    task_thread.start()

    return jsonify({"status": f"{task_type} task started"}), 200


def main():
    # Connect to MQTT
    connect_mqtt()
    # Start MQTT loop in background
    thread = threading.Thread(target=mqtt_loop)
    thread.daemon = True
    thread.start()

    print(f"[{DEVICE_ID}] MQTT loop started.")

    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    flask_thread.daemon = True
    flask_thread.start()

    print(f"[{DEVICE_ID}] Flask server started on port 5000.")

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print(f"[{DEVICE_ID}] Shutting down.")
        client.disconnect()


if __name__ == "__main__":
    main()
    # Removed redundant calls to app.run and task_processing

    