import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from flask import Flask, request, jsonify
from aggregator import ModelAggregator
from monitor import ResponsibleMonitor
# import mlflow
import os
import json
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration
from datasets.mt_processor import process_medical_transcriptions_data
from datasets.chest_xray_processor import process_chest_xray_data
from edge.load_models import load_mobilenet_model, load_t5_model

app = Flask(__name__)
aggregator = ModelAggregator()
monitor = ResponsibleMonitor()

# Load configuration
# MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.12.93.246')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC_UPLOAD = os.getenv('MQTT_TOPIC_UPLOAD', 'models/upload')
MQTT_TOPIC_AGGREGATED = os.getenv('MQTT_TOPIC_AGGREGATED', 'models/aggregated')

# Set up MLflow tracking
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("Edge_Responsible_Analytics")

# Load test data and sensitive features
X_test, y_test, sensitive_features = None, None, None
data_type = os.getenv('DATA_TYPE', 'chest_xray')  # or 'mt'

if data_type == 'chest_xray':
    _, X_test, _, y_test, _, sensitive_features = process_chest_xray_data("datasets/chest_xray")
elif data_type == 'mt':
    _, X_test, _, y_test, _, sensitive_features = process_medical_transcriptions_data("datasets/medical_transcriptions")

monitor.set_test_data(X_test, y_test, sensitive_features)

def on_message(client, userdata, message):
    payload = json.loads(message.payload.decode())
    device_id = payload['device_id']
    model_type = payload['model_type']
    model_data = bytes.fromhex(payload['model_data'])
    
    if model_type == 'MobileNet':
        model_path = f"models/{device_id}_mobilenet_model.h5"
    elif model_type == 'T5':
        model_path = f"models/{device_id}_t5_model"
    else:
        return
    
    with open(model_path, 'wb') as f:
        f.write(model_data)
    
    if aggregator.process_update(device_id, model_type):
        # Send the aggregated model back to the end-devices
        aggregated_model_path = f"models/aggregated_{model_type.lower()}_model.h5" if model_type == 'MobileNet' else f"models/aggregated_{model_type.lower()}_model"
        with open(aggregated_model_path, 'rb') as f:
            aggregated_model_data = f.read()
        
        aggregated_payload = {
            'model_type': model_type,
            'model_data': aggregated_model_data.hex()
        }
        
        client.publish(MQTT_TOPIC_AGGREGATED, json.dumps(aggregated_payload))
        
        # Evaluate the aggregated model against the policy
        if model_type == 'MobileNet':
            model = load_mobilenet_model()
        elif model_type == 'T5':
            model = load_t5_model
        
        fairness, security = monitor.evaluate_responsible_metrics(model)
        policy_result = monitor.evaluate_policy(fairness, security)
        
        if policy_result:
            print("Aggregated model meets the policy requirements.")
        else:
            print("Aggregated model does not meet the policy requirements.")

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(MQTT_TOPIC_UPLOAD)
client.loop_start()

@app.route('/monitor', methods=['GET'])
def monitor_models():
    # Load the latest aggregated model
    model_path = 'models/aggregated_mobilenet_model.h5'  # or 'models/aggregated_t5_model'
    model = tf.keras.models.load_model(model_path)  # or TFT5ForConditionalGeneration.from_pretrained(model_path)
    
    fairness, security = monitor.evaluate_responsible_metrics(model)
    monitor.log_metrics(fairness, security)
    return jsonify({"fairness": fairness, "security": security})

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(host='0.0.0.0', port=8000)