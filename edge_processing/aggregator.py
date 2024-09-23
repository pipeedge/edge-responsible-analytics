import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import os
import json
import base64
import time
import threading
import sys
from collections import defaultdict
import logging
import requests
from datasets.chest_xray_processor import process_chest_xray_data
import mlflow
import mlflow.tensorflow

import paho.mqtt.client as mqtt
import tensorflow as tf
import pandas as pd

from policy_evaluator import evaluate_fairness  # Import the policy evaluator

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# MLflow Configuration
mlflow.set_tracking_uri("http://10.210.32.158:5002")  # Adjust if MLflow runs on a different host/port
mlflow.set_experiment("Model_Fairness_Evaluation")

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.210.32.158')  # Adjust if necessary
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC_UPLOAD = os.getenv('MQTT_TOPIC_UPLOAD', 'models/upload')
MQTT_TOPIC_AGGREGATED = os.getenv('MQTT_TOPIC_AGGREGATED', 'models/aggregated')

# Number of end devices expected
EXPECTED_DEVICES = int(os.getenv('EXPECTED_DEVICES', 1))  # Set accordingly

# Initialize MQTT Client
client = mqtt.Client(client_id='aggregator')

# Dictionary to store received models
received_models = {}
lock = threading.Lock()

# Load fairness thresholds
with open('./policies/fairness_thresholds.json') as f:
    fairness_thresholds = json.load(f)['fairness']['threshold']

# Initialize previous aggregated model path
PREVIOUS_MODEL_PATH = 'aggregated_model_previous.h5'

def on_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_UPLOAD:
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            device_id = payload.get('device_id')
            model_type = payload.get('model_type')
            model_b64 = payload.get('model_data')

            if device_id and model_type and model_b64:
                with lock:
                    if device_id not in received_models:
                        logger.info(f"Received {model_type} model from {device_id}")
                        received_models[device_id] = {
                            'model_type': model_type,
                            'model_data': model_b64
                        }
                    else:
                        logger.warning(f"Model from {device_id} already received.")

                # After receiving, evaluate fairness
                evaluate_and_aggregate()

            else:
                logger.error("Received message with missing fields.")

        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON payload.")
        except Exception as e:
            logger.exception(f"Unexpected error in on_message: {e}")

def connect_mqtt():
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.subscribe(MQTT_TOPIC_UPLOAD)
    logger.info(f"Subscribed to {MQTT_TOPIC_UPLOAD}")

def mqtt_loop():
    client.loop_forever()

def aggregate_and_publish_models():
    try:
        # Implement your aggregation logic here
        # For demonstration, we'll assume aggregation is averaging weights

        logger.info("Starting model aggregation.")

        # Load all received models
        models = []
        for device_id, model_info in received_models.items():
            model_data = base64.b64decode(model_info['model_data'])
            model_path = f"received_{device_id}.h5"
            with open(model_path, 'wb') as f:
                f.write(model_data)
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            logger.info(f"Loaded model from {device_id}")

        # Aggregate models (example: averaging weights)
        averaged_weights = []
        for weights in zip(*[model.get_weights() for model in models]):
            averaged = [tf.reduce_mean([w.numpy() for w in layer_weights], axis=0) for layer_weights in zip(*weights)]
            averaged_weights.append(averaged)

        # Create a new model and set averaged weights
        aggregated_model = models[0].__class__()
        aggregated_model.build(models[0].input_shape)
        aggregated_model.set_weights(averaged_weights)
        aggregated_model_path = 'aggregated_model.h5'
        aggregated_model.save(aggregated_model_path)
        logger.info(f"Aggregated model saved to {aggregated_model_path}")

        # Log the aggregated model with MLflow
        with mlflow.start_run(run_name="Aggregated_Model_Run"):
            mlflow.log_param("aggregated_from_devices", EXPECTED_DEVICES)
            mlflow.log_artifact(aggregated_model_path)
            logger.info("Aggregated model logged to MLflow.")

        # Backup the previous model
        if os.path.exists(PREVIOUS_MODEL_PATH):
            os.remove(PREVIOUS_MODEL_PATH)
        os.rename(aggregated_model_path, PREVIOUS_MODEL_PATH)

        # Publish the aggregated model to edge devices
        with open(PREVIOUS_MODEL_PATH, 'rb') as f:
            aggregated_model_bytes = f.read()
        aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')

        payload = json.dumps({
            'model_data': aggregated_model_b64,
            'model_type': 'MobileNet'  # Adjust as needed
        })
        client.publish(MQTT_TOPIC_AGGREGATED, payload)
        logger.info(f"Published aggregated model to {MQTT_TOPIC_AGGREGATED}")

    except Exception as e:
        logger.exception(f"Aggregation or Publishing failed: {e}")

def evaluate_and_aggregate():
    with lock:
        if len(received_models) >= EXPECTED_DEVICES:
            logger.info("All models received. Evaluating fairness policies.")

            # Load aggregated model for fairness evaluation
            # Assuming aggregation logic has been applied already
            aggregated_model_path = 'aggregated_model.h5'

            if not os.path.exists(aggregated_model_path):
                logger.error(f"Aggreated model not found at {aggregated_model_path}")
                return

            aggregated_model = tf.keras.models.load_model(aggregated_model_path)

            # Prepare validation data
            # Prepare validation data
            # Replace with your actual validation dataset and sensitive feature(s)
            X_val, y_val, sensitive_features = process_chest_xray_data("datasets/chest_xray/val")

            # Evaluate fairness
            is_fair, failed_policies = evaluate_fairness(
                model=aggregated_model,
                X=X_val,
                y_true=y_val,
                sensitive_features=sensitive_features['race'],       # Example: race as sensitive feature
                thresholds=fairness_thresholds
            )

            if is_fair:
                logger.info("Models passed fairness policies. Proceeding to aggregate and publish.")
                aggregate_and_publish_models()
                received_models.clear()
            else:
                logger.warning(f"Models failed fairness policies: {failed_policies}. Retaining previous model.")
                # Notify (e.g., via logging, email, MQTT)
                notify_policy_failure(failed_policies)
                # Use the previous aggregated model if it exists
                if os.path.exists(PREVIOUS_MODEL_PATH):
                    with open(PREVIOUS_MODEL_PATH, 'rb') as f:
                        aggregated_model_bytes = f.read()
                    aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')
                    payload = json.dumps({
                        'model_data': aggregated_model_b64,
                        'model_type': 'MobileNet'  # Adjust as needed
                    })
                    client.publish(MQTT_TOPIC_AGGREGATED, payload)
                    logger.info(f"Published previous aggregated model to {MQTT_TOPIC_AGGREGATED}")
                else:
                    logger.error("No previous aggregated model available to deploy.")
                received_models.clear()

def notify_policy_failure(failed_policies):
    """
    Handles notifications when a model fails fairness policies.
    """
    try:
        notification = {
            "status": "Policy Failure",
            "failed_policies": failed_policies,
            "message": "Aggregated model does not satisfy fairness policies. Retaining previous model."
        }
        # Example: Publish to a notification topic
        NOTIFICATION_TOPIC = "models/notification"
        client.publish(NOTIFICATION_TOPIC, json.dumps(notification))
        logger.info(f"Published policy failure notification to {NOTIFICATION_TOPIC}")
    except Exception as e:
        logger.exception(f"Failed to send policy failure notification: {e}")

def main():
    connect_mqtt()
    # Start MQTT loop in separate thread
    thread = threading.Thread(target=mqtt_loop)
    thread.daemon = True
    thread.start()
    logger.info("MQTT loop started.")

    logger.info("Aggregation triggered.")
    evaluate_and_aggregate()

if __name__ == "__main__":
    main()