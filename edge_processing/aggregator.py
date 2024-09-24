import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import os
import json
import base64
import time
import threading
import sys
import logging
import requests
from datasets.chest_xray_processor import process_chest_xray_data
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

import paho.mqtt.client as mqtt
import tensorflow as tf
import numpy as np

from policy_evaluator import evaluate_fairness_policy  # Import the policy evaluator

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# MLflow Configuration
mlflow.set_tracking_uri("http://10.200.3.99:5002")  # Adjust if MLflow runs on a different host/port
mlflow.set_experiment("Model_Fairness_Evaluation")
# Initialize MLflow client
mlflow_client = MlflowClient()

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.200.3.99')  # Adjust if necessary
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
thresholds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fairness_thresholds.json')
with open(thresholds_path) as f:
    fairness_thresholds = json.load(f)['fairness']['threshold']

# Initialize previous aggregated model path
PREVIOUS_MODEL_PATH = 'aggregated_model_previous.keras'
aggregated_model_path = 'aggregated_model.keras'

def on_message(client, userdata, msg):
    logger.info(f"[Aggregator] Received message on topic: {msg.topic}")
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

def evaluate_and_aggregate():
    with lock:
        if len(received_models) >= EXPECTED_DEVICES:
            logger.info("All models received. Evaluating fairness policies.")

            # Aggregate models
            try:
                aggregate_models(received_models, aggregated_model_path)
                logger.info(f"Aggregated model saved to {aggregated_model_path}")
            except Exception as e:
                logger.exception(f"Failed to aggregate models: {e}")
                return

            # Load the aggregated model for fairness evaluation
            if not os.path.exists(aggregated_model_path):
                logger.error(f"Aggregated model not found at {aggregated_model_path}")
                return

            aggregated_model = tf.keras.models.load_model(aggregated_model_path, compile=False)
            aggregated_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Prepare validation data
            X_val, y_val, sensitive_features = [], [], []

            # Iterate over the generator to collect all batches
            for batch_X, batch_y, batch_sensitive in process_chest_xray_data("datasets/chest_xray/val", batch_size=32):
                X_val.append(batch_X)
                y_val.append(batch_y)
                sensitive_features.append(batch_sensitive)

            # Concatenate all batches into single arrays
            X_val = np.concatenate(X_val, axis=0)
            y_val = np.concatenate(y_val, axis=0)
            sensitive_features = np.concatenate(sensitive_features, axis=0)
            print(X_val.shape, y_val.shape, sensitive_features.shape)
            
            num_samples = len(sensitive_features)
            group_size = num_samples // 3  # Adjust to create 3 groups

            # Create synthetic groups
            sensitive_features[:group_size] = 0
            sensitive_features[group_size:2*group_size] = 1
            sensitive_features[2*group_size:] = 2

            # Log the unique values in sensitive_features
            unique_values = np.unique(sensitive_features)
            logger.info(f"Unique sensitive feature values: {unique_values}")

            # Start MLflow run
            with mlflow.start_run(run_name="AggregatedModel_Evaluation"):
                try:
                    # Evaluate fairness using the policy evaluator
                    is_fair, failed_policies = evaluate_fairness_policy(
                        model=aggregated_model,
                        X=X_val,
                        y_true=y_val,
                        sensitive_features=sensitive_features,
                        thresholds=fairness_thresholds
                    )

                    # Log parameters and metrics
                    mlflow.log_param("threshold_demographic_parity_difference", fairness_thresholds["demographic_parity_difference"])
                    mlflow.log_metric("is_fair", int(is_fair))

                    if is_fair:
                        logger.info("Aggregated model passed fairness policies. Publishing the model.")
                        publish_aggregated_model(aggregated_model_path)
                        
                        # Log the aggregated Model 
                        mlflow.keras.log_model(aggregated_model, "model", signature=infer_signature(X_val, aggregated_model.predict(X_val)))
                        # Register the model in MLflow
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                        model_details = mlflow.register_model(model_uri, "AggregatedModel")

                        # Transition to Production stage
                        mlflow_client.transition_model_version_stage(
                            name="AggregatedModel",
                            version=model_details.version,
                            stage="Production"
                        )

                        logger.info(f"Registered model version {model_details.version} as Production.")
                        
                        # Backup the current aggregated model
                        if os.path.exists(PREVIOUS_MODEL_PATH):
                            os.remove(PREVIOUS_MODEL_PATH)
                        os.rename(aggregated_model_path, PREVIOUS_MODEL_PATH)
                        received_models.clear()
                    else:
                        logger.warning(f"Aggregated model failed fairness policies: {failed_policies}. Retaining previous model.")
                        notify_policy_failure(failed_policies)
                        mlflow.log_param("failed_policies", failed_policies)
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

                except Exception as e:
                    logger.exception(f"Error during MLflow logging and model registration: {e}")
                    mlflow.end_run(status="FAILED")
                    return

def aggregate_models(received_models, save_path):
    """
    Aggregate models by averaging their weights.
    """
    models = []
    for device_id, model_info in received_models.items():
        model_data = base64.b64decode(model_info['model_data'])
        temp_model_path = f"temp_{device_id}.keras"
        with open(temp_model_path, 'wb') as f:
            f.write(model_data)
        model = tf.keras.models.load_model(temp_model_path, compile=False)
        models.append(model)
        os.remove(temp_model_path)

    if not models:
        raise ValueError("No models to aggregate.")

    # Initialize the aggregated model with the first model's weights
    aggregated_model = models[0]
    for model in models[1:]:
        for agg_layer, layer in zip(aggregated_model.layers, model.layers):
            agg_layer.set_weights([
                (agg_w + layer_w) / 2
                for agg_w, layer_w in zip(agg_layer.get_weights(), layer.get_weights())
            ])

    # Save the aggregated model in .keras format
    aggregated_model.save(save_path, save_format='keras')

def publish_aggregated_model(model_path):
    """
    Publish the aggregated model to the MQTT broker.
    """
    with open(model_path, 'rb') as f:
        aggregated_model_bytes = f.read()
    aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')

    payload = json.dumps({
        'model_data': aggregated_model_b64,
        'model_type': 'MobileNet'  # Adjust as needed
    })
    client.publish(MQTT_TOPIC_AGGREGATED, payload)
    logger.info(f"Published aggregated model to {MQTT_TOPIC_AGGREGATED}")

def notify_policy_failure(failed_policies):
    """
    Notify stakeholders about the failed fairness policies.
    """
    logger.warning(f"Fairness policies failed: {failed_policies}")
    # Implement additional notification mechanisms as needed (e.g., email, alerts)

def connect_mqtt():
    """
    Connect to the MQTT broker and subscribe to relevant topics.
    """
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        logger.exception(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
    client.subscribe(MQTT_TOPIC_UPLOAD)
    client.loop_start()
    logger.info(f"Subscribed to {MQTT_TOPIC_UPLOAD}")

def main():
    connect_mqtt()
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logger.info("Shutting down aggregator.")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    # Define expected number of devices
    EXPECTED_DEVICES = 1  # Adjust as needed
    main()