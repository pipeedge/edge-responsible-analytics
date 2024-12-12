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
import pandas as pd

from policy_evaluator import * 

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# MLflow Configuration
mlflow.set_tracking_uri("http://10.200.3.159:5002")  # Adjust if MLflow runs on a different host/port
mlflow.set_experiment("Model_Evaluation")
# Initialize MLflow client
mlflow_client = MlflowClient()

# MQTT Configuration
# MQTT_BROKER = os.getenv('MQTT_BROKER', 'mosquitto-service')
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.200.3.159')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC_UPLOAD = os.getenv('MQTT_TOPIC_UPLOAD', 'models/upload')
MQTT_TOPIC_AGGREGATED = os.getenv('MQTT_TOPIC_AGGREGATED', 'models/aggregated')

# Number of end devices expected
EXPECTED_DEVICES = int(os.getenv('EXPECTED_DEVICES', 1))  # Set accordingly

# Initialize MQTT Client
client = mqtt.Client(client_id='aggregator', protocol=mqtt.MQTTv5)

# Dictionary to store received models
received_models = {}
lock = threading.Lock()

# Load thresholds
thresholds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../opa/policies/')
with open(os.path.join(thresholds_path,'fairness_thresholds.json')) as f:
    fairness_thresholds = json.load(f)['fairness']['threshold']
with open(os.path.join(thresholds_path,'explainability_thresholds.json')) as f:
    explainability_thresholds = json.load(f)['explainability']['threshold']
with open(os.path.join(thresholds_path,'reliability_thresholds.json')) as f:
    reliability_thresholds = json.load(f)['reliability']['threshold']
with open(os.path.join(thresholds_path,'privacy_thresholds.json')) as f:
    privacy_thresholds = json.load(f)['privacy']['threshold']


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
        # Group received models by model_type
        model_types = set()
        for model_info in received_models.values():
            model_types.add(model_info['model_type'])
        
        for model_type in model_types:
            models_of_type = {device_id:info for device_id, info in received_models.items() if info['model_type'] == model_type}
            if len(models_of_type) >= EXPECTED_DEVICES:
                logger.info(f"All models of type '{model_type}' received. Evaluating policies.")

                # Aggregate models
                try:
                    aggregate_models(models_of_type, model_type, aggregated_model_path)
                    logger.info(f"Aggregated {model_type} model saved to {aggregated_model_path}")
                except Exception as e:
                    logger.exception(f"Failed to aggregate {model_type} models: {e}")
                    continue

                # Load the aggregated model for evaluation
                if model_type == 'MobileNet':
                    aggregated_model = tf.keras.models.load_model(aggregated_model_path, compile=False)
                    aggregated_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                elif model_type == 't5_small':
                    from transformers import TFT5ForConditionalGeneration, T5Tokenizer
                    aggregated_model = TFT5ForConditionalGeneration.from_pretrained(aggregated_model_path)
                else:
                    logger.error(f"Unsupported model_type: {model_type}")
                    continue

                # Prepare validation data
                if model_type == 'MobileNet':
                    X_val, y_val, sensitive_features = [], [], []
                    for batch_X, batch_y, batch_sensitive in process_chest_xray_data("datasets/chest_xray/val", batch_size=32):
                        X_val.append(batch_X)
                        y_val.append(batch_y)
                        sensitive_features.append(batch_sensitive)
                    # Concatenate all batches into single arrays
                    X_val = np.concatenate(X_val, axis=0)
                    y_val = np.concatenate(y_val, axis=0)
                    sensitive_features = np.concatenate(sensitive_features, axis=0)
                elif model_type == 't5_small':
                    # Assuming similar processing for 'mt' dataset
                    from datasets.mt_processor import process_medical_transcriptions_data
                    X_train, X_test, y_train, y_test = process_medical_transcriptions_data("datasets/mt/val")
                    X_val = X_test
                    y_val = y_test
                    # Assume no sensitive_features or adapt as needed
                    sensitive_features = np.zeros(len(X_val))  # Placeholder

                # Log data details
                logger.info(f"Model Type: {model_type}")
                logger.info(f"Number of samples - X_val: {len(X_val)}, y_val: {len(y_val)}, sensitive_features: {len(sensitive_features)}")

                # Start MLflow run
                with mlflow.start_run(run_name=f"AggregatedModel_Evaluation_{model_type}"):
                    try:
                        # Convert to DataFrame for privacy evaluation
                        if model_type == 'MobileNet':
                            df_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))  # Adjust reshape as necessary
                            df_val['gender'] = sensitive_features
                        elif model_type == 't5_small':
                            df_val = pd.DataFrame({'transcription': X_val})
                            # Assuming 'transcription' has been prefix with 'summarize: '
                            
                        # Define quasi-identifiers
                        QUASI_IDENTIFIERS = ['gender'] if model_type == 'MobileNet' else []
                        if QUASI_IDENTIFIERS:
                            missing_columns = [col for col in QUASI_IDENTIFIERS if col not in df_val.columns]
                            if missing_columns:
                                logger.error(f"Missing quasi-identifier columns in df_val: {missing_columns}")
                                continue
                        # Perform privacy evaluation
                        if QUASI_IDENTIFIERS:
                            is_private, failed_privacy_policies = evaluate_privacy_policy(df=df_val, 
                                                                                           quasi_identifiers=QUASI_IDENTIFIERS, 
                                                                                           k_threshold=privacy_thresholds["k"])
                        else:
                            is_private, failed_privacy_policies = True, []
    
                        if not is_private:
                            logger.warning(f"Privacy policies failed: {failed_privacy_policies}. Aborting aggregation for {model_type}.")
                            notify_policy_failure(failed_privacy_policies)
                            continue
    
                        # Evaluate fairness using the policy evaluator
                        if model_type == 'MobileNet':
                            is_fair, failed_fairness_policies = evaluate_fairness_policy(
                                model=aggregated_model,
                                X=X_val,
                                y_true=y_val,
                                sensitive_features=sensitive_features,
                                thresholds=fairness_thresholds
                            )
                        else:
                            # For t5_small, fairness evaluation might differ
                            is_fair, failed_fairness_policies = True, []
    
                        # Evaluate explainability
                        if model_type == 'MobileNet':
                            is_explainable, failed_explainability_policies = evaluate_explainability_policy(aggregated_model, X_val, explainability_thresholds)
                        elif model_type == 't5_small':
                            # Define explainability for seq2seq model
                            is_explainable, failed_explainability_policies = evaluate_explainability_policy_t5(aggregated_model, X_val, explainability_thresholds)
                        else:
                            is_explainable, failed_explainability_policies = False, ["unsupported_model_type"]
    
                        # Evaluate reliability
                        if model_type == 'MobileNet':
                            is_reliable, failed_reliability_policies = evaluate_reliability_policy(aggregated_model, X_val, y_val, reliability_thresholds)
                        elif model_type == 't5_small':
                            # Define reliability for seq2seq model
                            is_reliable, failed_reliability_policies = evaluate_reliability_policy_t5(aggregated_model, X_val, y_val, reliability_thresholds)
                        else:
                            is_reliable, failed_reliability_policies = False, ["unsupported_model_type"]
    
                        mlflow.log_param("threshold_demographic_parity_difference", fairness_thresholds.get("demographic_parity_difference", None))
                        if model_type == 'MobileNet':
                            mlflow.log_metric("is_fair", int(is_fair))
                            mlflow.log_metric("is_explainable", int(is_explainable))
                            mlflow.log_metric("is_reliable", int(is_reliable))
                        elif model_type == 't5_small':
                            mlflow.log_metric("is_fair", int(is_fair))
                            mlflow.log_metric("is_explainable", int(is_explainable))
                            mlflow.log_metric("is_reliable", int(is_reliable))
    
                        if is_fair and is_explainable and is_reliable:
                            logger.info(f"Aggregated {model_type} model passed all policies. Publishing the model.")
                            publish_aggregated_model(model_type, aggregated_model_path)
                            
                            # Log the aggregated Model 
                            if model_type == 'MobileNet':
                                mlflow.keras.log_model(
                                    aggregated_model, 
                                    "model", 
                                    signature=infer_signature(X_val, aggregated_model.predict(X_val))
                                )
                            elif model_type == 't5_small':
                                mlflow.transformers.log_model(
                                    aggregated_model, 
                                    "model", 
                                    signature=infer_signature(X_val, aggregated_model.generate(tokenizer(X_val.tolist(), return_tensors="tf", padding=True, truncation=True).input_ids))
                                )
    
                            # Register the model in MLflow
                            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                            model_name = "aggregated_" + model_type
                            model_details = mlflow.register_model(model_uri, model_name)
    
                            # Transition to Production stage
                            mlflow_client.transition_model_version_stage(
                                name=model_name,
                                version=model_details.version,
                                stage="Production"
                            )
    
                            logger.info(f"Registered model version {model_details.version} as Production.")
                            
                            # Backup the current aggregated model
                            if os.path.exists(PREVIOUS_MODEL_PATH):
                                os.remove(PREVIOUS_MODEL_PATH)
                            os.rename(aggregated_model_path, PREVIOUS_MODEL_PATH)
                            # Remove received models of this type
                            for device_id in list(received_models.keys()):
                                if received_models[device_id]['model_type'] == model_type:
                                    del received_models[device_id]
                        else:
                            failed_policies = failed_fairness_policies + failed_explainability_policies + failed_reliability_policies
                            logger.warning(f"Aggregated {model_type} model failed policies: {failed_policies}. Retaining previous model.")
                            notify_policy_failure(failed_policies)
                            mlflow.log_param("failed_policies", failed_policies)
                            # Use the previous aggregated model if it exists
                            if os.path.exists(PREVIOUS_MODEL_PATH):
                                with open(PREVIOUS_MODEL_PATH, 'rb') as f:
                                    aggregated_model_bytes = f.read()
                                aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')
                                payload = json.dumps({
                                    'model_data': aggregated_model_b64,
                                    'model_type': model_type  # Adjust as needed
                                })
                                client.publish(MQTT_TOPIC_AGGREGATED, payload)
                                logger.info(f"Published previous aggregated {model_type} model to {MQTT_TOPIC_AGGREGATED}")
                            else:
                                logger.error("No previous aggregated model available to deploy.")
                            # Remove received models of this type
                            for device_id in list(received_models.keys()):
                                if received_models[device_id]['model_type'] == model_type:
                                    del received_models[device_id]

                    except Exception as e:
                        logger.exception(f"Error during MLflow logging and model registration: {e}")
                        mlflow.end_run(status="FAILED")
                        continue

def aggregate_models(models_of_type, model_type, save_path):
    """
    Aggregate models by averaging their weights based on model_type.
    """
    import tempfile
    from transformers import TFT5ForConditionalGeneration

    models = []
    for device_id, model_info in models_of_type.items():
        model_data = base64.b64decode(model_info['model_data'])
        temp_model_path = f"temp_{device_id}.keras"
        if model_type == 'MobileNet':
            with open(temp_model_path, 'wb') as f:
                f.write(model_data)
            model = tf.keras.models.load_model(temp_model_path, compile=False)
        elif model_type == 't5_small':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
                tmp_file.write(model_data)
                temp_model_path = tmp_file.name
            model = TFT5ForConditionalGeneration.from_pretrained(temp_model_path)
            os.remove(temp_model_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        models.append(model)
        if model_type == 'MobileNet':
            os.remove(temp_model_path)

    if not models:
        raise ValueError("No models to aggregate.")

    if model_type == 'MobileNet':
        # Initialize the aggregated model with the first model's weights
        aggregated_model = models[0]
        for model in models[1:]:
            for agg_layer, layer in zip(aggregated_model.layers, model.layers):
                agg_layer_weights = agg_layer.get_weights()
                layer_weights = layer.get_weights()
                if len(agg_layer_weights) == len(layer_weights):
                    new_weights = [(agg_w + layer_w) / 2 for agg_w, layer_w in zip(agg_layer_weights, layer_weights)]
                    agg_layer.set_weights(new_weights)
                else:
                    logger.warning(f"Layer weight mismatch: {agg_layer.name}")
        # Save the aggregated model in .keras format
        aggregated_model.save(save_path, save_format='keras')

    elif model_type == 't5_small':
        # Average weights for T5 model layers
        # Initialize the aggregated model with the first model
        aggregated_model = models[0]
        for model in models[1:]:
            for param_agg, param in zip(aggregated_model.parameters(), model.parameters()):
                param_agg.data = (param_agg.data + param.data) / 2
        # Save the aggregated model using save_pretrained
        aggregated_model.save_pretrained(save_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def publish_aggregated_model(model_type, model_path):
    """
    Publish the aggregated model to the MQTT broker.
    """
    with open(model_path, 'rb') as f:
        aggregated_model_bytes = f.read()
    aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')

    payload = json.dumps({
        'model_data': aggregated_model_b64,
        'model_type': model_type  # Adjust as needed
    })
    client.publish(MQTT_TOPIC_AGGREGATED, payload)
    logger.info(f"Published aggregated {model_type} model to {MQTT_TOPIC_AGGREGATED}")

def notify_policy_failure(failed_policies):
    """
    Notify stakeholders about the failed policies.
    """
    logger.warning(f"Policies failed: {failed_policies}")
    # Implement additional notification mechanisms as needed (e.g., email, alerts)

def connect_mqtt():
    """
    Connect to the MQTT broker and subscribe to relevant topics.
    """
    client.on_message = on_message
    try:
        print(f"Connect to {MQTT_BROKER}, {MQTT_PORT}")
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        logger.exception(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
    client.subscribe(MQTT_TOPIC_UPLOAD)
    client.loop_start()
    logger.info(f"Subscribed to {MQTT_TOPIC_UPLOAD}")

def send_to_opa(input_data, policy_type):
    """
    Sends evaluation data to OPA for policy decision.
    
    Args:
        input_data (dict): Data containing metrics and thresholds.
        policy_type (str): Type of policy (e.g., 'fairness', 'reliability', 'explainability').
    
    Returns:
        bool: Whether the policy is allowed.
        list: List of failed policies.
    """
    failed_policies = []
    try:
        policy_url = OPA_SERVER_URL + POLICIES.get(policy_type)
        if not policy_url:
            logger.error(f"No policy URL found for policy type: {policy_type}")
            return False, [f"{policy_type}_policy_not_found"]

        response = requests.post(policy_url, json={"input": input_data})
        if response.status_code == 200:
            result = response.json()
            allowed = result.get('result', False)
            if not allowed:
                failed_policies.append(policy_type)
            return allowed, failed_policies
        else:
            logger.error(f"OPA request failed with status code {response.status_code}: {response.text}")
            return False, [f"{policy_type}_opa_request_failed"]
    except Exception as e:
        logger.exception(f"Error sending data to OPA: {e}")
        return False, [f"{policy_type}_opa_exception"]

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
