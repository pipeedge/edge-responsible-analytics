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
import tarfile
import shutil
import tempfile

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

# Model paths configuration
MODEL_PATHS = {
    'MobileNet': {
        'current': 'aggregated_mobilenet.keras',
        'previous': 'previous_mobilenet.keras'
    },
    't5_small': {
        'current': 'aggregated_t5',
        'previous': 'previous_t5'
    },
    'tinybert': {
        'current': 'aggregated_tinybert',
        'previous': 'previous_tinybert'
    }
}

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

                # Get the appropriate model paths
                model_paths = MODEL_PATHS.get(model_type)
                if not model_paths:
                    logger.error(f"No model path defined for model type: {model_type}")
                    continue

                current_path = model_paths['current']
                previous_path = model_paths['previous']

                # Clean up existing model files/directories
                if os.path.exists(current_path):
                    if os.path.isdir(current_path):
                        shutil.rmtree(current_path)
                    else:
                        os.remove(current_path)

                # Aggregate models
                try:
                    aggregate_models(models_of_type, model_type, current_path)
                    logger.info(f"Aggregated {model_type} model saved to {current_path}")
                except Exception as e:
                    logger.exception(f"Failed to aggregate {model_type} models: {e}")
                    continue

                # Load the aggregated model for evaluation
                try:
                    if model_type == 'MobileNet':
                        aggregated_model = tf.keras.models.load_model(current_path, compile=False)
                        aggregated_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    elif model_type == 't5_small':
                        from transformers import TFT5ForConditionalGeneration
                        aggregated_model = TFT5ForConditionalGeneration.from_pretrained(current_path)
                    elif model_type == 'tinybert':
                        from transformers import TFAutoModelForSequenceClassification
                        aggregated_model = TFAutoModelForSequenceClassification.from_pretrained(current_path)
                    else:
                        logger.error(f"Unsupported model_type: {model_type}")
                        continue
                except Exception as e:
                    logger.exception(f"Failed to load aggregated model: {e}")
                    continue

                # Prepare validation data
                try:
                    X_val, y_val, sensitive_features = None, None, None
                    
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
                        
                        logger.info(f"MobileNet validation data loaded - X_val: {X_val.shape}, y_val: {y_val.shape}, sensitive_features: {sensitive_features.shape}")
                        
                    elif model_type in ['t5_small', 'tinybert']:
                        # Process medical transcription data
                        from datasets.mt_processor import process_medical_transcriptions_data
                        X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data("datasets/mt/val")
                        X_val = X_test
                        y_val = y_test
                        sensitive_features = sf_test
                        
                        logger.info(f"Text model validation data loaded - X_val: {len(X_val)}, y_val: {len(y_val)}, sensitive_features shape: {sensitive_features.shape}")
                    else:
                        logger.error(f"Unsupported model type for validation data: {model_type}")
                        return

                    # Start MLflow run
                    with mlflow.start_run(run_name=f"AggregatedModel_Evaluation_{model_type}"):
                        try:
                            # Convert to DataFrame for privacy evaluation
                            if model_type == 'MobileNet':
                                df_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
                                df_val['gender'] = sensitive_features
                                QUASI_IDENTIFIERS = ['gender']
                            elif model_type in ['t5_small', 'tinybert']:
                                df_val = pd.DataFrame({
                                    'text': X_val,
                                    'gender': sensitive_features['gender'],
                                    'age_group': sensitive_features['age_group']
                                })
                                QUASI_IDENTIFIERS = ['gender', 'age_group']
                            
                            # Evaluate privacy
                            missing_columns = [col for col in QUASI_IDENTIFIERS if col not in df_val.columns]
                            if missing_columns:
                                logger.error(f"Missing quasi-identifier columns in df_val: {missing_columns}")
                                return
                                
                            is_private, failed_privacy_policies = evaluate_privacy_policy(
                                df=df_val,
                                quasi_identifiers=QUASI_IDENTIFIERS,
                                k_threshold=privacy_thresholds["k"]
                            )

                            # Rest of the evaluation code...

                        except Exception as e:
                            logger.exception(f"Error during model evaluation: {e}")
                            mlflow.end_run(status="FAILED")
                            return

                except Exception as e:
                    logger.exception(f"Error preparing validation data: {e}")
                    return

                # Log data details
                logger.info(f"Model Type: {model_type}")
                logger.info(f"Number of samples - X_val: {len(X_val)}, y_val: {len(y_val)}, sensitive_features: {len(sensitive_features)}")

                # Evaluate fairness using the policy evaluator
                if model_type == 'MobileNet':
                    is_fair, failed_fairness_policies = evaluate_fairness_policy(
                        model=aggregated_model,
                        X=X_val,
                        y_true=y_val,
                        sensitive_features=sensitive_features,
                        thresholds=fairness_thresholds
                    )
                elif model_type == 't5_small':
                    # For t5_small, evaluate fairness using sensitive features from mt_processor
                    from transformers import T5Tokenizer
                    tokenizer = T5Tokenizer.from_pretrained('t5-small')
                    
                    # Generate predictions
                    inputs = tokenizer(X_val.tolist(), return_tensors="tf", padding=True, truncation=True)
                    outputs = aggregated_model.generate(inputs.input_ids)
                    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Convert predictions to categorical values for fairness evaluation
                    # You might need to adjust this based on your specific task
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_pred = le.fit_transform([pred.strip().lower() for pred in predictions])
                    y_true = le.transform([label.strip().lower() for label in y_val])
                    
                    # Extract sensitive features
                    gender = sf_test['gender'].values
                    age_group = sf_test['age_group'].values
                    
                    # Evaluate fairness for each protected attribute
                    is_fair_gender, failed_gender = evaluate_fairness_policy(
                        model=None,  # Not needed as we pass predictions directly
                        X=None,
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=gender,
                        thresholds=fairness_thresholds
                    )
                    
                    is_fair_age, failed_age = evaluate_fairness_policy(
                        model=None,
                        X=None,
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=age_group,
                        thresholds=fairness_thresholds
                    )
                    
                    # Combine results
                    is_fair = is_fair_gender and is_fair_age
                    failed_fairness_policies = failed_gender + failed_age
                else:
                    is_fair, failed_fairness_policies = False, ["unsupported_model_type"]
    
                # Evaluate explainability
                if model_type == 'MobileNet':
                    is_explainable, failed_explainability_policies = evaluate_explainability_policy(aggregated_model, X_val, explainability_thresholds)
                
                # Evaluate reliability
                if model_type == 'MobileNet':
                    is_reliable, failed_reliability_policies = evaluate_reliability_policy(aggregated_model, X_val, y_val, reliability_thresholds)
    
                mlflow.log_param("threshold_demographic_parity_difference", fairness_thresholds.get("demographic_parity_difference", None))
                if model_type == 'MobileNet':
                    mlflow.log_metric("is_fair", int(is_fair))
                    mlflow.log_metric("is_explainable", int(is_explainable))
                    mlflow.log_metric("is_reliable", int(is_reliable))
                
    
                if is_fair and is_explainable and is_reliable:
                    logger.info(f"Aggregated {model_type} model passed all policies. Publishing the model.")
                    publish_aggregated_model(model_type, current_path)
                    
                    # Log the aggregated Model 
                    if model_type == 'MobileNet':
                        mlflow.keras.log_model(
                            aggregated_model, 
                            "model", 
                            signature=infer_signature(X_val, aggregated_model.predict(X_val))
                        )
                    elif model_type in ['t5_small', 'tinybert']:
                        mlflow.transformers.log_model(
                            aggregated_model, 
                            "model"
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
                    if os.path.exists(previous_path):
                        if os.path.isdir(previous_path):
                            shutil.rmtree(previous_path)
                        else:
                            os.remove(previous_path)
                    
                    if os.path.isdir(current_path):
                        shutil.copytree(current_path, previous_path)
                    else:
                        shutil.copy2(current_path, previous_path)
                        
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
                    if os.path.exists(previous_path):
                        publish_aggregated_model(model_type, previous_path)
                        logger.info(f"Published previous aggregated {model_type} model")
                    else:
                        logger.error("No previous aggregated model available to deploy.")
                    
                    # Remove received models of this type
                    for device_id in list(received_models.keys()):
                        if received_models[device_id]['model_type'] == model_type:
                            del received_models[device_id]

def aggregate_models(models_of_type, model_type, save_path):
    """
    Aggregate models by averaging their weights based on model_type.
    """
    from transformers import TFT5ForConditionalGeneration, TFAutoModelForSequenceClassification

    models = []
    for device_id, model_info in models_of_type.items():
        model_data = base64.b64decode(model_info['model_data'])
        if model_type == 'MobileNet':
            temp_model_path = f"temp_{device_id}.keras"
            with open(temp_model_path, 'wb') as f:
                f.write(model_data)
            model = tf.keras.models.load_model(temp_model_path, compile=False)
            os.remove(temp_model_path)
        elif model_type in ['t5_small', 'tinybert']:
            # Handle directory-based models (T5 and TinyBERT)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
                tmp_file.write(model_data)
                temp_model_path = tmp_file.name
            
            # Extract the tar file
            temp_dir = f"temp_model_{device_id}"
            os.makedirs(temp_dir, exist_ok=True)
            with tarfile.open(temp_model_path, 'r:gz') as tar:
                tar.extractall(path=temp_dir)
            
            # Load the appropriate model type
            if model_type == 't5_small':
                model = TFT5ForConditionalGeneration.from_pretrained(temp_dir)
            else:  # tinybert
                model = TFAutoModelForSequenceClassification.from_pretrained(temp_dir)
            
            # Cleanup
            os.remove(temp_model_path)
            shutil.rmtree(temp_dir)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        models.append(model)

    if not models:
        raise ValueError("No models to aggregate.")

    if model_type == 'MobileNet':
        # Aggregate MobileNet models
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
        # Save the aggregated model
        aggregated_model.save(save_path, save_format='keras')

    elif model_type in ['t5_small', 'tinybert']:
        # Aggregate transformer models (T5 or TinyBERT)
        aggregated_model = models[0]
        for model in models[1:]:
            # Average weights for all parameters
            for param_name, param in aggregated_model.trainable_weights:
                other_param = next(p for n, p in model.trainable_weights if n == param_name)
                param.assign((param + other_param) / 2)
        
        # Create directory for saving if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        # Save the aggregated model
        aggregated_model.save_pretrained(save_path)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def publish_aggregated_model(model_type, model_path):
    """
    Publish the aggregated model to the MQTT broker.
    """
    try:
        if model_type == 'MobileNet':
            # Handle single file model
            with open(model_path, 'rb') as f:
                aggregated_model_bytes = f.read()
        else:  # Directory-based models (T5, TinyBERT)
            # Create a temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                # Create tar archive of the model directory
                shutil.make_archive(tmp.name[:-7], 'gztar', model_path)
                with open(tmp.name, 'rb') as f:
                    aggregated_model_bytes = f.read()
                os.unlink(tmp.name)  # Clean up temp file

        aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')
        payload = json.dumps({
            'model_data': aggregated_model_b64,
            'model_type': model_type
        })
        client.publish(MQTT_TOPIC_AGGREGATED, payload)
        logger.info(f"Published aggregated {model_type} model to {MQTT_TOPIC_AGGREGATED}")
    except Exception as e:
        logger.error(f"Failed to publish aggregated model: {e}")

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
