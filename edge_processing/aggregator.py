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
from flask import Flask, request, jsonify
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from typing import Dict

import tensorflow as tf
import numpy as np
import pandas as pd
import tarfile
import shutil
import tempfile

from utils.policy_evaluator import *
import schedule
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# MLflow Configuration
# mlflow.set_tracking_uri("http://10.200.3.159:5002")
mlflow.set_tracking_uri("http://mlflow-service:5002")
mlflow.set_experiment("Model_Evaluation")
# Initialize MLflow client
mlflow_client = MlflowClient()

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Temporary folder for uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Number of end devices expected
EXPECTED_DEVICES = int(os.getenv('EXPECTED_DEVICES', 1))  # Set accordingly

# Dictionary to store received models
received_models = {}
lock = threading.Lock()

# Load thresholds
thresholds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../opa/policies/')
# thresholds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opa/policies/')
logger.info(f"Loading thresholds from {thresholds_path}")
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

CLOUD_API_URL = os.getenv('CLOUD_API_URL', 'http://cloud-service:8080')
SYNC_INTERVAL_MINUTES = int(os.getenv('SYNC_INTERVAL_MINUTES', 30))

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        logger.info("Received upload_model request")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request form data: {request.form}")
        logger.info(f"Request files: {request.files}")
        
        if 'model' not in request.files:
            logger.error("No model file in request.files")
            return jsonify({'error': 'No model file provided'}), 400
            
        device_id = request.form.get('device_id')
        model_type = request.form.get('model_type')
        data_type = request.form.get('data_type')
        callback_url = request.form.get('callback_url')
        
        logger.info(f"Received request from device_id: {device_id}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Data type: {data_type}")
        logger.info(f"Callback URL: {callback_url}")
        
        if not all([device_id, model_type, data_type, callback_url]):
            logger.error(f"Missing required fields. Got: device_id={device_id}, model_type={model_type}, data_type={data_type}, callback_url={callback_url}")
            return jsonify({'error': 'Missing required fields'}), 400
            
        model_file = request.files['model']
        logger.info(f"Reading model file: {model_file.filename}")
        model_bytes = model_file.read()
        logger.info(f"Read {len(model_bytes)} bytes from model file")
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        logger.info(f"Received model from {device_id}, size: {len(model_bytes)} bytes")
        
        with lock:
            if device_id not in received_models:
                received_models[device_id] = {
                    'model_type': model_type,
                    'model_data': model_b64,
                    'data_type': data_type,
                    'callback_url': callback_url
                }
                logger.info(f"Stored model from {device_id}")
                
                # After receiving model, evaluate and aggregate
                logger.info("Starting evaluation and aggregation")
                evaluate_and_aggregate()
                logger.info("Completed evaluation and aggregation")
                return jsonify({'status': 'success', 'message': 'Model received and processing started'}), 200
            else:
                logger.warning(f"Model already received from device {device_id}")
                return jsonify({'error': 'Model already received from this device'}), 409
                
    except Exception as e:
        logger.exception(f"Error processing model upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_aggregated_model/<model_type>', methods=['GET'])
def get_aggregated_model(model_type):
    try:
        if model_type == 'MobileNet':
            model_path = MODEL_PATHS[model_type]['current']
            if not os.path.exists(model_path):
                return jsonify({'error': 'No aggregated model available'}), 404
                
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
        else:  # TinyBERT or T5
            model_dir = MODEL_PATHS[model_type]['current']
            if not os.path.exists(model_dir):
                return jsonify({'error': 'No aggregated model available'}), 404
                
            # Create temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                shutil.make_archive(tmp.name[:-7], 'gztar', model_dir)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
                os.unlink(tmp.name)
        
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        return jsonify({
            'model_data': model_b64,
            'model_type': model_type
        }), 200
        
    except Exception as e:
        logger.exception(f"Error serving aggregated model: {e}")
        return jsonify({'error': str(e)}), 500

def evaluate_and_aggregate():
    with lock:
        # Group received models by model_type
        model_types = set()
        for model_info in received_models.values():
            model_types.add(model_info['model_type'])
        
        for model_type in model_types:
            # Ensure any existing run is ended before starting a new one
            try:
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.end_run()
            except Exception as e:
                logger.debug(f"No active MLflow run to end: {e}")

            models_of_type = {device_id:info for device_id, info in received_models.items() 
                            if info['model_type'] == model_type}
            
            if len(models_of_type) >= EXPECTED_DEVICES:
                logger.info(f"All models of type '{model_type}' received. Evaluating policies.")

                # Get the data type from the first model (assuming all models of same type use same data)
                data_type = next(iter(models_of_type.values())).get('data_type', 'chest_xray')
                logger.info(f"Processing models with data_type: {data_type}")

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
                    
                    # After successful aggregation, send to edge devices
                    publish_aggregated_model(model_type, current_path)
                    
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
                        if data_type == 'chest_xray':
                            from dataset.chest_xray_processor import process_chest_xray_data
                            X_val, y_val, sensitive_features = [], [], []
                            for batch_X, batch_y, batch_sensitive in process_chest_xray_data("dataset/chest_xray/val", batch_size=32):
                                X_val.append(batch_X)
                                y_val.append(batch_y)
                                sensitive_features.append(batch_sensitive)
                            # Concatenate all batches into single arrays
                            X_val = np.concatenate(X_val, axis=0)
                            y_val = np.concatenate(y_val, axis=0)
                            sensitive_features = np.concatenate(sensitive_features, axis=0)
                            
                            logger.info(f"Chest X-ray validation data loaded - X_val: {X_val.shape}, y_val: {y_val.shape}, sensitive_features: {sensitive_features.shape}")
                            
                            # Convert to DataFrame for privacy evaluation
                            df_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
                            df_val['gender'] = sensitive_features
                            QUASI_IDENTIFIERS = ['gender']
                            
                        elif data_type == 'cxr8':
                            from dataset.cxr8_processor import process_cxr8_data
                            _, val_gen = process_cxr8_data(batch_size=32, max_samples=1000)  # Use smaller sample size
                            
                            # Process validation data
                            X_val_list = []
                            y_val_list = []
                            sensitive_features_list = []
                            
                            # Collect a few batches for validation
                            num_val_batches = 5  # Limit number of validation batches
                            for i, (batch_X, batch_y, batch_sensitive) in enumerate(val_gen):
                                if i >= num_val_batches:
                                    break
                                X_val_list.append(batch_X)
                                y_val_list.append(batch_y)
                                sensitive_features_list.append(batch_sensitive)
                            
                            # Check if we have any data
                            if not X_val_list:
                                logger.error("No validation data collected for CXR8")
                                return
                            
                            # Concatenate batches
                            X_val = np.concatenate(X_val_list, axis=0)
                            y_val = np.concatenate(y_val_list, axis=0)
                            sensitive_features = pd.concat(sensitive_features_list, ignore_index=True)
                            
                            logger.info(f"CXR8 validation data loaded - X_val: {X_val.shape}, y_val: {y_val.shape}, sensitive_features: {len(sensitive_features)}")
                            
                            # Prepare DataFrame for privacy evaluation
                            df_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
                            df_val['gender'] = sensitive_features['gender']
                            df_val['age_group'] = sensitive_features['age_group']
                            QUASI_IDENTIFIERS = ['gender', 'age_group']
                            
                        else:
                            logger.error(f"Unsupported data type for MobileNet: {data_type}")
                            return
                            
                    elif model_type in ['t5_small', 'tinybert']:
                        # Process medical transcription data
                        from dataset.mt_processor import process_medical_transcriptions_data
                        X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data("dataset/mt")
                        X_val = X_test
                        y_val = y_test
                        sensitive_features = sf_test
                        
                        logger.info(f"Text model validation data loaded - X_val: {len(X_val)}, y_val: {len(y_val)}, sensitive_features shape: {sensitive_features.shape}")
                        
                        # Prepare DataFrame for privacy evaluation
                        df_val = pd.DataFrame({
                            'text': X_val,
                            'gender': sensitive_features['gender'],
                            'age_group': sensitive_features['age_group']
                        })
                        QUASI_IDENTIFIERS = ['gender', 'age_group']
                    else:
                        logger.error(f"Unsupported model type for validation data: {model_type}")
                        return

                    # Start MLflow run
                    with mlflow.start_run(run_name=f"AggregatedModel_Evaluation_{model_type}"):
                        try:
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

                            if not is_private:
                                notify_policy_failure(failed_privacy_policies)

                        except Exception as e:
                            logger.exception(f"Error during model evaluation: {e}")
                            mlflow.end_run(status="FAILED")
                            return

                except Exception as e:
                    logger.exception(f"Error preparing validation data: {e}")
                    return

                # # Log data details
                # logger.info(f"Model Type: {model_type}")
                # logger.info(f"Number of samples - X_val: {len(X_val)}, y_val: {len(y_val)}, sensitive_features: {len(sensitive_features)}")

                # Evaluate fairness using the policy evaluator
                if model_type == 'MobileNet':
                    is_fair, failed_fairness_policies = evaluate_fairness_policy(
                        model=aggregated_model,
                        X=X_val,
                        y_true=y_val,
                        sensitive_features=sensitive_features,
                        thresholds=fairness_thresholds
                    )
                elif model_type == 'tinybert':
                    # Get the tokenizer for TinyBERT
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
                    
                    # Generate predictions with proper sequence length handling
                    inputs = tokenizer(
                        X_val.tolist(),
                        return_tensors="tf",
                        padding=True,
                        truncation=True,
                        max_length=512,  # TinyBERT's maximum sequence length
                        return_attention_mask=True
                    )
                    
                    # Process in batches to manage memory
                    batch_size = 32
                    all_predictions = []
                    
                    for i in range(0, len(X_val), batch_size):
                        batch_texts = X_val[i:i + batch_size].tolist()
                        batch_inputs = tokenizer(
                            batch_texts,
                            return_tensors="tf",
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_attention_mask=True
                        )
                        
                        # Get predictions for batch
                        outputs = aggregated_model(
                            input_ids=batch_inputs["input_ids"],
                            attention_mask=batch_inputs["attention_mask"],
                            token_type_ids=batch_inputs["token_type_ids"],
                            training=False
                        )
                        batch_predictions = tf.nn.softmax(outputs.logits, axis=-1).numpy()
                        all_predictions.append(batch_predictions)
                    
                    # Combine all batch predictions
                    predictions = np.vstack(all_predictions)
                    
                    # Convert predictions to categorical values for fairness evaluation
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_pred = predictions.argmax(axis=1)  # Get class with highest probability
                    y_true = le.fit_transform([label.strip().lower() for label in y_val])
                    
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
                    is_reliable, failed_reliability_policies = evaluate_reliability_policy(aggregated_model, X_val, y_val, reliability_thresholds)
                
                # Add TinyBERT-specific explainability and reliability evaluations
                if model_type == 'tinybert':
                    is_explainable, failed_explainability_policies = evaluate_explainability_policy_tinybert(
                        model=aggregated_model,
                        X_val=X_val,
                        tokenizer=tokenizer,
                        thresholds=explainability_thresholds
                    )
                    
                    is_reliable, failed_reliability_policies = evaluate_reliability_policy_tinybert(
                        model=aggregated_model,
                        X_val=X_val,
                        tokenizer=tokenizer,
                        thresholds=reliability_thresholds
                    )
    
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
                        # Create a pipeline for the model
                        from transformers import Pipeline, pipeline
                        nlp_pipeline = pipeline(
                            task="text-classification",
                            model=aggregated_model,
                            tokenizer=tokenizer
                        )
                        mlflow.transformers.log_model(
                            nlp_pipeline,
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
                    failed_policies = []
                    if failed_privacy_policies: failed_policies.append(failed_privacy_policies[0])
                    if failed_fairness_policies: failed_policies.append(failed_fairness_policies[0])
                    if failed_explainability_policies: failed_policies.append(failed_explainability_policies[0])
                    if failed_reliability_policies: failed_policies.append(failed_reliability_policies[0])
                    
                    logger.warning(f"Aggregated {model_type} model failed policies: {failed_policies}. Retaining previous model.")
                    notify_policy_failure(failed_policies)
                    mlflow.log_param("failed_policies", failed_policies)
                    
                    # Use the previous aggregated model if it exists
                    if os.path.exists(previous_path):
                        publish_aggregated_model(model_type, previous_path)
                        logger.info(f"Published previous aggregated {model_type} model")
                    else:
                        logger.warning("No previous aggregated model available to deploy.")
                    
                    # Remove received models of this type
                    for device_id in list(received_models.keys()):
                        if received_models[device_id]['model_type'] == model_type:
                            del received_models[device_id]

def adaptive_aggregation(models_of_type: Dict[str, Dict], model_type: str) -> Dict[str, float]:
    """
    Compute adaptive weights for model aggregation based on data diversity and underrepresentation.
    
    Args:
        models_of_type: Dictionary of models with their metadata
        model_type: Type of the model being aggregated
        
    Returns:
        Dictionary mapping device_ids to their computed weights
    """
    import numpy as np
    from scipy.stats import entropy
    
    device_weights = {}
    total_samples = sum(len(info.get('data_distribution', [])) for info in models_of_type.values())
    
    # Compute entropy and inverse frequency for each device
    for device_id, info in models_of_type.items():
        data_dist = np.array(info.get('data_distribution', []))
        if len(data_dist) == 0:
            logger.warning(f"No data distribution information for device {device_id}")
            device_weights[device_id] = 1.0 / len(models_of_type)
            continue
            
        # Normalize distribution
        data_dist = data_dist / np.sum(data_dist)
        
        # Compute entropy (diversity)
        H_k = entropy(data_dist)
        
        # Compute inverse frequency (underrepresentation)
        global_dist = np.zeros_like(data_dist)
        for other_info in models_of_type.values():
            other_dist = np.array(other_info.get('data_distribution', []))
            if len(other_dist) > 0:
                global_dist += other_dist
        global_dist = global_dist / np.sum(global_dist)
        
        # Avoid division by zero
        global_dist = np.maximum(global_dist, 1e-10)
        gamma_k = np.sum(data_dist / global_dist) / len(data_dist)
        
        # Combine metrics with balancing parameter beta
        beta = 0.5  # Can be adjusted based on requirements
        combined_score = beta * H_k + (1 - beta) * gamma_k
        device_weights[device_id] = combined_score
    
    # Normalize weights
    total_weight = sum(device_weights.values())
    if total_weight > 0:
        device_weights = {k: v/total_weight for k, v in device_weights.items()}
    else:
        # Fallback to uniform weights if something goes wrong
        device_weights = {k: 1.0/len(models_of_type) for k in models_of_type.keys()}
    
    return device_weights

def aggregate_models(models_of_type, model_type, save_path):
    """
    Aggregate models by averaging their weights based on model_type.
    """
    from transformers import TFT5ForConditionalGeneration, TFAutoModelForSequenceClassification
    
    # Compute adaptive weights for aggregation
    device_weights = adaptive_aggregation(models_of_type, model_type)
    logger.info(f"Computed adaptive weights for aggregation: {device_weights}")

    models = []
    weights = []
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
        weights.append(device_weights[device_id])

    if not models:
        raise ValueError("No models to aggregate.")

    if model_type == 'MobileNet':
        # Aggregate MobileNet models with adaptive weights
        aggregated_model = models[0]
        for layer in aggregated_model.layers:
            if layer.weights:  # Only process layers with weights
                weighted_weights = []
                for model, weight in zip(models, weights):
                    layer_weights = model.get_layer(layer.name).get_weights()
                    weighted_weights.append([w * weight for w in layer_weights])
                
                # Sum up the weighted weights
                new_weights = []
                for layer_weights in zip(*weighted_weights):
                    new_weights.append(sum(w for w in layer_weights))
                
                layer.set_weights(new_weights)
                
        # Save the aggregated model
        aggregated_model.save(save_path, save_format='keras')

    elif model_type in ['t5_small', 'tinybert']:
        # Aggregate transformer models (T5 or TinyBERT) with adaptive weights
        aggregated_model = models[0]
        for param_name, param in aggregated_model.trainable_weights:
            weighted_params = []
            for model, weight in zip(models, weights):
                other_param = next(p for n, p in model.trainable_weights if n == param_name)
                weighted_params.append(other_param * weight)
            
            # Sum up the weighted parameters
            param.assign(sum(weighted_params))
        
        # Create directory for saving if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        # Save the aggregated model
        aggregated_model.save_pretrained(save_path)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def publish_aggregated_model(model_type, model_path):
    """
    Send the aggregated model back to the edge devices.
    """
    try:
        # Read the model file
        if model_type == 'MobileNet':
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
        else:  # TinyBERT or T5
            # Create temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                shutil.make_archive(tmp.name[:-7], 'gztar', model_path)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
                os.unlink(tmp.name)
        
        # Encode model to base64
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Prepare payload
        payload = {
            'model_type': model_type,
            'model_data': model_b64
        }
        
        # Send to each edge device that contributed
        with lock:
            for device_id, info in received_models.items():
                if info['model_type'] == model_type:
                    callback_url = info['callback_url']
                    try:
                        logger.info(f"Sending aggregated model to device {device_id} at {callback_url}")
                        response = requests.post(
                            callback_url,
                            json=payload,
                            headers={'Content-Type': 'application/json'}
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"Successfully sent aggregated model to device {device_id}")
                        else:
                            logger.error(f"Failed to send aggregated model to device {device_id}: {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        logger.exception(f"Error sending aggregated model to device {device_id}: {e}")
            
            # Clear received models of this type after sending
            received_models.clear()
            
    except Exception as e:
        logger.exception(f"Error in publish_aggregated_model: {e}")

def notify_policy_failure(failed_policies):
    """
    Notify stakeholders about the failed policies.
    """
    logger.warning(f"Policies failed: {failed_policies}")
    # Implement additional notification mechanisms as needed (e.g., email, alerts)

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

def sync_with_cloud():
    """
    Sync aggregated models with the cloud layer periodically.
    """
    logger.info("Starting cloud synchronization...")
    
    try:
        # Prepare model updates for each model type
        for model_type, paths in MODEL_PATHS.items():
            current_path = paths['current']
            
            if not os.path.exists(current_path):
                logger.warning(f"No aggregated model found for {model_type}")
                continue
                
            # Read and encode the model
            if os.path.isdir(current_path):
                # For directory-based models (T5, TinyBERT)
                with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                    shutil.make_archive(tmp.name[:-7], 'gztar', current_path)
                    with open(tmp.name, 'rb') as f:
                        model_bytes = f.read()
                    os.unlink(tmp.name)
            else:
                # For single file models (MobileNet)
                with open(current_path, 'rb') as f:
                    model_bytes = f.read()
            
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            upload_thresholds = {
                'fairness': fairness_thresholds,
                'reliability': reliability_thresholds,
                'explainability': explainability_thresholds,
                'privacy': privacy_thresholds
            }
            # Prepare metrics from latest evaluation
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'thresholds': upload_thresholds
            }
            
            # Send to cloud
            payload = {
                'edge_server_id': 'edge_processor_1',  # Unique ID for this edge server
                'model_params': model_b64,
                'metrics': metrics,
                'model_type': model_type
            }
            
            response = requests.post(f"{CLOUD_API_URL}/edge/update", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success' and 'model' in result:
                    # Update local model with global model
                    global_model_b64 = result['model']
                    global_model_bytes = base64.b64decode(global_model_b64)
                    
                    # Backup current model
                    if os.path.exists(paths['previous']):
                        if os.path.isdir(paths['previous']):
                            shutil.rmtree(paths['previous'])
                        else:
                            os.remove(paths['previous'])
                    
                    # Save global model
                    if os.path.isdir(current_path):
                        # Extract directory-based model
                        if os.path.exists(current_path):
                            shutil.rmtree(current_path)
                        os.makedirs(current_path)
                        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                            tmp.write(global_model_bytes)
                            with tarfile.open(tmp.name, 'r:gz') as tar:
                                tar.extractall(path=current_path)
                            os.unlink(tmp.name)
                    else:
                        # Save single file model
                        with open(current_path, 'wb') as f:
                            f.write(global_model_bytes)
                    
                    logger.info(f"Updated local {model_type} model with global model")
                    
                    # Distribute updated model to edge devices
                    publish_aggregated_model(model_type, current_path)
                else:
                    logger.warning(f"Cloud sync failed for {model_type}: {result.get('message')}")
            else:
                logger.error(f"Failed to sync with cloud: {response.status_code}")
                
    except Exception as e:
        logger.exception(f"Error during cloud synchronization: {e}")

def schedule_cloud_sync():
    """
    Schedule periodic cloud synchronization
    """
    schedule.every(SYNC_INTERVAL_MINUTES).minutes.do(sync_with_cloud)
    
    while True:
        schedule.run_pending()
        time.sleep(30)  # Check every minute

def create_app():
    """Create and configure the Flask app for gunicorn"""
    # Start cloud sync in a separate thread
    sync_thread = threading.Thread(target=schedule_cloud_sync)
    sync_thread.daemon = True
    sync_thread.start()
    logger.info("Started cloud synchronization thread")
    
    # Start MLflow run
    mlflow.start_run(run_name="Model_Aggregation_Parent")
    
    return app

def main():
    try:
        # Import gunicorn's workers
        from gunicorn.app.base import BaseApplication

        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            'bind': '0.0.0.0:8000',
            'workers': 3,
            'timeout': 300,
            'worker_class': 'gthread',  # Use threaded worker
            'worker_connections': 1000,
            'threads': 3,
            'keepalive': 5,
            'max_requests': 0,  # Disable max requests
            'max_requests_jitter': 0
        }

        logger.info("Starting gunicorn server to receive models...")
        StandaloneApplication(create_app(), options).run()
            
    except Exception as e:
        logger.exception("Error in main loop")
    finally:
        # Ensure MLflow run is ended
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

if __name__ == "__main__":
    main()
