# policy_evaluator.py

import requests
import json
import logging
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score
import foolbox as fb
from foolbox import Criterion
import tensorflow as tf
import shap
import yaml
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# OPA_URL = "http://10.200.3.99:8181/v1/data/policies/fairness/demographic_parity/allow"

# POLICY_URLS = {
#     "fairness": "http://10.200.3.99:8181/v1/data/policies/fairness/demographic_parity/allow",
#     "reliability": "http://10.200.3.99:8181/v1/data/policies/reliability/allow",
#     "explainability": "http://10.200.3.99:8181/v1/data/policies/explainability/allow"
# }

opa_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opa_config.yaml')
with open(opa_config_path, 'r') as file:
    config = yaml.safe_load(file)

OPA_SERVER_URL = config['opa_server_url']
POLICIES = config['policies']

def evaluate_fairness_policy(model, X, y_true, sensitive_features, thresholds):
    """
    Evaluates model fairness using OPA policies.

    Args:
        model (tf.keras.Model): The machine learning model.
        X (pd.DataFrame or np.ndarray): Input features.
        y_true (pd.Series or np.ndarray): True labels.
        sensitive_features (pd.Series or np.ndarray): Protected attribute(s).
        thresholds (dict): Thresholds for fairness metrics.

    Returns:
        bool: True if all fairness policies are satisfied, False otherwise.
        list: List of failed policies.
    """
    try:
        # Generate predictions
        y_pred = model.predict(X)
        y_pred_binary = (y_pred >= 0.5).astype(int)  # Assuming binary classification

        # Log data details
        # logger.info(f"Evaluating model fairness:")
        # logger.info(f"X shape: {X.shape}")
        # logger.info(f"y_true shape: {y_true.shape}")
        # logger.info(f"sensitive_features shape: {sensitive_features.shape}")
        # logger.info(f"sensitive_features sample: {sensitive_features}")

        logger.info(f"Number of samples - X_val: {len(X)}, y_val: {len(y_pred_binary)}, sensitive_features: {len(sensitive_features)}")
        if np.isnan(sensitive_features).any():
            logger.error("sensitive_features contain NaN values.")
        sample_params = {'demographic_parity_difference': {'sensitive_features': sensitive_features}}
        # Create MetricFrame
        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "demographic_parity_difference": demographic_parity_difference,
            },
            y_true=y_true,
            y_pred=y_pred_binary,
            sensitive_features=sensitive_features,
            sample_params = sample_params
        )

        # Extract overall metrics
        model_metrics = metric_frame.overall.to_dict()
        
        logger.info(f"Model Metrics: {model_metrics}")
        
        '''
        ### via local policy
        # Check thresholds
        is_fair = True
        failed_policies = []
        for metric_name, value in metric_frame.overall.items():
            logger.info(f"metric_name: {metric_name}, value: {value}")
            threshold = thresholds.get(metric_name, None)
            if threshold is not None:
                if metric_name in ['demographic_parity_difference']:
                    # For these metrics, smaller absolute values are better
                    if abs(value) > threshold:
                        is_fair = False
                        failed_policies.append(metric_name)
                else:
                    # For metrics like accuracy, higher is better
                    if value < threshold:
                        is_fair = False
                        failed_policies.append(metric_name)
        logger.info(f"is_fair: {is_fair}, failed_policies: {failed_policies}")
        return is_fair, failed_policies
        '''
        
        ### via OPA
        input_data = {
            "fairness": {
                "metrics": model_metrics,
                "threshold": thresholds
            }
        }

        allowed, failed_policies= send_to_opa(input_data, "fairness")

        if allowed:
            logger.info("Model passed all fairness policies.")
            return True, []
        else:
            logger.warning("Model failed fairness policies.")
            #For these metrics, smaller absolute values are better
            if model_metrics.get("demographic_parity_difference", 0) > thresholds.get("demographic_parity_difference", 0):
                failed_policies.append("demographic_parity")
            # For metrics like accuracy, higher is better
            if model_metrics.get("accuracy", 0) < thresholds.get("accuracy", 0):
                failed_policies.append("accuracy")
            return False, failed_policies
        
    except requests.exceptions.RequestException as e:
        logger.exception("Failed to communicate with OPA.")
        return False, ["OPA Communication Error"]
    except Exception as e:
        logger.exception(f"Error during fairness evaluation: {e}")
        return False, ["Fairness Evaluation Error"]


def evaluate_reliability_policy(model, X_test, y_test, thresholds):
    """
    Evaluates model reliability using adversarial attacks via Foolbox and sends metrics to OPA for policy evaluation.

    Args:
        model (tf.keras.Model): The machine learning model.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): True labels for test data.
        thresholds (dict): Thresholds for reliability metrics.

    Returns:
        bool: True if all reliability policies are satisfied, False otherwise.
        list: List of failed policies.
    """
    try:
        # Convert NumPy arrays to TensorFlow tensors
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int32)

        preprocessing = {
            "mean": 0.0,
            "std": 1.0
        }
        # Create a Foolbox model with logits=False since the model outputs probabilities
        fmodel = fb.TensorFlowModel(model, bounds=(0, 1), preprocessing=preprocessing)

        # Define the criterion for binary classification
        criterion = fb.criteria.Misclassification(labels=y_test)

        # Initialize the attack with the specified criterion
        attack = fb.attacks.LinfPGD(steps=40, epsilon=0.03, random_start=True, criterion=criterion)

        # Run the attack
        raw_advs, clipped_advs, success = attack(fmodel, X_test_tf, y_test_tf)

        logger.info(f"Adversarial examples generated: {raw_advs.shape}")
        logger.info(f"Clipped adversarial examples: {clipped_advs.shape}")
        logger.info(f"Attack Success Rate: {np.mean(success)}")

        # Calculate the success rate of the attack
        success_rate = np.mean(success)

        # Reliability score is inversely related to the success rate of the attack
        reliability_score = 1 - success_rate

        # Prepare metrics for OPA
        reliability_metrics = {
            "success_rate": float(success_rate),
            "reliability_score": float(reliability_score)
        }

        # Log reliability metrics
        logger.info(f"Reliability Metrics: {reliability_metrics}")

        # Prepare input data for OPA
        input_data = {
            "reliability": {
                "metrics": reliability_metrics,
                "threshold": thresholds
            }
        }

        # Send data to OPA for policy evaluation
        allowed, failed_policies = send_to_opa(input_data, "reliability")

        if allowed:
            logger.info("Model passed all reliability policies.")
            return True, []
        else:
            logger.warning("Model failed reliability policies.")
            return False, failed_policies

    except requests.exceptions.RequestException as e:
        logger.exception("Failed to communicate with OPA.")
        return False, ["OPA Communication Error"]
    except Exception as e:
        logger.exception(f"Error during reliability evaluation: {e}")
        return False, ["Reliability Evaluation Error"]

def evaluate_explainability_policy(model, X_sample, thresholds):
    """
    Evaluates model explainability using SHAP's GradientExplainer and sends metrics to OPA for policy evaluation.

    Args:
        model (tf.keras.Model): The machine learning model.
        X_sample (np.ndarray): Sample input data for explanation.
        thresholds (dict): Thresholds for explainability metrics.

    Returns:
        bool: True if all explainability policies are satisfied, False otherwise.
        list: List of failed policies.
    """
    try:
        # Ensure model is compiled
        if not model.optimizer:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Select a background dataset for SHAP
        background_size = min(100, X_sample.shape[0])
        if background_size < 100:
            logger.warning(f"Insufficient background samples. Using {background_size} samples instead of 100.")

        background = X_sample[:background_size]

        # Initialize the SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Assuming binary classification; select SHAP values for the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Index 1 corresponds to the positive class

        # Calculate explainability score as the mean absolute SHAP value
        explainability_score = np.mean(np.abs(shap_values))
        logger.info(f"Explainability Score: {explainability_score}")

        # Prepare metrics for OPA
        explainability_metrics = {
            "explainability_score": float(explainability_score)
        }

        # Log explainability metrics
        logger.info(f"Explainability Metrics: {explainability_metrics}")

        # Prepare input data for OPA
        input_data = {
            "explainability": {
                "metrics": explainability_metrics,
                "threshold": thresholds
            }
        }

        # Send data to OPA for policy evaluation
        allowed, failed_policies = send_to_opa(input_data, "explainability")

        if allowed:
            logger.info("Model passed all explainability policies.")
            return True, []
        else:
            logger.warning("Model failed explainability policies.")
            return False, failed_policies

    except requests.exceptions.RequestException as e:
        logger.exception("Failed to communicate with OPA.")
        return False, ["OPA Communication Error"]
    except Exception as e:
        logger.exception(f"Error during explainability evaluation: {e}")
        return False, ["Explainability Evaluation Error"]
    
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
                # Assuming the policy type corresponds to a single policy
                failed_policies.append(policy_type)
            return allowed, failed_policies
        else:
            logger.error(f"OPA request failed with status code {response.status_code}: {response.text}")
            return False, [f"{policy_type}_opa_request_failed"]
    except Exception as e:
        logger.exception(f"Error sending data to OPA: {e}")
        return False, [f"{policy_type}_opa_exception"]
