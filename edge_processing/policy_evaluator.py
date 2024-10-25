# policy_evaluator.py

import requests
import json
import logging
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score
import foolbox as fb
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

        # Send metrics to OPA for policy evaluation
        policy_url = OPA_SERVER_URL + POLICIES['fairness']
        response = requests.post(policy_url, json={"input": input_data})
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Result: {result}")
        allowed = result.get("result", False)
        failed_policies = []

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


def evaluate_reliability_policy(model, X_test, y_test):
    """
    Evaluates model reliability using adversarial attacks via Foolbox.

    Args:
        model (tf.keras.Model): The machine learning model.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): True labels for test data.

    Returns:
        float: Reliability score.
    """
    try:
        # Ensure TensorFlow's eager execution is enabled
        tf.config.run_functions_eagerly(True)

        # Verify input bounds and normalize if necessary
        if X_test.min() < 0 or X_test.max() > 1:
            logger.info(f"Normalizing X_test to [0, 1] range. Original min: {X_test.min()}, max: {X_test.max()}")
            X_test = X_test.astype('float32')
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
            logger.info(f"After normalization - min: {X_test.min()}, max: {X_test.max()}")

        # Convert NumPy arrays to TensorFlow tensors
        X_test_tf = tf.convert_to_tensor(X_test)
        y_test_tf = tf.convert_to_tensor(y_test)

        # Create a Foolbox model
        fmodel = fb.TensorFlowModel(model, bounds=(0, 1))

        # Create an attack
        attack = fb.attacks.LinfPGD()

        # Run the attack
        raw_advs, clipped_advs, success = attack(fmodel, X_test_tf, y_test_tf, epsilons=0.03)

        logger.info(f"Adversarial examples generated: {raw_advs.shape}")
        logger.info(f"Clipped adversarial examples: {clipped_advs.shape}")
        logger.info(f"Attack Success Rate: {np.mean(success)}")

        # Calculate the success rate of the attack
        success_rate = np.mean(success)

        # Reliability score is inversely related to the success rate of the attack
        reliability_score = 1 - success_rate

        return reliability_score
    except Exception as e:
        logger.exception(f"Error during reliability evaluation: {e}")
        return 0.0

def evaluate_explainability_policy(model, X_sample):
    """
    Evaluates model explainability using SHAP's KernelExplainer.

    Args:
        model (tf.keras.Model): The machine learning model.
        X_sample (np.ndarray): Sample input data for explanation.

    Returns:
        float: Explainability score.
    """
    try:
        # Select a background dataset for Integrated Gradients
        # Typically, a small subset of the training data
        background_size = min(100, X_sample.shape[0])
        if background_size < 100:
            logger.warning(f"Insufficient background samples. Using {background_size} samples instead of 100.")

        background = X_sample[:background_size]

        # Initialize the SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Calculate explainability score as the mean absolute SHAP value
        # For classification, shap_values is a list (one per class). Assume binary classification.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Assuming index 1 corresponds to the positive class

        explainability_score = np.mean(np.abs(shap_values))
        
        return explainability_score
    except Exception as e:
        logger.exception(f"Error during explainability evaluation: {e}")
        return 0.0