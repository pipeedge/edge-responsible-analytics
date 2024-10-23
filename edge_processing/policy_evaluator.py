# policy_evaluator.py

import requests
import json
import logging
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

OPA_URL = "http://10.200.3.99:8181/v1/data/policies/fairness/allow"

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
        response = requests.post(OPA_URL, json={"input": input_data})
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
