# aggregator/policy_evaluator.py

import requests
import json
import logging
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference

OPA_URL = "http://10.200.3.99:8181/v1/data/policies/fairness/allow"  # Adjust if OPA runs on a different host/port

def evaluate_fairness(model, X, y_true, sensitive_features, thresholds):
    """
    Evaluates model fairness using OPA policies.

    Args:
        model (tf.keras.Model): The machine learning model.
        X (pd.DataFrame): Input features.
        y_true (pd.Series): True labels.
        sensitive_features (pd.Series): Protected attribute(s).
        thresholds (dict): Thresholds for fairness metrics.

    Returns:
        bool: True if all fairness policies are satisfied, False otherwise.
        list: List of failed policies.
    """
    try:
        # Generate predictions
        y_pred = model.predict(X).flatten()
        y_pred_binary = (y_pred >= 0.5).astype(int)  # Assuming binary classification

        # Calculate fairness metrics
        metric_frame = MetricFrame(
            metrics={
                "demographic_parity": demographic_parity_difference,
                # "equal_opportunity": equalized_odds_difference
            },
            y_true=y_true,
            y_pred=y_pred_binary,
            sensitive_features=sensitive_features
        )

        dp_diff = metric_frame.metrics["demographic_parity"]
        # eo_diff = metric_frame.metrics["equal_opportunity"]

        model_metrics = {
            "demographic_parity": abs(dp_diff),
            # "equal_opportunity": abs(eo_diff)
        }

        input_data = {
            "fairness": {
                "metrics": model_metrics,
                "threshold": thresholds
            }
        }

        # Send to OPA for evaluation
        response = requests.post(OPA_URL, json={"input": input_data})
        response.raise_for_status()
        result = response.json()

        allowed = result.get("result", False)
        failed_policies = []

        if allowed:
            logging.info("Model passed all fairness policies.")
            return True, []
        else:
            logging.warning("Model failed fairness policies.")
            # Determine which policies failed based on metrics
            if model_metrics["demographic_parity"] > thresholds["demographic_parity"]:
                failed_policies.append("demographic_parity")
            # if model_metrics["equal_opportunity"] > thresholds["equal_opportunity"]:
            #     failed_policies.append("equal_opportunity")
            return False, failed_policies

    except requests.exceptions.RequestException as e:
        logging.exception("Failed to communicate with OPA.")
        return False, ["OPA Communication Error"]
    except Exception as e:
        logging.exception(f"Error during fairness evaluation: {e}")
        return False, ["Fairness Evaluation Error"]
