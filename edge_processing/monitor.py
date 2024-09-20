import numpy as np
# import mlflow
import prometheus_client as prom
import os
import requests
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score

# Set up MLflow tracking
# mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'))
# mlflow.set_experiment("Edge_Responsible_Analytics")

# Set up Prometheus metrics
fairness_metric = prom.Gauge('model_fairness', 'Fairness metric of the model')
security_metric = prom.Gauge('model_security', 'Security metric of the model')

class ResponsibleMonitor:
    def __init__(self):
        self.X_test = None
        self.y_test = None
        self.sensitive_features = None

    def set_test_data(self, X_test, y_test, sensitive_features):
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features

    def evaluate_responsible_metrics(self, model):
        """Evaluate fairness and security metrics of the model."""
        if self.X_test is None or self.y_test is None or self.sensitive_features is None:
            raise ValueError("Test data or sensitive features are not set.")
        
        # Evaluate fairness using Fairlearn
        y_pred = model.predict(self.X_test)
        
        # Create a MetricFrame to evaluate fairness
        metric_frame = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'selection_rate': selection_rate,
                'demographic_parity_difference': demographic_parity_difference
            },
            y_true=self.y_test,
            y_pred=y_pred,
            sensitive_features=self.sensitive_features
        )
        
        # Extract fairness metrics
        fairness_score = 1 - metric_frame.overall['demographic_parity_difference']
        security_score = np.random.uniform(0.8, 1.0)  # Simulated security score this stage
        
        fairness_metric.set(fairness_score)
        security_metric.set(security_score)
        
        return fairness_score, security_score

    # def log_metrics(self, fairness, security):
    #     with mlflow.start_run():
    #         mlflow.log_metric("fairness", fairness)
    #         mlflow.log_metric("security", security)

    def evaluate_policy(self, fairness, security):
        """Evaluate the model against a policy using OPA."""
        opa_url = os.getenv('OPA_URL', 'http://localhost:8181/v1/data/model/allow')
        input_data = {
            "input": {
                "fairness": fairness,
                "security": security
            }
        }
        
        response = requests.post(opa_url, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            return result['result']
        else:
            return None
