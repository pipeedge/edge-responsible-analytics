import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import tensorflow as tf

def register_model(model_path, model_name):
    client = MlflowClient()
    mlflow.set_experiment("Global_Model_Management")
    
    with mlflow.start_run(run_name="Model_Aggregation"):
        model = tf.keras.models.load_model(model_path)
        # Optionally, perform hyperparameter optimization here
        
        # Log the model
        mlflow.keras.log_model(model, "model")
        
        # Infer signature
        # Assuming X_val is available for signature
        # X_val = ... (load validation data)
        # signature = infer_signature(X_val, model.predict(X_val))
        # mlflow.keras.log_model(model, "model", signature=signature)
        
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        # Register the model in the Model Registry
        registered_model = mlflow.register_model(model_uri, model_name)
        
        return registered_model.uri

