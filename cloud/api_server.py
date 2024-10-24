from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import json
import tensorflow as tf
import os
from model_aggregator import aggregate_models
from mlflow_client import register_model

app = FastAPI()

class ModelPayload(BaseModel):
    device_id: str
    model_type: str
    model_data: str  # Base64 encoded model

@app.post("/upload_model/")
async def upload_model(payload: ModelPayload):
    try:
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        
        # Decode the model
        model_bytes = base64.b64decode(payload.model_data)
        model_path = f"models/{payload.device_id}_{payload.model_type}.keras"
        with open(model_path, 'wb') as f:
            f.write(model_bytes)
        
        # Aggregate models
        aggregated_model_path = aggregate_models("models/")
        
        # Register the aggregated model with MLflow
        model_uri = register_model(aggregated_model_path, "GlobalModel")
        
        return {"status": "success", "model_uri": model_uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
