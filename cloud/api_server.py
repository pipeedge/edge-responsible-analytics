from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from model_aggregation import FederatedModelAggregator
from monitoring_service import ModelMonitoringService
import logging
import base64
import json
import os
import tempfile
import shutil
import tarfile

app = FastAPI()
logger = logging.getLogger(__name__)

# Initialize services
aggregator = FederatedModelAggregator(mlflow_uri="http://mlflow:5000")
monitor = ModelMonitoringService(prometheus_port=8000)

class ModelUpdate(BaseModel):
    edge_server_id: str
    model_params: str  # base64 encoded model
    metrics: Dict[str, float]
    model_type: str

@app.post("/edge/update")
async def receive_edge_update(update: ModelUpdate):
    """
    Receive model updates from edge processing servers
    """
    try:
        # Track the update
        monitor.track_performance(update.metrics)
        
        # Store the update in MLflow
        logger.info(f"Received update from edge server {update.edge_server_id}")
        
        # Trigger aggregation if we have enough updates
        aggregated_model = await aggregator.aggregate_models([update.model_params])
        
        # Evaluate aggregated model
        passed_policies, failed_policies = aggregator.evaluate_aggregated_model(
            model=aggregated_model,
            validation_data=update.metrics.get('validation_data'),
            thresholds=update.metrics.get('thresholds')
        )
        
        if passed_policies:
            try:
                # Encode the aggregated model
                if isinstance(aggregated_model, str):  # If path to saved model
                    if os.path.isdir(aggregated_model):
                        # Handle directory-based models
                        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                            shutil.make_archive(tmp.name[:-7], 'gztar', aggregated_model)
                            with open(tmp.name, 'rb') as f:
                                model_bytes = f.read()
                            os.unlink(tmp.name)
                    else:
                        # Handle single file models
                        with open(aggregated_model, 'rb') as f:
                            model_bytes = f.read()
                else:
                    # Handle model object (needs serialization)
                    model_bytes = aggregator.serialize_model(aggregated_model)
                
                model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                
                return {
                    "status": "success",
                    "model": model_b64,
                    "model_type": update.model_type,
                    "message": "Model aggregated successfully"
                }
                
            except Exception as e:
                logger.exception(f"Error processing global model: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to process global model: {str(e)}"
                }
        else:
            return {
                "status": "failed",
                "failed_policies": failed_policies,
                "message": "Model failed policy validation"
            }
            
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 