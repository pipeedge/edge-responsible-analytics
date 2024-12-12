import tensorflow as tf
import numpy as np
import time
from load_models import load_mobilenet_model, load_t5_model
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def perform_inference(data, data_type):
    """
    Perform inference using the appropriate model based on data type.
    """
    if data_type == "chest_xray":
        model = load_mobilenet_model()
        predictions = model.predict(data)
        return predictions
    elif data_type == "mt":
        model, tokenizer = load_t5_model()
        
        # Ensure data is in the correct format (list of strings)
        if isinstance(data, pd.Series):
            data = data.tolist()
        elif isinstance(data, str):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError(f"Unexpected data type for medical transcription: {type(data)}")
        
        # Tokenize and generate predictions
        try:
            input_ids = tokenizer(data, return_tensors="tf", padding=True, truncation=True).input_ids
            outputs = model.generate(input_ids)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return predictions
        except Exception as e:
            logger.error(f"Error during T5 inference: {str(e)}")
            raise
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
