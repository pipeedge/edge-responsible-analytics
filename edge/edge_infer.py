import tensorflow as tf
import numpy as np
import time
from load_models import load_mobilenet_model, load_t5_model
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def perform_inference(data, data_type, batch_size=16):
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
        
        # Process in batches
        predictions = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                # Tokenize with padding and truncation
                inputs = tokenizer(
                    batch, 
                    return_tensors="tf", 
                    padding=True, 
                    truncation=True,
                    max_length=64  # Input sequence length
                )
                
                # Generate text with optimized settings
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,  # Only set max_new_tokens, not max_length
                    do_sample=False,     # Deterministic generation
                    num_beams=1,         # No beam search for memory efficiency
                    early_stopping=False, # Disable early stopping with num_beams=1
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True       # Enable cache for single sample generation
                )
                
                # Decode predictions
                batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(batch_predictions)
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error during T5 inference batch {i}: {str(e)}")
                raise
                
        return predictions
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
