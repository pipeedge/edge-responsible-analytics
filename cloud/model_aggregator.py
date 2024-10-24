import os
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

def aggregate_models(models_dir, save_path='aggregated_model.keras'):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    if not model_files:
        raise FileNotFoundError("No models found for aggregation.")
    
    aggregated_model = None
    num_models = len(model_files)
    
    for idx, model_file in enumerate(model_files):
        model_path = os.path.join(models_dir, model_file)
        model = tf.keras.models.load_model(model_path, compile=False)
        if aggregated_model is None:
            aggregated_model = model
            logger.info(f"Loaded first model for aggregation: {model_file}")
        else:
            # Average the weights
            for agg_layer, layer in zip(aggregated_model.layers, model.layers):
                agg_weights = agg_layer.get_weights()
                layer_weights = layer.get_weights()
                averaged_weights = [(agg + layer_w) / 2 for agg, layer_w in zip(agg_weights, layer_weights)]
                agg_layer.set_weights(averaged_weights)
            logger.info(f"Aggregated model weights with: {model_file}")
    
    # Save the aggregated model
    aggregated_model.save(save_path, save_format='keras')
    logger.info(f"Aggregated model saved to {save_path}")
    
    # Clean up individual models
    for model_file in model_files:
        os.remove(os.path.join(models_dir, model_file))
    
    return save_path

