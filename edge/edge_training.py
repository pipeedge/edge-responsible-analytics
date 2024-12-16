import tensorflow as tf
import time
import logging
import gc
import os
from typing import Dict, Any, Tuple
import numpy as np
from load_models import load_mobilenet_model, load_bert_model, load_satellite_model
from datasets.chest_xray_processor import process_chest_xray_data
from datasets.mt_processor import process_medical_transcriptions_data
from datasets.odc_processor import ODCProcessor
from datasets.cxr8_processor import process_cxr8_data
import pandas as pd

logger = logging.getLogger(__name__)

def configure_memory_settings():
    """
    Configure TensorFlow memory settings for IoT devices.
    """
    try:
        # Configure GPU memory growth if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Limit GPU memory for IoT devices
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
                )
        
        # Configure CPU memory settings
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.set_soft_device_placement(True)
        
    except Exception as e:
        logger.warning(f"Error configuring memory settings: {e}")

def train_in_batches(model: tf.keras.Model, 
                    dataset: tf.data.Dataset, 
                    batch_size: int = 16, 
                    epochs: int = 1) -> Dict[str, list]:
    """
    Train model in small batches to manage memory usage.
    """
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        for batch_idx, (X_batch, y_batch) in enumerate(dataset):
            # Clear memory before each batch
            if batch_idx % 5 == 0:
                gc.collect()
                
            # Train on batch
            metrics = model.train_on_batch(X_batch, y_batch)
            history['loss'].append(metrics[0])
            if len(metrics) > 1:
                history['accuracy'].append(metrics[1])
                
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: loss = {metrics[0]:.4f}")
                
    return history

def train_model(data_path: str, 
                model_type: str, 
                batch_size: int = 16, 
                epochs: int = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train a model with memory-efficient settings for IoT devices.
    
    Args:
        data_path: Path to training data
        model_type: Type of model ('MobileNet' or 'tinybert')
        batch_size: Size of training batches
        epochs: Number of training epochs
        
    Returns:
        Tuple containing training metrics and validation data
    """
    configure_memory_settings()
    start_time = time.time()
    
    try:
        if model_type == 'MobileNet':
            # Load and prepare MobileNet model
            model = load_mobilenet_model()
            
            # Determine which dataset to use based on data_path
            if 'cxr8' in data_path.lower():
                train_generator, val_generator = process_cxr8_data(batch_size=batch_size)
            else:
                # Use existing chest_xray dataset
                train_generator = process_chest_xray_data(data_path, batch_size)
                val_path = os.path.join(os.path.dirname(data_path), 'val')
                val_generator = process_chest_xray_data(val_path, batch_size)
            
            # Train model
            history = {'loss': [], 'accuracy': []}
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                # Training
                for X_batch, y_batch, sensitive_batch in train_generator:
                    metrics = model.train_on_batch(X_batch, y_batch)
                    history['loss'].append(metrics[0])
                    history['accuracy'].append(metrics[1])
                    
                    # Clear memory periodically
                    if len(history['loss']) % 5 == 0:
                        gc.collect()
                
            # Collect validation data
            val_data = []
            for X_val, y_val, sensitive_val in val_generator:
                val_data.append((X_val, y_val, sensitive_val))
            
            # Combine validation data
            validation_data = (
                np.concatenate([x[0] for x in val_data]),
                np.concatenate([x[1] for x in val_data]),
                pd.concat([x[2] for x in val_data])
            )
            
        elif model_type == 'tinybert':
            # Load TinyBERT model
            model, tokenizer = load_bert_model()
            
            # Process data
            logger.info("Processing medical transcription data...")
            X_train, X_test, y_train, y_test, sf_train, sf_test = process_medical_transcriptions_data(data_path)
            
            # Prepare data in chunks to manage memory
            def prepare_bert_data(texts, labels, chunk_size=100):
                for i in range(0, len(texts), chunk_size):
                    chunk_texts = texts[i:i + chunk_size]
                    chunk_labels = labels[i:i + chunk_size]
                    
                    inputs = tokenizer(
                        chunk_texts.tolist(),
                        return_tensors="tf",
                        padding=True,
                        truncation=True,
                        max_length=128  # Reduced for IoT devices
                    )
                    
                    yield inputs, chunk_labels
            
            # Create training dataset
            train_dataset = tf.data.Dataset.from_generator(
                lambda: prepare_bert_data(X_train, y_train),
                output_signature=(
                    {
                        'input_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                        'attention_mask': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                        'token_type_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32)
                    },
                    tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            ).batch(batch_size)
            
            # Configure training
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            
            # Train with memory-efficient settings
            history = train_in_batches(model, train_dataset, batch_size, epochs)
            
            # Prepare validation data
            validation_data = (
                tokenizer(X_test.tolist(), 
                         return_tensors="tf", 
                         padding=True, 
                         truncation=True, 
                         max_length=128),
                y_test,
                sf_test
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate metrics
        training_metrics = {
            "loss": float(np.mean(history['loss'])),
            "accuracy": float(np.mean(history.get('accuracy', [0]))),
            "duration": time.time() - start_time
        }
        
        # Save the trained model
        save_path = f"{model_type.lower()}_model"
        if model_type == 'MobileNet':
            model.save(f"{save_path}.keras")
        else:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        
        logger.info(f"Training completed. Metrics: {training_metrics}")
        return training_metrics, validation_data
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Clean up memory
        gc.collect()

def train_satellite_model(data_path: str,
                         model_type: str,
                         region: dict,
                         time_range: Tuple[str, str],
                         batch_size: int = 16,
                         epochs: int = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train model on satellite data.
    """
    processor = ODCProcessor()
    model = load_satellite_model(model_type)
    
    # Process data in chunks
    train_data = processor.process_satellite_data(
        product=data_path,
        time_range=time_range,
        region=region,
        batch_size=batch_size
    )
    
    # Train model
    history = {'loss': [], 'accuracy': [], 'iou': []}
    
    for epoch in range(epochs):
        for X_batch, y_batch, metadata in train_data:
            metrics = model.train_on_batch(X_batch, y_batch)
            history['loss'].append(metrics[0])
            history['accuracy'].append(metrics[1])
            history['iou'].append(metrics[2])
    
    return {
        'metrics': {
            'loss': float(np.mean(history['loss'])),
            'accuracy': float(np.mean(history['accuracy'])),
            'iou': float(np.mean(history['iou']))
        }
    }
