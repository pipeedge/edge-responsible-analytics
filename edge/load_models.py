import os
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import logging

logger = logging.getLogger(__name__)

def load_mobilenet_model():
    model_path = "mobilenet_model.keras"
    if not os.path.exists(model_path):
        # Build and compile the model with 3-channel input
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save(model_path, save_format='keras')
        print(f"[Loader] New MobileNet model created and saved to {model_path}")
    else:
        # Load the model without compiling to reset the optimizer
        model = tf.keras.models.load_model(model_path, compile=False)
        # Recompile the model with a fresh optimizer
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(f"[Loader] MobileNet model loaded and recompiled from {model_path}")
    return model

def load_t5_model():
    """
    Memory-efficient T5 model loading for IoT devices.
    """
    model_dir = os.path.abspath("../t5_model")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Configure model to use less memory
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    except:
        logger.info("No GPU available, using CPU only")
    
    try:
        if os.path.exists(os.path.join(model_dir, "config.json")):
            # Load with memory-efficient settings
            model = TFT5ForConditionalGeneration.from_pretrained(
                model_dir,
                low_cpu_mem_usage=True,
                use_cache=False  # Disable caching to save memory
            )
            tokenizer = T5Tokenizer.from_pretrained(
                model_dir,
                model_max_length=512  # Limit maximum sequence length
            )
            return model, tokenizer
            
        # Download and save if not found locally
        model = TFT5ForConditionalGeneration.from_pretrained(
            't5-small',
            low_cpu_mem_usage=True,
            use_cache=False
        )
        tokenizer = T5Tokenizer.from_pretrained(
            't5-small',
            model_max_length=512
        )
        
        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading T5 model: {str(e)}")
        raise
