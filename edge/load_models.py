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
    Load T5 model and tokenizer, downloading and saving locally if needed.
    """
    model_dir = os.path.abspath("../t5_model")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Try to load from local directory first
        if os.path.exists(os.path.join(model_dir, "config.json")):
            logger.info(f"Loading T5 model from local directory: {model_dir}")
            try:
                model = TFT5ForConditionalGeneration.from_pretrained(model_dir)
                tokenizer = T5Tokenizer.from_pretrained(model_dir)
                logger.info("Successfully loaded model and tokenizer from local directory")
                return model, tokenizer
            except Exception as e:
                logger.warning(f"Failed to load from local directory: {e}")
                # If loading fails, we'll fall through to downloading
        
        # Download and save if not found locally
        logger.info("Downloading T5 model and tokenizer")
        model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Save both model and tokenizer
        logger.info(f"Saving model and tokenizer to {model_dir}")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Verify files were saved
        expected_files = ["config.json", "tokenizer.json", "special_tokens_map.json", "spiece.model"]
        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            logger.warning(f"Some expected files are missing: {missing_files}")
            
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error in load_t5_model: {str(e)}")
        raise RuntimeError(f"Failed to load T5 model and tokenizer: {str(e)}")
