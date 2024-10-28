import os
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

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
    model_dir = "../t5_model"
    if not os.path.exists(model_dir):
        # Load the model from Hugging Face and save it locally
        model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
        model.save_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        tokenizer.save_pretrained(model_dir)
        print(f"[Loader] T5 model and tokenizer saved to {model_dir}")
    else:
        # Load the model from the local directory
        model = TFT5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        print(f"[Loader] T5 model and tokenizer loaded from {model_dir}")
    return model, tokenizer
