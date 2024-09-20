import tensorflow as tf
import numpy as np
import time
from load_models import load_mobilenet_model, load_t5_model

def preprocess_image(image):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def perform_inference(data, data_type):
    start_time = time.time()
    
    if data_type == 'chest_xray':
        model = load_mobilenet_model()
        print(f"Model input shape: {model.input_shape}")  # Debug statement
        all_predictions = []
        for batch_images, _, _ in data:
            preprocessed_images = preprocess_image(batch_images)
            predictions = model.predict(preprocessed_images)
            all_predictions.extend(predictions)
        prediction = np.mean(all_predictions)  # Average prediction for all images
        # del model  # Unload the model to free up memory
    elif data_type == 'mt':
        model, tokenizer = load_t5_model()
        input_ids = tokenizer(data, return_tensors="tf", padding=True, truncation=True).input_ids
        outputs = model.generate(input_ids)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # del model  # Unload the model to free up memory
    else:
        return "Unknown data type"
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "prediction": prediction,
        "duration": duration
    }
