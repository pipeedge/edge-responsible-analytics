import tensorflow as tf
import time
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import numpy as np
from load_models import load_mobilenet_model, load_t5_model
from datasets.chest_xray_processor import process_chest_xray_data
from datasets.mt_processor import process_medical_transcriptions_data


# def prepare_data(data, data_type):
#     if data_type == 'chest_xray':
#         datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
#         train_generator = datagen.flow_from_directory(
#             data,
#             target_size=(224, 224),
#             batch_size=32,
#             class_mode='binary',
#             subset='training'
#         )
#         return train_generator
#     elif data_type == 'mt':
#         _, t5_tokenizer = load_t5_model()
#         inputs = t5_tokenizer(data['inputs'], return_tensors="tf", padding=True, truncation=True)
#         labels = t5_tokenizer(data['labels'], return_tensors="tf", padding=True, truncation=True).input_ids
#         return inputs, labels
#     else:
#         return None

def train_model(data_path, data_type, batch_size=32, epochs=1):
    start_time = time.time()
    
    if data_type == 'chest_xray':
        model = load_mobilenet_model()
        for epoch in range(epochs):
            for X_batch, y_batch, _ in process_chest_xray_data(data_path, batch_size):
                model.fit(X_batch, y_batch, epochs=1, verbose=1)
        model.save("mobilenet_model.h5")
        training_loss = model.history.history['loss'][-1]
        training_accuracy = model.history.history['accuracy'][-1]
    elif data_type == 'mt':
        model, tokenizer = load_t5_model()
        X_train, X_test, y_train, y_test = process_medical_transcriptions_data(data_path)
        inputs = tokenizer(X_train.tolist(), return_tensors="tf", padding=True, truncation=True)
        labels = tokenizer(y_train.tolist(), return_tensors="tf", padding=True, truncation=True).input_ids
        dataset = tf.data.Dataset.from_tensor_slices((inputs.input_ids, labels)).shuffle(buffer_size=1024).batch(batch_size)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(dataset, epochs=epochs)
        
        model.save_pretrained("t5_model")
        training_loss = model.history.history['loss'][-1]
        training_accuracy = None  # T5 model does not provide accuracy in the same way
    else:
        return "Unknown data type"
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "training_loss": training_loss,
        "training_accuracy": training_accuracy,
        "duration": duration
    }
