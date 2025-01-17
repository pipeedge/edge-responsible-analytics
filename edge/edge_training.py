import os
import gc
import tensorflow as tf
import numpy as np
from dataset.chest_xray_processor import process_chest_xray_data
from dataset.mt_processor import process_medical_transcriptions_data
from edge.load_models import load_mobilenet_model, load_t5_model, load_bert_model, medical_specialties

# Define the training step function outside any loops
@tf.function(reduce_retracing=True)
def train_step(model, optimizer, inputs, targets, loss_fn):
    """
    Single training step function defined outside the loop
    with reduce_retracing enabled
    """
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_mobilenet_edge(data_path, epochs=2, samples_per_class=50):
    """
    Train MobileNet with optimized memory usage and reduced retracing
    """
    # Create data generator with fixed batch size and shapes
    batch_size = 8
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Use flow_from_directory with fixed image size
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Create the model
    model = load_mobilenet_model()
    
    # Define the training step
    @tf.function(reduce_retracing=True)
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Configure training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        for images, labels in train_generator:
            loss = train_step(images, labels)
            total_loss += loss
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
            
            if steps * batch_size >= samples_per_class * 2:  # For binary classification
                break
        
        gc.collect()
    
    return total_loss / steps

def train_t5_edge(data_path, epochs=5, max_samples=200):
    print("Starting T5 training on edge device...")
    
    # Load model and tokenizer
    model, tokenizer = load_t5_model()
    
    # Get limited dataset
    X_train, _, y_train, _, _, _ = process_medical_transcriptions_data(
        data_path, batch_size=4
    )
    
    # Limit dataset size
    X_train = X_train[:max_samples]
    y_train = y_train[:max_samples]
    
    # Pre-tokenize all data at once
    inputs = tokenizer(
        list(X_train),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="tf"
    )
    
    targets = tokenizer(
        list(y_train),
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="tf"
    )
    
    # Convert to tf.data.Dataset with fixed batch size
    batch_size = 2
    dataset = tf.data.Dataset.from_tensor_slices((
        {k: v for k, v in inputs.items()},
        targets.input_ids
    )).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    # Configure optimizer with low learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Training loop with reduced retracing
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        for batch_inputs, batch_targets in dataset:
            # Use the pre-defined training step
            loss = train_step(model, optimizer, batch_inputs, batch_targets, loss_fn)
            total_loss += loss
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
        
        # Clear memory after each epoch
        gc.collect()
        
    return total_loss / steps

def train_bert_edge(data_path, epochs=5, max_samples=300):
    print("Starting TinyBERT training on edge device...")
    
    # Load model and tokenizer
    model, tokenizer = load_bert_model()
    
    # Get limited dataset
    X_train, _, y_train, _, _, _ = process_medical_transcriptions_data(
        data_path, batch_size=4
    )
    
    # Limit dataset size
    X_train = X_train[:max_samples]
    y_train = y_train[:max_samples]
    
    # Convert labels to indices
    label_to_id = {label: idx for idx, label in enumerate(medical_specialties)}
    y_train = [label_to_id[label] for label in y_train]
    
    # Tokenize inputs
    inputs = tokenizer(
        list(X_train),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="tf"
    )
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(inputs),
        tf.constant(y_train)
    )).shuffle(100).batch(4)
    
    # Configure optimizer with low learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Train with manual loop for better memory control
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        for batch in dataset:
            batch_inputs, batch_labels = batch
            
            with tf.GradientTape() as tape:
                outputs = model(batch_inputs, training=True)
                loss = loss_fn(batch_labels, outputs.logits)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            total_loss += loss.numpy()
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
            
            # Clear memory
            gc.collect()
    
    # Save the model
    model.save_pretrained("tinybert_model")
    tokenizer.save_pretrained("tinybert_model")
    return total_loss/steps

if __name__ == "__main__":
    # Configure memory settings
    configure_memory_settings()
    
    # Train MobileNet
    chest_xray_path = "dataset/chest_xray/train"
    history_mobilenet = train_mobilenet_edge(
        chest_xray_path,
        epochs=2,
        samples_per_class=50  # 50 samples per class for quick training
    )
    
    # Clear memory before next training
    gc.collect()
    
    # Train T5
    mt_path = "dataset/mt"
    loss_t5 = train_t5_edge(
        mt_path,
        epochs=1,
        max_samples=100  # Limited samples for T5
    )
    
    # Clear memory before next training
    gc.collect()
    
    # Train TinyBERT
    loss_bert = train_bert_edge(
        mt_path,
        epochs=1,
        max_samples=150  # Limited samples for BERT
    )
