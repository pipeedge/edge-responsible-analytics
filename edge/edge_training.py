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
    
    val_generator = datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # Create the model
    model = load_mobilenet_model()
    
    # Configure training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        # Training steps
        for images, labels in train_generator:
            # Call train_step with all required arguments
            loss = train_step(
                model=model,
                optimizer=optimizer,
                inputs=images,
                targets=labels,
                loss_fn=loss_fn
            )
            
            predictions = model(images, training=False)
            train_acc_metric.update_state(labels, predictions)
            
            total_loss += loss
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
            
            if steps * batch_size >= samples_per_class * 2:
                break
        
        # Validation steps
        val_loss = 0
        val_steps = 0
        for val_images, val_labels in val_generator:
            val_predictions = model(val_images, training=False)
            val_loss += loss_fn(val_labels, val_predictions)
            val_acc_metric.update_state(val_labels, val_predictions)
            val_steps += 1
            
            if val_steps * batch_size >= samples_per_class:
                break
        
        # Calculate epoch metrics
        epoch_loss = total_loss / steps
        epoch_accuracy = train_acc_metric.result().numpy()
        epoch_val_loss = val_loss / val_steps
        epoch_val_accuracy = val_acc_metric.result().numpy()
        
        # Update best metrics
        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
        
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}, "
              f"val_loss={epoch_val_loss:.4f}, val_accuracy={epoch_val_accuracy:.4f}")
        
        # Reset metrics
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()
        gc.collect()
    
    # Save the model
    model.save('mobilenet_model.keras')
    
    return {
        'loss': float(epoch_loss),
        'accuracy': float(epoch_accuracy),
        'val_loss': float(epoch_val_loss),
        'val_accuracy': float(epoch_val_accuracy),
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss)
    }

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
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Training loop with reduced retracing
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        for batch_inputs, batch_targets in dataset:
            loss = train_step(
                model=model,
                optimizer=optimizer,
                inputs=batch_inputs,
                targets=batch_targets,
                loss_fn=loss_fn
            )
            
            # Update metrics
            predictions = model(batch_inputs, training=False)
            train_acc_metric.update_state(batch_targets, predictions)
            
            total_loss += loss
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / steps
        epoch_accuracy = train_acc_metric.result().numpy()
        
        # Update best metrics
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}")
        
        # Reset metrics
        train_acc_metric.reset_state()
        gc.collect()
    
    # Save the model
    model.save_pretrained("t5_small")
    tokenizer.save_pretrained("t5_small")
    
    return {
        'loss': float(epoch_loss),
        'accuracy': float(epoch_accuracy),
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss)
    }

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
    
    # Clean labels and convert to indices
    y_train = [label.strip() for label in y_train]  # Remove leading/trailing whitespace
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(medical_specialties)))}
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
    
    # Configure optimizer and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        for batch_inputs, batch_labels in dataset:
            loss = train_step(
                model=model,
                optimizer=optimizer,
                inputs=batch_inputs,
                targets=batch_labels,
                loss_fn=loss_fn
            )
            
            # Update metrics
            predictions = model(batch_inputs, training=False)
            train_acc_metric.update_state(batch_labels, predictions.logits)
            
            total_loss += loss
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / steps
        epoch_accuracy = train_acc_metric.result().numpy()
        
        # Update best metrics
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}")
        
        # Reset metrics
        train_acc_metric.reset_state()
        gc.collect()
    
    # Save the model
    model.save_pretrained("tinybert_model")
    tokenizer.save_pretrained("tinybert_model")
    
    return {
        'loss': float(epoch_loss),
        'accuracy': float(epoch_accuracy),
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss)
    }

if __name__ == "__main__":  
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
