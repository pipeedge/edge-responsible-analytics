import os
import gc
import tensorflow as tf
import numpy as np
from dataset.chest_xray_processor import process_chest_xray_data
from dataset.mt_processor import process_medical_transcriptions_data
from edge.load_models import load_mobilenet_model, load_t5_model, load_bert_model, medical_specialties

def configure_memory_settings():
    """Configure TensorFlow for memory-efficient training on Raspberry Pi"""
    # Limit TensorFlow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Limit CPU memory usage
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    # Set memory limit (adjust based on your Pi's RAM)
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices('CPU')[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

def train_mobilenet_edge(data_path, epochs=3, samples_per_class=100):
    """Train MobileNet model on edge device with limited data"""
    print("Starting MobileNet training on edge device...")
    
    # Load pre-trained model
    model = load_mobilenet_model()
    
    # Configure for edge training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Process limited dataset
    train_data = []
    train_labels = []
    count_normal = 0
    count_pneumonia = 0
    
    for images, labels, _ in process_chest_xray_data(data_path, batch_size=8):
        for img, label in zip(images, labels):
            if label == 0 and count_normal < samples_per_class:
                train_data.append(img)
                train_labels.append(label)
                count_normal += 1
            elif label == 1 and count_pneumonia < samples_per_class:
                train_data.append(img)
                train_labels.append(label)
                count_pneumonia += 1
                
            if count_normal >= samples_per_class and count_pneumonia >= samples_per_class:
                break
        
        if count_normal >= samples_per_class and count_pneumonia >= samples_per_class:
            break
    
    X_train = np.array(train_data)
    y_train = np.array(train_labels)
    
    # Train with small batches
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    model.save('mobilenet_model.keras')
    return history

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
    
    # Tokenize inputs and targets
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
    
    # Configure optimizer with low learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Train with gradient accumulation
    batch_size = 2
    accumulation_steps = 4
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        steps = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_inputs = {
                k: v[i:i + batch_size] 
                for k, v in inputs.items()
            }
            batch_targets = targets.input_ids[i:i + batch_size]
            
            with tf.GradientTape() as tape:
                outputs = model(batch_inputs, labels=batch_targets)
                loss = outputs.loss / accumulation_steps
            
            gradients = tape.gradient(loss, model.trainable_variables)
            
            if (steps + 1) % accumulation_steps == 0:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
            total_loss += loss.numpy()
            steps += 1
            
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps}")
            
            # Clear memory
            gc.collect()
    
    # Save the model
    model.save_pretrained("t5_small")
    tokenizer.save_pretrained("t5_small")
    return total_loss/steps

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
