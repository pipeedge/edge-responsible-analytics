import os
import json
import base64
import time
import threading
import sys
from collections import defaultdict

import paho.mqtt.client as mqtt
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', '10.210.32.158')  # 10.12.93.246
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC_UPLOAD = os.getenv('MQTT_TOPIC_UPLOAD', 'models/upload')
MQTT_TOPIC_AGGREGATED = os.getenv('MQTT_TOPIC_AGGREGATED', 'models/aggregated')

# Number of end devices expected
EXPECTED_DEVICES = int(os.getenv('EXPECTED_DEVICES', 1))  # Set this to the number of your end devices

# Initialize MQTT Client
client = mqtt.Client(client_id='aggregator')

# Dictionary to store received models
received_models = {}
lock = threading.Lock()

# Callback when a message is received
def on_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_UPLOAD:
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            device_id = payload.get('device_id')
            model_type = payload.get('model_type')
            model_b64 = payload.get('model_data')

            if device_id and model_type and model_b64:
                with lock:
                    if device_id not in received_models:
                        logger.info(f"Received {model_type} model from {device_id}")
                        received_models[device_id] = {
                            'model_type': model_type,
                            'model_data': model_b64
                        }
                    else:
                        logger.warning(f"Model from {device_id} already received.")
                
                # Check if all expected models are received
                with lock:
                    if len(received_models) >= EXPECTED_DEVICES:
                        logger.info("All models received. Starting aggregation.")
                        aggregate_and_publish_models()
                        # Clear received_models for next round
                        received_models.clear()
            else:
                logger.error("Received message with missing fields.")
        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON payload.")
        except Exception as e:
            logger.exception(f"Unexpected error in on_message: {e}")

def aggregate_and_publish_models():
    loaded_models = []
    device_ids = list(received_models.keys())
    logger.info(f"[Aggregator] Aggregating models from devices: {device_ids}")
    
    # Deserialize models from Base64
    for device_id in device_ids:
        model_b64 = received_models[device_id]['model_data']
        model_bytes = base64.b64decode(model_b64)
        model_path = f"temp_{device_id}.h5"
        with open(model_path, 'wb') as f:
            f.write(model_bytes)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded_models.append(model)
            logger.info(f"[Aggregator] Loaded model from {device_id}")
        except Exception as e:
            logger.error(f"[Aggregator] Failed to load model from {device_id}: {e}")
        finally:
            os.remove(model_path)  # Clean up temporary file

    if not loaded_models:
        logger.warning("[Aggregator] No valid models loaded for aggregation.")
        return

    # Initialize aggregated weights with zeros
    aggregated_weights = []
    for layer in loaded_models[0].layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            weight_shapes = [w.shape for w in layer.get_weights()]
            aggregated_weights.append([tf.zeros_like(w) for w in layer.get_weights()])
        else:
            aggregated_weights.append(None)  # Layers without weights

    # Sum weights from all models
    for model in loaded_models:
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                layer_weights = layer.get_weights()
                if aggregated_weights[idx] is not None:
                    aggregated_weights[idx] = [agg_w + w for agg_w, w in zip(aggregated_weights[idx], layer_weights)]
    
    # Average the weights
    num_models = len(loaded_models)
    for idx, layer_weights in enumerate(aggregated_weights):
        if layer_weights is not None:
            aggregated_weights[idx] = [w / num_models for w in layer_weights]
    
    # Create a new aggregated model
    # Assuming all models have the same architecture as the first model
    aggregated_model = tf.keras.models.clone_model(loaded_models[0])
    
    # Set the averaged weights
    for idx, layer in enumerate(aggregated_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights(aggregated_weights[idx])
    
    # Compile the aggregated model
    aggregated_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Save the aggregated model
    aggregated_model_path = 'aggregated_model.h5'
    aggregated_model.save(aggregated_model_path)
    logger.info(f"[Aggregator] Aggregated model saved to {aggregated_model_path}")
    
    # Read and encode the aggregated model
    with open(aggregated_model_path, 'rb') as f:
        aggregated_model_bytes = f.read()
    aggregated_model_b64 = base64.b64encode(aggregated_model_bytes).decode('utf-8')
    
    # Publish the aggregated model
    payload = json.dumps({
        'model_data': aggregated_model_b64,
        'model_type': 'MobileNet'  # Adjust as needed
    })
    client.publish(MQTT_TOPIC_AGGREGATED, payload)
    logger.info(f"[Aggregator] Published aggregated model to {MQTT_TOPIC_AGGREGATED}")

# Set up MQTT callbacks
client.on_message = on_message

# Connect to MQTT Broker
def connect_mqtt():
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.subscribe(MQTT_TOPIC_UPLOAD)
        logger.info(f"[Aggregator] Subscribed to {MQTT_TOPIC_UPLOAD}")
    except Exception as e:
        logger.error(f"[Aggregator] Failed to connect to MQTT Broker: {e}")
        sys.exit(1)

# Start MQTT loop in a separate thread
def mqtt_loop():
    client.loop_forever()

def main():
    connect_mqtt()
    # Start MQTT loop in background
    thread = threading.Thread(target=mqtt_loop)
    thread.daemon = True
    thread.start()

    logger.info("[Aggregator] Running. Waiting for models...")
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logger.info("[Aggregator] Shutting down.")
        client.disconnect()

if __name__ == "__main__":
    main()