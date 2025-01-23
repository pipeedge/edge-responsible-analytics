import base64
import json
import math
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 100 * 1024  # 100KB chunks
MQTT_QOS = 1  # Use QoS 1 for reliable delivery
TRANSFER_TIMEOUT = 300  # 5 minutes timeout for transfers

class ChunkedMQTTTransfer:
    def __init__(self, mqtt_client, device_id: str):
        self.mqtt_client = mqtt_client
        self.device_id = device_id
        self.received_chunks: Dict[str, Dict[int, str]] = {}
        self.total_chunks: Dict[str, int] = {}
        self.transfer_metadata: Dict[str, Dict[str, Any]] = {}
        self.transfer_start_times: Dict[str, float] = {}  # Track when transfers start
        
    def send_file_in_chunks(self, file_data: bytes, topic: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a large file in chunks over MQTT.
        
        Args:
            file_data: The file data to send
            topic: The MQTT topic to publish to
            metadata: Additional metadata to include with the file
        """
        try:
            # Calculate total chunks needed
            total_chunks = math.ceil(len(file_data) / CHUNK_SIZE)
            transfer_id = f"{self.device_id}_{hash(file_data)}"
            
            logger.info(f"Starting transfer {transfer_id} with {total_chunks} chunks")
            
            # Send transfer start message with metadata
            start_payload = {
                'type': 'transfer_start',
                'transfer_id': transfer_id,
                'total_chunks': total_chunks,
                'device_id': self.device_id,
                'metadata': metadata or {}
            }
            self.mqtt_client.publish(f"{topic}/control", json.dumps(start_payload), qos=MQTT_QOS)
            
            # Send chunks with progress logging
            for chunk_num in range(total_chunks):
                start_idx = chunk_num * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, len(file_data))
                chunk_data = file_data[start_idx:end_idx]
                
                chunk_payload = {
                    'type': 'chunk',
                    'transfer_id': transfer_id,
                    'chunk_num': chunk_num,
                    'total_chunks': total_chunks,
                    'data': base64.b64encode(chunk_data).decode('utf-8'),
                    'device_id': self.device_id
                }
                
                self.mqtt_client.publish(f"{topic}/chunks", json.dumps(chunk_payload), qos=MQTT_QOS)
                
                if chunk_num % 10 == 0:  # Log progress every 10 chunks
                    logger.info(f"Sent chunk {chunk_num}/{total_chunks} for transfer {transfer_id}")
            
            logger.info(f"Completed sending all chunks for transfer {transfer_id}")
            
            # Send transfer complete message
            complete_payload = {
                'type': 'transfer_complete',
                'transfer_id': transfer_id,
                'device_id': self.device_id
            }
            self.mqtt_client.publish(f"{topic}/control", json.dumps(complete_payload), qos=MQTT_QOS)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending file in chunks: {e}")
            return False
            
    def handle_chunk_message(self, msg) -> Optional[Dict[str, Any]]:
        """
        Handle incoming chunk messages and reassemble the file when complete.
        Returns the complete file data and metadata when all chunks are received.
        """
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            msg_type = payload.get('type')
            transfer_id = payload.get('transfer_id')
            device_id = payload.get('device_id')
            
            # Check for timeout on existing transfers
            current_time = time.time()
            for tid in list(self.transfer_start_times.keys()):
                if current_time - self.transfer_start_times[tid] > TRANSFER_TIMEOUT:
                    logger.warning(f"Transfer {tid} timed out after {TRANSFER_TIMEOUT} seconds")
                    self._cleanup_transfer(tid)
            
            if msg_type == 'transfer_start':
                self.received_chunks[transfer_id] = {}
                self.total_chunks[transfer_id] = payload['total_chunks']
                self.transfer_metadata[transfer_id] = payload.get('metadata', {})
                self.transfer_start_times[transfer_id] = time.time()
                logger.info(f"Started new transfer {transfer_id} with {payload['total_chunks']} expected chunks")
                return None
                
            elif msg_type == 'chunk':
                if transfer_id not in self.received_chunks:
                    logger.warning(f"Received chunk for unknown transfer {transfer_id}")
                    return None
                    
                chunk_num = payload['chunk_num']
                chunk_data = base64.b64decode(payload['data'])
                self.received_chunks[transfer_id][chunk_num] = chunk_data
                
                # Log progress every 10 chunks
                if chunk_num % 10 == 0:
                    logger.info(f"Received chunk {chunk_num}/{self.total_chunks[transfer_id]} for transfer {transfer_id}")
                
                # Check if we have all chunks
                if len(self.received_chunks[transfer_id]) == self.total_chunks[transfer_id]:
                    logger.info(f"Received all chunks for transfer {transfer_id}, reassembling...")
                    
                    # Reassemble the file
                    chunks = [self.received_chunks[transfer_id][i] 
                            for i in range(self.total_chunks[transfer_id])]
                    complete_data = b''.join(chunks)
                    metadata = self.transfer_metadata.get(transfer_id, {})
                    
                    # Clean up
                    self._cleanup_transfer(transfer_id)
                    
                    logger.info(f"Successfully reassembled transfer {transfer_id}")
                    return {
                        'data': complete_data,
                        'metadata': metadata,
                        'device_id': device_id
                    }
            
            elif msg_type == 'transfer_complete':
                logger.info(f"Received transfer complete for {transfer_id}")
                self._cleanup_transfer(transfer_id)
                    
            return None
            
        except Exception as e:
            logger.error(f"Error handling chunk message: {e}")
            logger.exception("Detailed error:")
            return None
            
    def _cleanup_transfer(self, transfer_id: str):
        """Helper method to clean up transfer data"""
        if transfer_id in self.received_chunks:
            del self.received_chunks[transfer_id]
        if transfer_id in self.total_chunks:
            del self.total_chunks[transfer_id]
        if transfer_id in self.transfer_metadata:
            del self.transfer_metadata[transfer_id]
        if transfer_id in self.transfer_start_times:
            del self.transfer_start_times[transfer_id] 