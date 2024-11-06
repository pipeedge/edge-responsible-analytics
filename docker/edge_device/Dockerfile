# Use an ARM-compatible Python base image
FROM arm64v8/python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY docker/edge_device/requirements_edge_device.txt .
RUN pip install --upgrade pip && pip install -r requirements_edge_device.txt

# Copy application code from the 'edge' directory
COPY edge/edge_task_processing.py \
     edge/edge_infer.py \
     edge/edge_training.py \
     edge/load_models.py \
     ./

# Copy the 'datasets' directory into the Docker image
COPY datasets/ ./datasets/

# Expose necessary ports
EXPOSE 8001

# Command to run the edge device script
CMD ["python3", "edge_task_processing.py"]