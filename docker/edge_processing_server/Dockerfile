# Use the official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies for building h5py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY docker/edge_processing_server/requirements_edge_processing.txt .
RUN pip install --upgrade pip && pip install -r requirements_edge_processing.txt

# Copy application code from the 'edge' directory
COPY edge_processing/aggregator.py \
     edge_processing/policy_evaluator.py \
     edge_processing/opa_config.yaml \
     ./

# Copy the 'datasets' directory into the Docker image
COPY datasets/ ./datasets/

# Expose necessary ports
EXPOSE 8000 5002 8181

# Command to run the aggregator
CMD ["python3", "aggregator.py"]