# Use the official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

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

# Copy policy JSON files into /opa/policies/
COPY opa/policies/fairness_thresholds.json  \
     opa/policies/reliability_thresholds.json \
     opa/policies/explainability_thresholds.json \
     opa/policies/privacy_thresholds.json \
     ./

# Copy the 'datasets' directory into the Docker image
COPY dataset/ ./dataset/
# Copy kaggle.json into the Docker image
COPY kaggle_credentials/kaggle.json /app/.kaggle/kaggle.json
# Set permissions for kaggle.json
RUN chmod 600 /app/.kaggle/kaggle.json
# Set the Kaggle configuration directory
ENV KAGGLE_CONFIG_DIR=/app/.kaggle

# # Install OPA
# RUN curl -L -o opa https://openpolicyagent.org/downloads/v0.55.0/opa_linux_amd64_static && \
#     chmod 755 opa && \
#     mv opa /usr/local/bin

# # Create directory for OPA policies
# RUN mkdir -p /app/opa/policies

# Copy policy files into /app/opa/policies/
# COPY opa/policies/*.rego \
#      opa/policies/*.json \
#      /app/opa/policies/

# Expose necessary ports
EXPOSE 8000 5002 8181

CMD ["python3", "aggregator.py"]