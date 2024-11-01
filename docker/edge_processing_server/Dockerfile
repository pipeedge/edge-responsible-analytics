# Use the official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY docker/edge_processing_server/requirements_edge_processing.txt .
RUN pip install --upgrade pip && pip install -r requirements_edge_processing.txt

# Copy application code from the 'edge' directory
COPY edge/aggregator.py \
     edge/policy_evaluator.py \
     edge/opa_config.yaml \
     ./

# Expose necessary ports
EXPOSE 8000 5002 8181

# Command to run the aggregator
CMD ["python3", "aggregator.py"]