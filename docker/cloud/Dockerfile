FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY docker/cloud/requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

# Copy cloud service files
COPY cloud/api_server.py \
     cloud/model_aggregation.py \
     cloud/monitoring_service.py \
     ./

# Copy utils directory for policy evaluation
COPY utils/ ./utils/

# Expose ports for FastAPI and Prometheus
EXPOSE 8080 8000

# Start the API server
CMD ["python3", "api_server.py"] 