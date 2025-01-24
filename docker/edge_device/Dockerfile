# Stage 1: Dependencies
FROM arm64v8/python:3.8-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory for dependencies
WORKDIR /deps

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements file
COPY docker/edge_device/requirements_edge_device.txt .

# Create virtual environment and install dependencies if not exists
RUN if [ ! -d "/opt/venv" ]; then \
        python -m venv /opt/venv && \
        /opt/venv/bin/pip install --upgrade pip && \
        /opt/venv/bin/pip install -r requirements_edge_device.txt; \
    fi

# Stage 2: Final image
FROM arm64v8/python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy application code (excluding dataset)
COPY edge/ ./edge/
COPY utils/ ./utils/
COPY kaggle_credentials/ ./kaggle_credentials/

RUN chmod 600 /app/kaggle_credentials/kaggle.json
ENV KAGGLE_CONFIG_DIR=/app/kaggle_credentials

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Expose necessary ports
EXPOSE 8001

# Change the CMD to ENTRYPOINT and CMD
ENTRYPOINT ["python3", "edge/edge_task_processing.py"]
CMD ["--model_type", "MobileNet", "--data_type", "chest_xray"]