FROM arm64v8/python:3.8-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY docker/edge_device/requirements_edge_device.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements_edge_device.txt

# Final stage
FROM arm64v8/python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KAGGLE_CONFIG_DIR=/app/.kaggle

# Set work directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-103 && \
    rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
COPY docker/edge_device/requirements_edge_device.txt .
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && \
    pip install --no-cache-dir -r requirements_edge_device.txt && \
    rm -rf /wheels

# Copy application code from the 'edge' directory
COPY edge/ ./edge/
# Copy the 'datasets' directory into the Docker image
COPY dataset/ ./dataset/
COPY kaggle_credentials/ ./kaggle_credentials/
COPY utils/ ./utils/

# Copy kaggle.json into the Docker image
COPY kaggle_credentials/kaggle.json /app/.kaggle/kaggle.json
# Set permissions for kaggle.json
RUN chmod 600 /app/.kaggle/kaggle.json

# Expose necessary ports
EXPOSE 8001

# Set the entrypoint and default command
ENTRYPOINT ["python3", "edge/edge_task_processing.py"]
CMD ["--model_type", "MobileNet", "--data_type", "chest_xray"]