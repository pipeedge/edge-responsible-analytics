# Use Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements_edge_device.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements_edge_device.txt

# Copy application code
COPY edge_task_processing.py edge_infer.py edge_training.py load_models.py ./

# Command to run the edge device script
CMD ["python3", "edge_task_processing.py"]