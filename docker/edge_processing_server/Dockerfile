# Use Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements_edge_processing.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements_edge_processing.txt

# Copy application code
COPY aggregator.py policy_evaluator.py opa_config.yaml ./

# Command to run the aggregator
CMD ["python3", "aggregator.py"]