FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for results and cache
RUN mkdir -p /app/results /app/cache

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["python3", "src/run_experiment.py", "--config", "configs/production_config.yaml"]
