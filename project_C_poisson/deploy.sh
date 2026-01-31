#!/bin/bash
set -e

echo "=== HH Reservoir Experiment Deployment Script (v2 FIX) ==="
echo "Using Docker Compose V2 for stability."
echo ""

# Calculate resource limits
TOTAL_CPUS=$(nproc)
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$(echo "scale=2; $TOTAL_RAM_KB / 1024 / 1024" | bc)

CPU_LIMIT=$(echo "scale=2; $TOTAL_CPUS * 0.8" | bc)
RAM_LIMIT=$(echo "scale=2; $TOTAL_RAM_GB * 0.8" | bc)

echo "Detected Resources:"
echo "  Total CPUs: $TOTAL_CPUS"
echo "  Total RAM: ${TOTAL_RAM_GB}GB"
echo ""
echo "Docker Limits (80%):"
echo "  CPU Limit: ${CPU_LIMIT} cores"
echo "  RAM Limit: ${RAM_LIMIT}GB"
echo ""

# Configuration generator function
generate_compose() {
    local cmd=$1
    cat > docker-compose.yml <<EOF
version: '3.8'

services:
  hh_experiment:
    build: .
    container_name: hh_reservoir_experiment
    volumes:
      - ./results:/app/results
      - ./cache:/app/cache
      - ./configs:/app/configs
    deploy:
      resources:
        limits:
          cpus: '${CPU_LIMIT}'
          memory: '${RAM_LIMIT}g'
    environment:
      - PYTHONUNBUFFERED=1
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
      - MKL_NUM_THREADS=1
    command: ${cmd}
EOF
}

echo "Step 1: Building Docker image..."
generate_compose "python3 src/run_experiment.py --config configs/production_config.yaml"
docker compose build --no-cache

echo ""
echo "Step 2: Running VALIDATION (8 trials)..."
# Clean shutdown of any previous runs to avoid KeyError: 'ContainerConfig'
docker compose down || true
generate_compose "python3 src/run_experiment.py --config configs/validation_config.yaml"
docker compose up --build --abort-on-container-exit

echo ""
echo "✓ VALIDATION PASSED"

echo ""
echo "Starting PRODUCTION sweep automatically..."
# Clean shutdown again before production
docker compose down || true
generate_compose "python3 src/run_experiment.py --config configs/production_config.yaml"

# Store validation results
mkdir -p results_validation
mv results/*.parquet results_validation/ 2>/dev/null || true

# Run production in foreground (nohup handles the backgrounding)
docker compose up --build --abort-on-container-exit

echo ""
echo "Step 3: Running ESN Baseline Sweep..."
generate_compose "python3 src/run_esn.py --config configs/production_config.yaml"
docker compose up --build --abort-on-container-exit

echo ""
echo "✓ ALL EXPERIMENTS COMPLETED"
echo "Results saved to: ./results/"
