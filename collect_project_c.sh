#!/bin/bash
PROJECT_DIR="project_C_poisson"
find "$PROJECT_DIR" -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.config" \) -not -path "*/.*" -exec sh -c 'for f; do echo "========================================"; echo "FILE: $f"; echo "========================================"; cat "$f"; echo -e "\n"; done' _ {} + > "${PROJECT_DIR}.txt"
