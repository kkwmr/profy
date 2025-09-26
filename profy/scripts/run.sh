#!/bin/bash
# Profy Main Experiment Runner (modality comparison)
set -euo pipefail

# Setup environment
export EXPERIMENT_ID="experiment_$(date +%Y%m%d_%H%M%S)"
# Ensure outputs under profy/results as requested
export RESULTS_DIR="profy/results/${EXPERIMENT_ID}"

# Create directory structure
mkdir -p "${RESULTS_DIR}"{,/models,/logs,/figures}

echo "============================================"
echo "PROFY EXPERIMENT - REAL DATA (6,476 samples)"
echo "============================================"
echo "Experiment ID: ${EXPERIMENT_ID}"
echo "Results Directory: ${RESULTS_DIR}"
echo "============================================"

# Change to project root so that 'profy' package is resolvable
cd /home/kazuki/Projects/Profy

# Persist env for Python
echo "export RESULTS_DIR='${RESULTS_DIR}'" > profy/.experiment_env
echo "export EXPERIMENT_ID='${EXPERIMENT_ID}'" >> profy/.experiment_env

echo "Starting experiment (multimodal / sensor-only / audio-only)..."

# Forward optional flags (e.g., --debug)
python3 -m profy.src.training.run_experiments "$@"

echo ""
echo "=========================================="
echo "✅ EXPERIMENT COMPLETED"
echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}"
echo "Key outputs:"
echo "  - complete_results.json (F1/Acc per modality)"
echo "  - figures/ (comparison plots, if generated)"
echo "  - models/ (saved checkpoints, if enabled)"
echo "  - logs/ (runtime logs)"
echo "=========================================="
