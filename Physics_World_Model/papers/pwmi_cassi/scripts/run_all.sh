#!/bin/bash
# PWMI-CASSI: Complete reproducibility pipeline
# Usage: bash run_all.sh [DEVICE]
#   DEVICE: cuda:0 (default) or cpu

set -e

DEVICE=${1:-cuda:0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "PWMI-CASSI Reproducibility Pipeline"
echo "Device: $DEVICE"
echo "========================================"

cd "$SCRIPT_DIR"

echo ""
echo "[1/5] Main 4-scenario validation (10 scenes x 4 scenarios x 3 methods)..."
python validate_cassi_pwmi.py --device "$DEVICE" --scenes 10

echo ""
echo "[2/5] Sensitivity sweep (7 scales x 2 methods x 10 scenes)..."
python run_sensitivity_sweep.py --device "$DEVICE" --scenes 10

echo ""
echo "[3/5] Ablation study (Alg1 vs Alg1+Alg2, MST-L, 10 scenes)..."
python run_ablation.py --device "$DEVICE" --scenes 10

echo ""
echo "[4/5] Generating figures..."
python generate_figures.py

echo ""
echo "[5/5] Generating tables..."
python generate_tables.py

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results: $(dirname "$SCRIPT_DIR")/results/"
echo "Figures: $(dirname "$SCRIPT_DIR")/figures/"
echo "Tables:  $(dirname "$SCRIPT_DIR")/tables/"
echo "========================================"
