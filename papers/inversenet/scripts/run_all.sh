#!/bin/bash
# InverseNet CASSI Validation: Complete Pipeline
# Runs all validation and visualization steps

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: $PROJECT_DIR"

echo ""
echo "======================================================================"
echo "InverseNet CASSI Validation Pipeline"
echo "======================================================================"
echo ""

# Check prerequisites
echo "[1/3] Checking prerequisites..."
if [ ! -f "${PROJECT_DIR}/../../../packages/pwm_core/setup.py" ]; then
    echo "ERROR: PWM core not found"
    exit 1
fi

if [ ! -d "/home/spiritai/MST-main/datasets/TSA_simu_data" ]; then
    echo "WARNING: KAIST dataset not found at expected location"
fi

echo "✓ Prerequisites OK"
echo ""

# Run validation
echo "[2/3] Running CASSI validation..."
echo "Command: python scripts/validate_cassi_inversenet.py --device cuda:0"
cd "${PROJECT_DIR}"
python scripts/validate_cassi_inversenet.py --device cuda:0

if [ ! -f "${PROJECT_DIR}/results/cassi_summary.json" ]; then
    echo "ERROR: Validation failed - no results generated"
    exit 1
fi
echo "✓ Validation complete"
echo ""

# Generate figures
echo "[3/3] Generating visualization figures..."
echo "Command: python scripts/generate_cassi_figures.py"
python scripts/generate_cassi_figures.py

if [ ! -f "${PROJECT_DIR}/figures/cassi/scenario_comparison.png" ]; then
    echo "ERROR: Figure generation failed"
    exit 1
fi
echo "✓ Figure generation complete"
echo ""

# Summary
echo "======================================================================"
echo "Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  Results:"
echo "    - ${PROJECT_DIR}/results/cassi_validation_results.json"
echo "    - ${PROJECT_DIR}/results/cassi_summary.json"
echo ""
echo "  Figures:"
ls -1 "${PROJECT_DIR}/figures/cassi/"*.png 2>/dev/null | sed 's/^/    - /'
echo ""
echo "  Tables:"
ls -1 "${PROJECT_DIR}/tables/"*.csv 2>/dev/null | sed 's/^/    - /'
echo ""
echo "Next steps:"
echo "  1. Review results: cat ${PROJECT_DIR}/results/cassi_summary.json | python -m json.tool"
echo "  2. View figures: open ${PROJECT_DIR}/figures/cassi/"
echo "  3. Generate paper figures: python papers/inversenet/scripts/generate_cassi_figures.py"
echo ""
