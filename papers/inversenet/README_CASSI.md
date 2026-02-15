# InverseNet ECCV: CASSI Validation

This directory contains the comprehensive CASSI (Coded Aperture Snapshot Spectral Imaging) validation framework for the InverseNet ECCV paper.

## Overview

The CASSI validation benchmark compares 4 reconstruction methods across 3 scenarios on 10 KAIST hyperspectral scenes, evaluating reconstruction quality under realistic operator mismatch without calibration correction.

**Key Components:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), IV (Truth Forward Model)
- **4 Methods:** GAP-TV, HDNet, MST-S, MST-L
- **10 Scenes:** 256×256×28 KAIST hyperspectral dataset
- **120 Total Reconstructions:** (10 scenes × 3 scenarios × 4 methods)

## Quick Start

### 1. Prerequisites

```bash
# Ensure PWM core is installed
cd /home/spiritai/PWM/test2/Physics_World_Model
pip install -e packages/pwm_core

# Check dataset availability
ls /home/spiritai/MST-main/datasets/TSA_simu_data/scene01.mat
ls /home/spiritai/MST-main/datasets/TSA_real_data/mask.mat
```

### 2. Run Validation

```bash
cd /home/spiritai/PWM/test2/Physics_World_Model/papers/inversenet

# Run validation (requires GPU for best performance)
python scripts/validate_cassi_inversenet.py --device cuda:0

# Expected output:
# - results/cassi_validation_results.json (detailed per-scene results)
# - results/cassi_summary.json (aggregated statistics)
```

### 3. Generate Figures

```bash
# Create visualization plots
python scripts/generate_cassi_figures.py

# Output files:
# - figures/cassi/scenario_comparison.png
# - figures/cassi/method_comparison_heatmap.png
# - figures/cassi/gap_comparison.png
# - figures/cassi/psnr_distribution.png
# - tables/cassi_results_table.csv
```

## File Structure

```
papers/inversenet/
├── README.md                          # Main manuscript skeleton
├── README_CASSI.md                    # This file
├── cassi_plan_inversenet.md           # Detailed validation plan (1000+ lines)
├── scripts/
│   ├── validate_cassi_inversenet.py   # Main validation script (650 lines)
│   ├── generate_cassi_figures.py      # Visualization generation (380 lines)
│   └── run_all.sh                     # Shell script to run all steps
├── results/
│   ├── cassi_validation_results.json  # Per-scene detailed results
│   └── cassi_summary.json             # Aggregated statistics
├── figures/
│   └── cassi/
│       ├── scenario_comparison.png
│       ├── method_comparison_heatmap.png
│       ├── gap_comparison.png
│       └── psnr_distribution.png
└── tables/
    └── cassi_results_table.csv        # LaTeX-ready results table
```

## Validation Plan Details

**See `cassi_plan_inversenet.md` for comprehensive documentation:**

### Scenario I: Ideal (Oracle)
- **Purpose:** Theoretical upper bound for perfect measurements
- **Expected PSNR:** GAP-TV ~32 dB, HDNet ~35 dB, MST-S ~34 dB, MST-L ~36 dB

### Scenario II: Assumed/Baseline (Uncorrected Mismatch)
- **Purpose:** Realistic baseline showing degradation from uncorrected operator mismatch
- **Mismatch Injected:** dx=0.5 px, dy=0.3 px, θ=0.1°
- **Expected Degradation:** ~3-5 dB from Scenario I

### Scenario IV: Truth Forward Model (Oracle Operator)
- **Purpose:** Upper bound for corrupted measurements when true mismatch is known
- **Expected Recovery:** ~1-2 dB from Scenario II
- **Gap IV→I:** Residual (~1-3 dB, noise + solver limitation)

## Reconstruction Methods

### 1. GAP-TV (Classical Baseline)
- Iterative algebraic reconstruction
- 50 iterations, TV weight 0.05
- ~32 dB on clean measurements
- GPU time: ~10 sec/scene

### 2. HDNet (Deep Unrolled)
- Dual-domain unrolled network
- 2.37M parameters, pre-trained on KAIST
- ~35 dB on clean measurements
- GPU time: ~15 sec/scene

### 3. MST-S (Transformer Small)
- Multi-stage Transformer (compact)
- 0.9M parameters, pre-trained on KAIST
- ~34 dB on clean measurements
- GPU time: ~20 sec/scene

### 4. MST-L (Transformer Large)
- Multi-stage Transformer (large capacity)
- 2.0M parameters, pre-trained on KAIST
- ~36 dB on clean measurements (state-of-the-art)
- GPU time: ~30 sec/scene

## Expected Results

### PSNR Hierarchy (per method)

```
Scenario I (Ideal) > Scenario IV (Oracle) > Scenario II (Baseline)

Example (MST-L):
  I:  36.0 dB (perfect knowledge)
  IV: 33.6 dB (oracle with mismatch)
  II: 32.3 dB (uncorrected mismatch)

Gaps:
  Gap I→II:  3.7 dB (mismatch impact)
  Gap II→IV: 1.3 dB (recovery with oracle)
  Gap IV→I:  2.4 dB (residual noise/solver)
```

### Method Ranking (all scenarios)

1. **MST-L:** Best quality in all scenarios
2. **HDNet:** Second best, good robustness
3. **MST-S:** Lightweight alternative, ~1 dB behind MST-L
4. **GAP-TV:** Classical baseline, ~4 dB behind deep learning

**Key Insight:** Deep learning methods maintain ~3-4 dB advantage even under operator mismatch, showing inherent robustness to misalignment.

## Execution Timeline

| Step | Duration | Command |
|------|----------|---------|
| Setup | 5 min | `pip install -e packages/pwm_core` |
| Validation | 2 hours | `python scripts/validate_cassi_inversenet.py` |
| Visualization | 30 sec | `python scripts/generate_cassi_figures.py` |
| **Total** | **~2.5 hours** | End-to-end pipeline |

**GPU Requirements:**
- NVIDIA CUDA GPU recommended for Transformer methods
- Minimum 8-12 GB VRAM for MST-L
- Estimated throughput: ~1-2 min per scene across all 4 methods

## Output Files Explained

### results/cassi_validation_results.json

Per-scene detailed results structure:

```json
[
  {
    "scene_idx": 1,
    "scenario_i": {
      "gap_tv": {"psnr": 32.1, "ssim": 0.95, "sam": 1.2},
      "hdnet": {"psnr": 35.0, "ssim": 0.97, "sam": 0.8},
      ...
    },
    "scenario_ii": {...},
    "scenario_iv": {...},
    "gaps": {
      "gap_tv": {"gap_i_ii": 3.6, "gap_ii_iv": 1.3, ...},
      ...
    }
  },
  ...
]
```

### results/cassi_summary.json

Aggregated statistics across all 10 scenes:

```json
{
  "num_scenes": 10,
  "scenarios": {
    "scenario_i": {
      "gap_tv": {
        "psnr": {"mean": 32.1, "std": 0.02, "min": 32.0, "max": 32.2},
        "ssim": {"mean": 0.95, "std": 0.01},
        "sam": {"mean": 1.2, "std": 0.1}
      },
      ...
    },
    ...
  },
  "gaps": {...},
  "execution_time": {"total_hours": 2.1, "per_scene_avg_seconds": 756}
}
```

## Verification Checklist

After running validation:

```bash
# 1. Check results exist
ls -lh results/cassi_*.json

# 2. Verify PSNR hierarchy (I > IV > II for all methods)
python -c "import json; s=json.load(open('results/cassi_summary.json')); \
  print('PSNR I > IV > II:', \
    all(s['scenarios']['scenario_i'][m]['psnr']['mean'] > \
        s['scenarios']['scenario_iv'][m]['psnr']['mean'] > \
        s['scenarios']['scenario_ii'][m]['psnr']['mean'] \
      for m in ['gap_tv', 'hdnet', 'mst_s', 'mst_l']))"

# 3. Check method ranking (MST-L > HDNet > MST-S > GAP-TV)
python -c "import json; s=json.load(open('results/cassi_summary.json')); \
  i_psnr = {m: s['scenarios']['scenario_i'][m]['psnr']['mean'] \
    for m in ['gap_tv', 'hdnet', 'mst_s', 'mst_l']}; \
  print('Method ranking (Scenario I):', sorted(i_psnr.items(), key=lambda x: x[1], reverse=True))"

# 4. Check figures generated
ls -l figures/cassi/*.png
```

## Troubleshooting

### Issue: "Dataset not found"
```bash
# Check dataset locations
ls /home/spiritai/MST-main/datasets/TSA_simu_data/
ls /home/spiritai/MST-main/datasets/TSA_real_data/

# Download if needed (see PWM documentation)
```

### Issue: "ImportError: No module named pwm_core"
```bash
cd /home/spiritai/PWM/test2/Physics_World_Model
pip install -e packages/pwm_core
```

### Issue: "CUDA out of memory"
```bash
# Use CPU fallback
python scripts/validate_cassi_inversenet.py --device cpu

# Or use GPU with reduced batch size (in reconstruction methods)
```

### Issue: Reconstruction methods return zeros/random values
```bash
# This indicates the methods are not installed properly
# Check PWM core installation:
python -c "from pwm_core.recon.gap_tv import gap_tv_cassi; print('✓ GAP-TV available')"
python -c "from pwm_core.recon.mst import create_mst; print('✓ MST available')"
python -c "from pwm_core.recon.hdnet import hdnet_recon_cassi; print('✓ HDNet available')"
```

## References

- **CASSI Calibration Plan:** `docs/cassi_plan.md`
- **CASSI Forward Model:** `packages/pwm_core/pwm_core/calibration/cassi_upwmi_alg12.py`
- **Reconstruction Methods:**
  - GAP-TV: `packages/pwm_core/pwm_core/recon/gap_tv.py`
  - HDNet: `packages/pwm_core/pwm_core/recon/hdnet.py`
  - MST: `packages/pwm_core/pwm_core/recon/mst.py`
- **KAIST Dataset:** `/home/spiritai/MST-main/datasets/`

## Citation

For papers using this benchmark, please cite:

```bibtex
@article{inversenet2026,
  title={InverseNet: Benchmarking Operator Mismatch Calibration in Computational Imaging},
  author={Physics World Model Team},
  journal={ECCV},
  year={2026}
}
```

## Contact

For issues or questions about the CASSI validation benchmark, refer to the main PWM project documentation or contact the Physics World Model team.

---

**Last Updated:** 2026-02-15
**Plan Document:** `cassi_plan_inversenet.md`
**Implementation Status:** Complete (Phase 1-3)
