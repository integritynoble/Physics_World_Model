# InverseNet Dataset Card

**Version:** 0.1.0
**License:** CC-BY-4.0
**Date:** 2026-02-09

## Overview

InverseNet is a benchmark dataset for evaluating inverse-problem-aware
calibration and reconstruction methods under operator mismatch.  It spans
three compressive-imaging modalities with labeled mismatch parameters,
enabling four standardised benchmark tasks.

## Modalities

| Modality | Sweep axis           | Values          | Mismatch families          |
|----------|----------------------|-----------------|----------------------------|
| SPC      | compression ratio    | 10%, 25%, 50%   | gain, mask_error           |
| CACTI    | temporal frames      | 4, 8, 16        | mask_shift, temporal_jitter|
| CASSI    | spectral bands       | 8, 16, 28       | disp_step, mask_shift, PSF_blur |

Each modality is further crossed with:
- **Photon levels:** 1e3, 1e4, 1e5 (low / medium / high SNR)
- **Severity:** mild, moderate, severe

## Sample contents

Each sample directory contains:

| File                      | Description                              |
|---------------------------|------------------------------------------|
| `x_gt.npy`               | Ground-truth signal (image / video / HSI)|
| `y.npy`                  | Noisy + mismatched measurement           |
| `y_clean.npy`            | Clean measurement (no noise / mismatch)  |
| `mask.npy` / `masks.npy` | Measurement matrix / temporal masks      |
| `theta.json`             | True operator parameters                 |
| `delta_theta.json`       | Applied mismatch delta                   |
| `y_cal.npy`              | Calibration captures                     |
| `runbundle_manifest.json` | RunBundle v0.3.0 manifest               |

## Generation recipe

```bash
# Full dataset
python -m experiments.inversenet.gen_spc   --out_dir datasets/inversenet_spc
python -m experiments.inversenet.gen_cacti --out_dir datasets/inversenet_cacti
python -m experiments.inversenet.gen_cassi --out_dir datasets/inversenet_cassi

# Quick smoke test
python -m experiments.inversenet.gen_spc   --smoke
python -m experiments.inversenet.gen_cacti --smoke
python -m experiments.inversenet.gen_cassi --smoke

# Run baselines
python -m experiments.inversenet.run_baselines --smoke

# Package for distribution
python -m experiments.inversenet.package --verify
```

## Benchmark tasks

| Task | Name                        | Primary metric           |
|------|-----------------------------|--------------------------|
| T1   | Operator parameter estimation | theta-error RMSE (lower is better) |
| T2   | Mismatch identification     | Accuracy, F1 (higher is better)    |
| T3   | Calibration                 | PSNR gain dB (higher is better)    |
| T4   | Reconstruction under mismatch | PSNR, SSIM, SAM                  |

## Manifest format

Each modality directory contains `manifest.jsonl` with one JSON object per
line.  See `experiments/inversenet/manifest_schema.py` for the Pydantic
schema.

## Statistics

- SPC:   3 CR x 3 photon x 2 families x 3 severities = **54 samples**
- CACTI: 3 frames x 3 photon x 2 families x 3 severities = **54 samples**
- CASSI: 3 bands x 3 photon x 3 families x 3 severities = **81 samples**
- **Total: 189 samples**

## Citation

```bibtex
@misc{inversenet2026,
  title   = {InverseNet: A Benchmark for Inverse-Problem-Aware Calibration},
  author  = {Physics World Model Team},
  year    = {2026},
  url     = {https://github.com/spiritai/Physics_World_Model},
  license = {CC-BY-4.0}
}
```

## Changelog

- **0.1.0** (2026-02-09): Initial release with SPC, CACTI, CASSI.
