# Magnetic Resonance Imaging (MRI) — ReconFormer

**GPU**  **PSNR**: ~40.1 dB  *Guo et al., IEEE TMI 2024 — 40.1 dB on fastMRI knee 4x*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('reconformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
