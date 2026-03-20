# Scanning Transmission Electron Microscopy (STEM) — 3D-Transformer

**GPU**  *Transformer for 3D, 2023*
**Input**: HAADF image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stem/public/`

```python
from algorithm_base.stem.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
