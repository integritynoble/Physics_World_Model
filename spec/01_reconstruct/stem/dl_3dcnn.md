# Scanning Transmission Electron Microscopy (STEM) — 3D-CNN

**GPU**  *3D CNN reconstruction, 2018*
**Input**: HAADF image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stem/public/`

```python
from algorithm_base.stem.solvers import run_solver
x = run_solver('dl_3dcnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
