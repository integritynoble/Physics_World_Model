# Electron Energy Loss Spectroscopy (EELS) — Spec-CNN

**GPU**  *CNN for spectroscopy, 2018*
**Input**: energy-loss spectrum (H × W × E, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/public/`

```python
from algorithm_base.eels.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
