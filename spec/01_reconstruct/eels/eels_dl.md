# Electron Energy Loss Spectroscopy (EELS) — EELS-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: energy-loss spectrum (H × W × E, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/public/`

```python
from algorithm_base.eels.solvers import run_solver
x = run_solver('eels_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
