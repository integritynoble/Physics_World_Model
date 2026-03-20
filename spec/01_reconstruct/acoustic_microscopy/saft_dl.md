# Scanning Acoustic Microscopy (SAM) — SAFT-DL [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: RF data (H × W × T, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_microscopy/public/`

```python
from algorithm_base.acoustic_microscopy.solvers import run_solver
x = run_solver('saft_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
