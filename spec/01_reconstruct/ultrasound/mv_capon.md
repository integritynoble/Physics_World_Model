# Ultrasound B-mode Imaging — Minimum-Variance Capon Beamformer

**CPU**  *Capon 1969, Proc. IEEE*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('mv_capon', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
