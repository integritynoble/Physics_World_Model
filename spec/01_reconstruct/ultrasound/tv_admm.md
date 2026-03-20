# Ultrasound B-mode Imaging — Total Variation ADMM

**CPU**  *Boyd et al. 2011, ADMM; Rudin-Osher-Fatemi 1992 TV*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('tv_admm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
