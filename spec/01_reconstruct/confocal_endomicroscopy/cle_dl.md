# Confocal Laser Endomicroscopy (CLE) — CLE-Net (CARE) [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: confocal frame (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_endomicroscopy/public/`

```python
from algorithm_base.confocal_endomicroscopy.solvers import run_solver
x = run_solver('cle_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
