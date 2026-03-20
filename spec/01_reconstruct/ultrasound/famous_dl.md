# Ultrasound B-mode Imaging — US-UNet (PnP-PGD DRUNet)

**GPU**  *Perdios et al. 2017, IEEE IUS*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
