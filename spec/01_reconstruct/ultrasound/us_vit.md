# Ultrasound B-mode Imaging — US-ViT (PnP-DRS DRUNet)

**GPU**  *Song et al. 2023, IEEE TMI*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('us_vit', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
