# Positron Emission Tomography (PET) — PET-DL (U-Net)

**GPU**  *Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9)*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet/public/`

```python
from algorithm_base.pet.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
