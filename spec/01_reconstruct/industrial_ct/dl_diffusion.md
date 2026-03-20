# Industrial X-ray CT — DiffusionRecon

**GPU**  *Song et al., 2024*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/public/`

```python
from algorithm_base.industrial_ct.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
