# Single Photon Emission CT (SPECT) — SPECT-UNet

**GPU**  *Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6)*
**Input**: projections (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
