# Single Photon Emission CT (SPECT) — MambaRecon

**GPU**  *SSM for inverse problems, 2026*
**Input**: projections (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
