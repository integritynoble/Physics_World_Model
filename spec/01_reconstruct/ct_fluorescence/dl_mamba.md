# CT + Fluorescence (FLIT) — MambaRecon

**GPU**  *SSM for inverse problems, 2026*
**Input**: XRF sinogram (angles × detectors × ch, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/public/`

```python
from algorithm_base.ct_fluorescence.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
