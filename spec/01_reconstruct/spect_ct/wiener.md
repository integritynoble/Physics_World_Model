# SPECT/CT Fusion — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: SPECT proj + CT sino (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect_ct/public/`

```python
from algorithm_base.spect_ct.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
