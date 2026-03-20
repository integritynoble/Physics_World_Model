# X-ray Fluorescence Tomography — XRFT-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: XRF sinograms (elem × angles × det, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/public/`

```python
from algorithm_base.xrf_tomo.solvers import run_solver
x = run_solver('xrft_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
