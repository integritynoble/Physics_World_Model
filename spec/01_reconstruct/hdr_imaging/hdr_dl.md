# High Dynamic Range (HDR) Imaging — HDR-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: multi-exposure (K × H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/public/`

```python
from algorithm_base.hdr_imaging.solvers import run_solver
x = run_solver('hdr_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
