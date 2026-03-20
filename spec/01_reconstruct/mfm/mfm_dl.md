# Magnetic Force Microscopy (MFM) — MFM-UNet

**GPU**  *Kim, M. et al. (2021) DL for magnetic force microscopy, npj Comput. Mater. 7:87*
**Input**: magnetic force map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mfm/public/`

```python
from algorithm_base.mfm.solvers import run_solver
x = run_solver('mfm_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
