# Second Harmonic Generation (SHG) Microscopy — SHG-CARE

**GPU**  *Weigert, M. et al. (2018) CARE for SHG imaging, Nature Methods 15:1090*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shg/public/`

```python
from algorithm_base.shg.solvers import run_solver
x = run_solver('shg_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
