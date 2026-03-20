# CEST MRI — SwinIR-Med

**GPU**  *Liang et al., ICCV 2021*
**Input**: Z-spectrum (offsets × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cest_mri/public/`

```python
from algorithm_base.cest_mri.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
