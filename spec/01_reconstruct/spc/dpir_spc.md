# Single-Pixel Camera (SPC) — DPIR

**GPU**  *Zhang et al., IEEE TPAMI 2022*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('dpir_spc', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
