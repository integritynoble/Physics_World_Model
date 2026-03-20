# Scanning Tunneling Microscopy (STM) — STM-Net

**GPU**  *Ziatdinov, M. et al. (2021) DL for atomic-level STM, Nat. Mach. Intell. 3:269*
**Input**: tunneling map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/public/`

```python
from algorithm_base.stm.solvers import run_solver
x = run_solver('stm_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
