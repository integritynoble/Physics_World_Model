# Single-Pixel Camera (SPC) — ISTA-Net+ v2

**GPU**  *Zhang & Ghanem, CVPR 2018 (DRS variant)*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('ista_net_plus', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
