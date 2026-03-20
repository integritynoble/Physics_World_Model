# Spinning Disk Confocal Microscopy — SD-CARE

**GPU**  *Weigert, M. et al. (2018) CARE for spinning disk confocal, Nature Methods 15:1090*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spinning_disk/public/`

```python
from algorithm_base.spinning_disk.solvers import run_solver
x = run_solver('sd_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
