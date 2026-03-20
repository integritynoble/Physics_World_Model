# Lattice Light-Sheet Microscopy — LLSM-CARE

**GPU**  *Weigert, M. et al. (2018) Content-aware restoration for lattice light-sheet, Nature Methods 15:1090*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lattice_lightsheet/public/`

```python
from algorithm_base.lattice_lightsheet.solvers import run_solver
x = run_solver('llsm_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
