# Expansion Microscopy (ExM) — EXpansionNet

**GPU**  *Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090*
**Input**: confocal + expansion (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/public/`

```python
from algorithm_base.expansion.solvers import run_solver
x = run_solver('exm_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
