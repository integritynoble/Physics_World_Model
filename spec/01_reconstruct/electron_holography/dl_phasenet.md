# Electron Holography — PhaseNet

**GPU**  *DL phase retrieval, 2018*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_holography/public/`

```python
from algorithm_base.electron_holography.solvers import run_solver
x = run_solver('dl_phasenet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
