# Phase Contrast Microscopy — PhaseNet

**GPU**  *Rivenson, Y. et al. (2018) Phase recovery with DL, Light: Sci. & Appl. 7:17141*
**Input**: image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_contrast/public/`

```python
from algorithm_base.phase_contrast.solvers import run_solver
x = run_solver('pc_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
