# Ptychographic Imaging — Ptychographic Iterative Engine (PIE)

**CPU**  *Rodenburg, J.M. & Faulkner, H.M.L. (2004) A phase retrieval algorithm for shifting illumination, Applied Physics Letters*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('pie', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
