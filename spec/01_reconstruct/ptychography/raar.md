# Ptychographic Imaging — Relaxed Averaged Alternating Reflections (RAAR)

**CPU**  *Luke, D.R. (2005) Relaxed averaged alternating reflections for diffraction imaging, Inverse Problems*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('raar', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
