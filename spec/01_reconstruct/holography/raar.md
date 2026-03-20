# Digital Holographic Microscopy — RAAR

**CPU**  *Luke D.R., Relaxed averaged alternating reflections for diffraction imaging, Inverse Problems, 2005*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('raar', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
