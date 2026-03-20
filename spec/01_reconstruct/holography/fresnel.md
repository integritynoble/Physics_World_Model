# Digital Holographic Microscopy — Fresnel Propagation

**CPU**  *Schnars U. & Jueptner W., Digital Holography, Springer, 2005; Fresnel integral formulation*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('fresnel', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
