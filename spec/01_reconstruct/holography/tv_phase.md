# Digital Holographic Microscopy — TV-Phase Retrieval

**CPU**  *Horisaki R. et al., Single-shot phase imaging with randomised light, Optics Express, 2016; TV regularisation for phase, 2008*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('tv_phase', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
