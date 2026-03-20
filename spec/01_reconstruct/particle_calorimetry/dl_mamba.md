# Particle Calorimetry — DL-Mamba

**GPU**  *SSM reconstruction, 2026*
**Input**: deposits (N × 5, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/public/`

```python
from algorithm_base.particle_calorimetry.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
