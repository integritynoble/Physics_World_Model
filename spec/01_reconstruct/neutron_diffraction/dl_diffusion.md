# Neutron Diffraction — 3D-Diffusion

**GPU**  *Diffusion for 3D reconstruction, 2025*
**Input**: pattern (2θ × intensity, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/public/`

```python
from algorithm_base.neutron_diffraction.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
