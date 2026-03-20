# Atom Probe Tomography (APT) — 3D-Diffusion

**GPU**  *Diffusion for 3D reconstruction, 2025*
**Input**: hit positions (N × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/atom_probe/public/`

```python
from algorithm_base.atom_probe.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
