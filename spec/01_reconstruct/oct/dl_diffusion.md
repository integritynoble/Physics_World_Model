# Optical Coherence Tomography (OCT) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: spectrum (wavenumbers × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`

```python
from algorithm_base.oct.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
