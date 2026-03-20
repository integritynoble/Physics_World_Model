# Optical Coherence Tomography (OCT) — OCT Denoising Net

**CPU**  *Devalla et al. 2019, Biomed. Optics Express*
**Input**: spectrum (wavenumbers × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`

```python
from algorithm_base.oct.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
