# Muon Tomography — DiffusionRecon

**GPU**  *Song et al., 2024*
**Input**: muon tracks (N × 6, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/public/`

```python
from algorithm_base.muon_tomo.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
