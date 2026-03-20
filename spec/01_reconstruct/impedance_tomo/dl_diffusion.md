# Electrical Impedance Tomography (EIT) — DiffusionRecon

**GPU**  *Song et al., 2024*
**Input**: boundary voltages (M, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/public/`

```python
from algorithm_base.impedance_tomo.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
