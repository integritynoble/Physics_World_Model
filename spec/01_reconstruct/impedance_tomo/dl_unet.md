# Electrical Impedance Tomography (EIT) — U-Net Recon

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: boundary voltages (M, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/public/`

```python
from algorithm_base.impedance_tomo.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
