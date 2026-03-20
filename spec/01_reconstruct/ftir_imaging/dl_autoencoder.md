# FTIR Spectroscopic Imaging — Spec-AE

**GPU**  *Autoencoder spectral unmixing, 2020*
**Input**: interferogram (H × W × OPD, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ftir_imaging/public/`

```python
from algorithm_base.ftir_imaging.solvers import run_solver
x = run_solver('dl_autoencoder', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
