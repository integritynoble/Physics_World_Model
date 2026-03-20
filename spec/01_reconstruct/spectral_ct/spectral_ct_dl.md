# Photon-Counting Spectral CT — SpectralCT-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: energy-bin sinos (bins × angles × det, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spectral_ct/public/`

```python
from algorithm_base.spectral_ct.solvers import run_solver
x = run_solver('spectral_ct_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
