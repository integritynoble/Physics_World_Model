# Pump-Probe Microscopy — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: transient spectra (T × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
