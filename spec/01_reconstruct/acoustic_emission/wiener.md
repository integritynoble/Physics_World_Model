# Acoustic Emission Testing (AE) — Wiener Deconvolution

**CPU**  *Wiener, Extrapolation, Interpolation... 1949*
**Input**: waveform (samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/public/`

```python
from algorithm_base.acoustic_emission.solvers import run_solver
cfg = {'reg': 0.01}
x = run_solver('wiener', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
