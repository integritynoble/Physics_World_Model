# Acoustic Emission Testing (AE) — DeepAE-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: waveform (samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/public/`

```python
from algorithm_base.acoustic_emission.solvers import run_solver
x = run_solver('dl_localizer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
