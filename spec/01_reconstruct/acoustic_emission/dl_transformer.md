# Acoustic Emission Testing (AE) — DL-Transformer

**GPU**  *Transformer reconstruction, 2023*
**Input**: waveform (samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/public/`

```python
from algorithm_base.acoustic_emission.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
