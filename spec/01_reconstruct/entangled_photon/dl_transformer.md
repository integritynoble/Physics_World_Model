# Entangled Photon Microscopy — CS-Transformer

**GPU**  *Transformer for CS, 2023*
**Input**: coincidence counts (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/public/`

```python
from algorithm_base.entangled_photon.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
