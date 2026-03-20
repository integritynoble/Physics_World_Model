# Electron Energy Loss Spectroscopy (EELS) — Spec-Transformer

**GPU**  *Transformer for spectroscopy, 2023*
**Input**: energy-loss spectrum (H × W × E, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/public/`

```python
from algorithm_base.eels.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
