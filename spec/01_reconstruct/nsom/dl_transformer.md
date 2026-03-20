# Near-field Scanning Optical Microscopy (NSOM) — Probe-Transformer

**GPU**  *Transformer for probe imaging, 2023*
**Input**: near-field signal (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/public/`

```python
from algorithm_base.nsom.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
