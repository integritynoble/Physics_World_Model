# Expansion Microscopy (ExM) — Noise2Void

**GPU**  *Krull et al., CVPR 2019*
**Input**: confocal + expansion (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/public/`

```python
from algorithm_base.expansion.solvers import run_solver
x = run_solver('dl_n2v', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
