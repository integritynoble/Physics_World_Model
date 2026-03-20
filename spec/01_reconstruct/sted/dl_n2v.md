# STED Microscopy — Noise2Void

**GPU**  *Krull et al., CVPR 2019*
**Input**: STED + confocal (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sted/public/`

```python
from algorithm_base.sted.solvers import run_solver
x = run_solver('dl_n2v', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
