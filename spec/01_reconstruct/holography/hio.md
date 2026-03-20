# Digital Holographic Microscopy — Hybrid Input-Output (HIO)

**CPU**  *Fienup J.R., Phase retrieval algorithms: a comparison, Applied Optics, 1982*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('hio', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
