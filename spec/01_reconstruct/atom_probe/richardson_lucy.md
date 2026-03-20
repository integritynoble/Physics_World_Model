# Atom Probe Tomography (APT) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: hit positions (N × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/atom_probe/public/`

```python
from algorithm_base.atom_probe.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
