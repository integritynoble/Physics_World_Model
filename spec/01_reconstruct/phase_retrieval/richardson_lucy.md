# Coherent Diffractive Imaging / Phase Retrieval — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: diffraction intensities (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/public/`

```python
from algorithm_base.phase_retrieval.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
