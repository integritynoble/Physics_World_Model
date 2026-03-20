# Quantum Illumination — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: coincidence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/quantum_illumination/public/`

```python
from algorithm_base.quantum_illumination.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
