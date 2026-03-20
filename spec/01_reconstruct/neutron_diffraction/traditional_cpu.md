# Neutron Diffraction — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: pattern (2θ × intensity, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/public/`

```python
from algorithm_base.neutron_diffraction.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
