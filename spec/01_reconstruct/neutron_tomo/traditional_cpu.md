# Neutron Radiography / Tomography — FBP (neutron tomography)

**CPU**
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/public/`

```python
from algorithm_base.neutron_tomo.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
