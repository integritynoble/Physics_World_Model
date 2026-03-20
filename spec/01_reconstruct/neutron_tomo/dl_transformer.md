# Neutron Radiography / Tomography — TransCT

**GPU**  *Wang et al., IEEE TMI 2023*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/public/`

```python
from algorithm_base.neutron_tomo.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
