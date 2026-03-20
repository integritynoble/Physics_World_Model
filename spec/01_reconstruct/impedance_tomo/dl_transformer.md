# Electrical Impedance Tomography (EIT) — TransCT

**GPU**  *Wang et al., IEEE TMI 2023*
**Input**: boundary voltages (M, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/public/`

```python
from algorithm_base.impedance_tomo.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
