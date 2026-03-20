# Electrical Impedance Tomography (EIT) — EIT-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: boundary voltages (M, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/public/`

```python
from algorithm_base.impedance_tomo.solvers import run_solver
x = run_solver('eit_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
