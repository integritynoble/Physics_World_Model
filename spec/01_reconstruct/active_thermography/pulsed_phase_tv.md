# Active Thermography (IR) — Pulsed-Phase TV [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: thermal sequence (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/public/`

```python
from algorithm_base.active_thermography.solvers import run_solver
x = run_solver('pulsed_phase_tv', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
