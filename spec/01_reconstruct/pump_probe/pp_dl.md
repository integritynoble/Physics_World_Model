# Pump-Probe Microscopy — PumpProbe-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: transient spectra (T × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver
x = run_solver('pp_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
