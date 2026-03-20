# Pump-Probe Microscopy — ReconNet

**GPU**  *DL for CS reconstruction, 2016*
**Input**: transient spectra (T × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver
x = run_solver('dl_reconnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
