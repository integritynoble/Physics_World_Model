# Full-Waveform Inversion (FWI) — InversionNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: seismic waveforms (receivers × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fwi/public/`

```python
from algorithm_base.fwi.solvers import run_solver
x = run_solver('fwi_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
