# Full-Waveform Inversion (FWI) — RS-Diffusion

**GPU**  *Diffusion model for RS, 2024*
**Input**: seismic waveforms (receivers × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fwi/public/`

```python
from algorithm_base.fwi.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
