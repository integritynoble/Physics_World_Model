# Full-Waveform Inversion (FWI) — RS-CNN

**GPU**  *Deep learning for remote sensing, 2018*
**Input**: seismic waveforms (receivers × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fwi/public/`

```python
from algorithm_base.fwi.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
