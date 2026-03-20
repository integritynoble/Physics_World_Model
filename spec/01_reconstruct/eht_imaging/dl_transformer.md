# Event Horizon Telescope (EHT) Imaging — RS-Transformer

**GPU**  *Transformer for remote sensing, 2022*
**Input**: hologram (H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/public/`

```python
from algorithm_base.eht_imaging.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
