# Passive Microwave Radiometry — RS-CNN

**GPU**  *Deep learning for remote sensing, 2018*
**Input**: brightness T (H × W × ch, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/public/`

```python
from algorithm_base.passive_microwave.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
