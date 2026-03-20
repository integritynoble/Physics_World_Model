# Sonar Imaging — RS-CNN

**GPU**  *Deep learning for remote sensing, 2018*
**Input**: echo data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sonar/public/`

```python
from algorithm_base.sonar.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
