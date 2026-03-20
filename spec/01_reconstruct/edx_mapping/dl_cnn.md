# STEM-EDX Elemental Mapping — Spec-CNN

**GPU**  *CNN for spectroscopy, 2018*
**Input**: X-ray counts (H × W × channels, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/public/`

```python
from algorithm_base.edx_mapping.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
