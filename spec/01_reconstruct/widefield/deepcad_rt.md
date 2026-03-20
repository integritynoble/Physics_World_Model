# Widefield Fluorescence Microscopy — DeepCAD-RT (PnP-DRS DRUNet)

**GPU**  *Li et al. 2023, Nature Methods*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('deepcad_rt', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
