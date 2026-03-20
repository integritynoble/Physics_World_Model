# Digital Holographic Microscopy — DeepDIH

**GPU**  *Ren Z. et al., End-to-end deep learning framework for digital holographic reconstruction, Optics Express, 2019*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('deep_dih', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
