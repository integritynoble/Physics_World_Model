# Digital Holographic Microscopy — HoloNet

**GPU**  *Wu Y. et al., Extended depth-of-field in holographic imaging using deep-learning-based autofocusing, Nature Methods / Optica, 2019*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('holonet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
