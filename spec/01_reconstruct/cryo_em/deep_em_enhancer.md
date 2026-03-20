# Cryo-EM Single Particle Analysis — DeepEMenhancer (DRUNet denoise)

**GPU**  *Sanchez-Garcia et al. 2021, Comms. Biol.*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('deep_em_enhancer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
