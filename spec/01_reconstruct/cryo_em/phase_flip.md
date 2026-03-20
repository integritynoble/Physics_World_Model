# Cryo-EM Single Particle Analysis — Phase-Flip CTF Correction

**CPU**  *Rosenthal & Henderson 2003, JMB*
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
x = run_solver('phase_flip', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
