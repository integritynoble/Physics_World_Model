# Cryo-Electron Tomography (Cryo-ET) — CryoCARE

**GPU**  *Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol.*
**Input**: tilt series (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_et/public/`

```python
from algorithm_base.cryo_et.solvers import run_solver
x = run_solver('cryo_et_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
