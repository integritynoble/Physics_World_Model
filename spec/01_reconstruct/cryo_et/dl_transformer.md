# Cryo-Electron Tomography (Cryo-ET) — TransCT

**GPU**  *Wang et al., IEEE TMI 2023*
**Input**: tilt series (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_et/public/`

```python
from algorithm_base.cryo_et.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
