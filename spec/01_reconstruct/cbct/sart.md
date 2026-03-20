# Cone-Beam Computed Tomography (CBCT) — Simultaneous ART (SART)

**CPU**  *Andersen, A.H. & Kak, A.C. (1984) Simultaneous algebraic reconstruction technique (SART), Ultrasonic Imaging*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('sart', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
