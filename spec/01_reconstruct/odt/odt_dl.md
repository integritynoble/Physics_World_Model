# Optical Diffraction Tomography (ODT) — ODT-Net (PhaseNet) [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: holograms (angles × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/odt/public/`

```python
from algorithm_base.odt.solvers import run_solver
x = run_solver('odt_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
