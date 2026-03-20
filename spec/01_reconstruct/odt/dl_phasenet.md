# Optical Diffraction Tomography (ODT) — PhaseNet

**GPU**  *DL phase retrieval, 2018*
**Input**: holograms (angles × H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/odt/public/`

```python
from algorithm_base.odt.solvers import run_solver
x = run_solver('dl_phasenet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
