# Ptychographic Imaging — Extended PIE (ePIE)

**CPU**  *Maiden, A.M. & Rodenburg, J.M. (2009) An improved ptychographical phase retrieval algorithm for diffractive imaging, Ultramicroscopy*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
