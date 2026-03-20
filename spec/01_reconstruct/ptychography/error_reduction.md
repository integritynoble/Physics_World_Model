# Ptychographic Imaging — Error Reduction (Fienup)

**CPU**  *Fienup, J.R. (1972) Phase retrieval algorithms: a comparison, Applied Optics*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('error_reduction', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
