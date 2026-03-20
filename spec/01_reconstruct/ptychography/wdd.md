# Ptychographic Imaging — Wigner Distribution Deconvolution (WDD)

**CPU**  *Rodenburg, J.M. & Bates, R.H.T. (1992) The theory of super-resolution electron microscopy via WDD, Phil. Trans. R. Soc. A*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('wdd', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
