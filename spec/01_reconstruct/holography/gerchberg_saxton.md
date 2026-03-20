# Digital Holographic Microscopy — Gerchberg-Saxton

**CPU**  *Gerchberg R.W. & Saxton W.O., A practical algorithm for the determination of phase from image and diffraction plane pictures, Optik, 1972*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('gerchberg_saxton', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
