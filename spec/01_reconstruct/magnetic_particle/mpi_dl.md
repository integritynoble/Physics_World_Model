# Magnetic Particle Imaging (MPI) — MPI-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: system function (freq × ch, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/public/`

```python
from algorithm_base.magnetic_particle.solvers import run_solver
x = run_solver('mpi_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
