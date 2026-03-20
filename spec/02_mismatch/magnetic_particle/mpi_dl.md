# Magnetic Particle Imaging (MPI) — MPI-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: system function (freq × ch, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/public/`

```python
from algorithm_base.magnetic_particle.solvers import run_solver


x_wrong = run_solver('mpi_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('mpi_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
