# Near-field Scanning Optical Microscopy (NSOM) — NSOM-Net

**GPU**  *Park, J. et al. (2020) DL for near-field optical microscopy, Optica 7(11)*
**Input**: near-field signal (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/public/`

```python
from algorithm_base.nsom.solvers import run_solver
x = run_solver('nsom_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
