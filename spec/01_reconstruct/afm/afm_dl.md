# Atomic Force Microscopy (AFM) — AFM-UNet

**GPU**  *Cherukara, M.J. et al. (2020) AI-enabled high-res, real-time imaging, npj Comput. Mater. 6:203*
**Input**: force-distance map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/afm/public/`

```python
from algorithm_base.afm.solvers import run_solver
x = run_solver('afm_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
