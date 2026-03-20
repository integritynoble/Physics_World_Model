# X-ray Crystallography — PhaseNet

**GPU**  *DL phase retrieval, 2018*
**Input**: structure factors (hkl × F, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_crystallography/public/`

```python
from algorithm_base.xray_crystallography.solvers import run_solver
x = run_solver('dl_phasenet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
