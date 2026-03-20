# Widefield Fluorescence Microscopy — WF-Diffusion (PnP-PGD DRUNet)

**GPU**  *Xie et al. 2023, arXiv*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('wf_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
