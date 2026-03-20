# Magnetic Resonance Imaging (MRI) — U-Net (fastMRI)

**GPU**  *Zbontar et al., arXiv 2018; Ronneberger et al., MICCAI 2015*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('unet_mri', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
