# Magnetic Resonance Imaging (MRI) — Nuclear Norm (SVT)

**CPU**  *Cai, Candes, Shen, SIAM J Optim 2010; Shin et al., MRM 2014 (SAKE)*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('nuclear_norm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
