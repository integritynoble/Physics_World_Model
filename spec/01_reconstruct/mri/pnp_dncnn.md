# Magnetic Resonance Imaging (MRI) — PnP-DnCNN

**GPU**  *Ahmad et al., IEEE SPM 2020; Zhang et al., TIP 2017*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('pnp_dncnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
