# Ptychographic Imaging — PtychoMamba (RED-DRUNet)

**GPU**  *Li, Z. et al. (2024) State-space models for efficient ptychographic reconstruction, ACS Photonics*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('ptycho_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
