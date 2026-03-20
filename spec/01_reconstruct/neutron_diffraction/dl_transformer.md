# Neutron Diffraction — 3D-Transformer

**GPU**  *Transformer for 3D, 2023*
**Input**: pattern (2θ × intensity, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/public/`

```python
from algorithm_base.neutron_diffraction.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
