# Seismic Tomography — RS-Transformer

**GPU**  *Transformer for remote sensing, 2022*
**Input**: travel times (src-recv, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/public/`

```python
from algorithm_base.seismic_tomo.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
