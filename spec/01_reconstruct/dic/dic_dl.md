# Differential Interference Contrast (DIC) — DIC-Net

**GPU**  *Mir, A. et al. (2015) Automated DIC microscopy, J. Microsc. 257(2)*
**Input**: DIC image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dic/public/`

```python
from algorithm_base.dic.solvers import run_solver
x = run_solver('dic_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
