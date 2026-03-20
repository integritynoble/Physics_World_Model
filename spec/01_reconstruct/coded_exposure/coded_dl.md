# Coded Exposure / Flutter Shutter — FlowNet-Coded [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: coded frames (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/public/`

```python
from algorithm_base.coded_exposure.solvers import run_solver
x = run_solver('coded_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
