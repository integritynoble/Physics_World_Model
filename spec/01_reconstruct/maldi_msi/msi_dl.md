# MALDI Mass Spectrometry Imaging — MSI-UNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: mass image (H × W × m/z, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/public/`

```python
from algorithm_base.maldi_msi.solvers import run_solver
x = run_solver('msi_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
