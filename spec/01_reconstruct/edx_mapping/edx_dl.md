# STEM-EDX Elemental Mapping — Richardson-Lucy (DL baseline)

**CPU**  *Tietz, C. et al. (2021) DL for EDS spectrum imaging, Ultramicroscopy 231*
**Input**: X-ray counts (H × W × channels, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/public/`

```python
from algorithm_base.edx_mapping.solvers import run_solver
x = run_solver('edx_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
