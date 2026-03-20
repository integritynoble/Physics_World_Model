# Digital Holographic Microscopy — prDeep

**GPU**  *Metzler C.A. et al., prDeep: Robust Phase Retrieval with a Flexible Deep Network, ICML, 2018*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
