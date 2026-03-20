# Digital Holographic Microscopy — PhaseGAN

**GPU**  *Zhang Y. et al., PhaseGAN: A deep-learning phase-retrieval approach for unpaired datasets, Optics Letters, 2021*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
