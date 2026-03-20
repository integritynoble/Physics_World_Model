# Digital Holographic Microscopy — PhaseNet

**GPU**  *Rivenson Y. et al., Phase recovery and holographic image reconstruction using deep learning in neural networks, Light: Science & Applications, 2018*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
