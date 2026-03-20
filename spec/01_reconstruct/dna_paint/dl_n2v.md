# DNA-PAINT Super-Resolution — Noise2Void

**GPU**  *Krull et al., CVPR 2019*
**Input**: localisation list (N × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dna_paint/public/`

```python
from algorithm_base.dna_paint.solvers import run_solver
x = run_solver('dl_n2v', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
