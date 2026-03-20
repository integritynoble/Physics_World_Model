# DNA-PAINT Super-Resolution — DECODE-PAINT

**GPU**  *Speiser, A. et al. (2021) DL for dense SMLM, Nature Methods 18:1090*
**Input**: localisation list (N × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dna_paint/public/`

```python
from algorithm_base.dna_paint.solvers import run_solver
x = run_solver('dna_paint_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
