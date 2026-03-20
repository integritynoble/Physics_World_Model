# PALM/STORM Single-Molecule Localization — DECODE-SMLM

**GPU**  *Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090*
**Input**: localisations (N × 4: x,y,σ,I)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

```python
from algorithm_base.palm_storm.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
