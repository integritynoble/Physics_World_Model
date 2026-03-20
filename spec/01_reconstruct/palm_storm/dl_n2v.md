# PALM/STORM Single-Molecule Localization — Noise2Void

**GPU**  *Krull et al., CVPR 2019*
**Input**: localisations (N × 4: x,y,σ,I)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

```python
from algorithm_base.palm_storm.solvers import run_solver
x = run_solver('dl_n2v', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
