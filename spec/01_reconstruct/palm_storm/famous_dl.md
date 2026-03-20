# PALM/STORM Single-Molecule Localization — DeepSTORM

**GPU**  *Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4)*
**Input**: localisations (N × 4: x,y,σ,I)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

```python
from algorithm_base.palm_storm.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
