# Ultrasound B-mode Imaging — DAS + NLM Post-filter

**CPU**  *Buades et al. 2005, CVPR; Coupe et al. 2009 TMI*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
