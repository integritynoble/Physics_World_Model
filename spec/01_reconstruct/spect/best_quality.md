# Single Photon Emission CT (SPECT) — SPECT-DL (OSEM+)

**GPU**  *Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging*
**Input**: projections (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
