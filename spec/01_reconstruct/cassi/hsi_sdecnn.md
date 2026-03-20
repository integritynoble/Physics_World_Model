# Coded Aperture Snapshot Spectral Imaging (CASSI) — PnP-HSICNN

**GPU**  **PSNR**: ~25.12 dB  *Maffei et al., TGRS 2020 — 25.12 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
x = run_solver('hsi_sdecnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
