# Coherent Diffractive Imaging / Phase Retrieval — PhaseNet

**GPU**  *DL phase retrieval, 2018*
**Input**: diffraction intensities (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/public/`

```python
from algorithm_base.phase_retrieval.solvers import run_solver
x = run_solver('dl_phasenet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
