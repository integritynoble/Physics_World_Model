# Magnetic Resonance Imaging (MRI) — k-t SPARSE-SENSE

**CPU**  *Lustig et al., ISMRM 2006*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('kt_sparse_sense', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
