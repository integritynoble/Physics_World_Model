# Positron Emission Tomography (PET) — NeuroLF-PET

**GPU**  *Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet/public/`

```python
from algorithm_base.pet.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
