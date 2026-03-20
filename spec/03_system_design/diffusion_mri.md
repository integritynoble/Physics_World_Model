# Diffusion MRI (DTI) — System Design

```
[Source] → [Forward (Diffusion MRI (DTI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: DWI (N_dirs × H × W × D, float32)  **Algorithms**: 15 — see `spec/diffusion_mri.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/public/`

```python
from algorithm_base.diffusion_mri.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
