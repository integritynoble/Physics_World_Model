# PET/MR Fusion — System Design

```
[Source] → [Forward (PET/MR Fusion)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: PET sino + MRI k-space (both float32)  **Algorithms**: 3 — see `spec/pet_mr.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_mr/public/`

```python
from algorithm_base.pet_mr.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
