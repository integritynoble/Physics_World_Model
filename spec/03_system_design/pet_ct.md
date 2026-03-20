# PET/CT Fusion — System Design

```
[Source] → [Forward (PET/CT Fusion)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: PET sino + CT proj (both float32)  **Algorithms**: 3 — see `spec/pet_ct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_ct/public/`

```python
from algorithm_base.pet_ct.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
