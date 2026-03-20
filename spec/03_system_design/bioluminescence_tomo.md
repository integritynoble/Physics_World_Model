# Bioluminescence Tomography (BLT) — System Design

```
[Source] → [Forward (Bioluminescence Tomography (BLT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: surface flux (H × W × angles, float32)  **Algorithms**: 15 — see `spec/bioluminescence_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/bioluminescence_tomo/public/`

```python
from algorithm_base.bioluminescence_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
