# Muon Tomography — System Design

```
[Source] → [Forward (Muon Tomography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: muon tracks (N × 6, float32)  **Algorithms**: 15 — see `spec/muon_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/public/`

```python
from algorithm_base.muon_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
