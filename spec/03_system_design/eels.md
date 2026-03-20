# Electron Energy Loss Spectroscopy (EELS) — System Design

```
[Source] → [Forward (Electron Energy Loss Spectroscopy (EELS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: energy-loss spectrum (H × W × E, float32)  **Algorithms**: 16 — see `spec/eels.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/public/`

```python
from algorithm_base.eels.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
