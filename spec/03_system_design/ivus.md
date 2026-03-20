# Intravascular Ultrasound (IVUS) — System Design

```
[Source] → [Forward (Intravascular Ultrasound (IVUS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: RF pullback (frames × elements × samples, float32)  **Algorithms**: 15 — see `spec/ivus.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ivus/public/`

```python
from algorithm_base.ivus.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
