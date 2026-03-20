# XFEL Serial Femtosecond Crystallography (SFX) — System Design

```
[Source] → [Forward (XFEL Serial Femtosecond Crystallography (SFX))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: diffraction patterns (N_shots × H × W, float32)  **Algorithms**: 15 — see `spec/xfel_sfx.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/public/`

```python
from algorithm_base.xfel_sfx.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
