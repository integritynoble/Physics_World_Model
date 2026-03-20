# STEM-EDX Elemental Mapping — System Design

```
[Source] → [Forward (STEM-EDX Elemental Mapping)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: X-ray counts (H × W × channels, float32)  **Algorithms**: 15 — see `spec/edx_mapping.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/public/`

```python
from algorithm_base.edx_mapping.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
