# SPECT/CT Fusion — System Design

```
[Source] → [Forward (SPECT/CT Fusion)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: SPECT proj + CT sino (both float32)  **Algorithms**: 15 — see `spec/spect_ct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect_ct/public/`

```python
from algorithm_base.spect_ct.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
