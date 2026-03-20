# CT + Fluorescence (FLIT) — System Design

```
[Source] → [Forward (CT + Fluorescence (FLIT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: XRF sinogram (angles × detectors × ch, float32)  **Algorithms**: 15 — see `spec/ct_fluorescence.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/public/`

```python
from algorithm_base.ct_fluorescence.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
