# Optical Diffraction Tomography (ODT) — System Design

```
[Source] → [Forward (Optical Diffraction Tomography (ODT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: holograms (angles × H × W, complex64)  **Algorithms**: 15 — see `spec/odt.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/odt/public/`

```python
from algorithm_base.odt.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
