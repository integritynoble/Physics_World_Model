# FTIR Spectroscopic Imaging — System Design

```
[Source] → [Forward (FTIR Spectroscopic Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: interferogram (H × W × OPD, float32)  **Algorithms**: 15 — see `spec/ftir_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ftir_imaging/public/`

```python
from algorithm_base.ftir_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
