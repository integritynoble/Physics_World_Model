# MALDI Mass Spectrometry Imaging — System Design

```
[Source] → [Forward (MALDI Mass Spectrometry Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: mass image (H × W × m/z, float32)  **Algorithms**: 15 — see `spec/maldi_msi.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/public/`

```python
from algorithm_base.maldi_msi.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
