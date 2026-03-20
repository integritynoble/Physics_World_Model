# Photon-Counting Spectral CT — System Design

```
[Source] → [Forward (Photon-Counting Spectral CT)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: energy-bin sinos (bins × angles × det, float32)  **Algorithms**: 3 — see `spec/spectral_ct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spectral_ct/public/`

```python
from algorithm_base.spectral_ct.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
