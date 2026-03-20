# Proton Therapy Imaging — System Design

```
[Source] → [Forward (Proton Therapy Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: dose distribution (H × W × D, float32)  **Algorithms**: 15 — see `spec/proton_therapy_img.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_therapy_img/public/`

```python
from algorithm_base.proton_therapy_img.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
