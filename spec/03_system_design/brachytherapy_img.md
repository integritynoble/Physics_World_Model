# Brachytherapy Imaging — System Design

```
[Source] → [Forward (Brachytherapy Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: dose map (H × W × D, float32)  **Algorithms**: 15 — see `spec/brachytherapy_img.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/public/`

```python
from algorithm_base.brachytherapy_img.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
