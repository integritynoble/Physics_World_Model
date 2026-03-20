# DNA-PAINT Super-Resolution — System Design

```
[Source] → [Forward (DNA-PAINT Super-Resolution)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: localisation list (N × 2, float32)  **Algorithms**: 15 — see `spec/dna_paint.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dna_paint/public/`

```python
from algorithm_base.dna_paint.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
