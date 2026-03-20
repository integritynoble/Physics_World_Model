# Quantum Illumination — System Design

```
[Source] → [Forward (Quantum Illumination)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: coincidence image (H × W, float32)  **Algorithms**: 15 — see `spec/quantum_illumination.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/quantum_illumination/public/`

```python
from algorithm_base.quantum_illumination.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
