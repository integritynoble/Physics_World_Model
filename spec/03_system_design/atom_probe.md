# Atom Probe Tomography (APT) — System Design

```
[Source] → [Forward (Atom Probe Tomography (APT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: hit positions (N × 3, float32)  **Algorithms**: 15 — see `spec/atom_probe.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/atom_probe/public/`

```python
from algorithm_base.atom_probe.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
