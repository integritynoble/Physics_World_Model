# PALM/STORM Single-Molecule Localization — System Design

```
[Source] → [Forward (PALM/STORM Single-Molecule Localization)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: localisations (N × 4: x,y,σ,I)  **Algorithms**: 15 — see `spec/palm_storm.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

```python
from algorithm_base.palm_storm.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
