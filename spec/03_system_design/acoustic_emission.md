# Acoustic Emission Testing (AE) — System Design

```
[Source] → [Forward (Acoustic Emission Testing (AE))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: waveform (samples, float32)  **Algorithms**: 15 — see `spec/acoustic_emission.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/public/`

```python
from algorithm_base.acoustic_emission.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
