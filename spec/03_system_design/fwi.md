# Full-Waveform Inversion (FWI) — System Design

```
[Source] → [Forward (Full-Waveform Inversion (FWI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: seismic waveforms (receivers × time, float32)  **Algorithms**: 15 — see `spec/fwi.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fwi/public/`

```python
from algorithm_base.fwi.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
