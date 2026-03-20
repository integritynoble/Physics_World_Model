# Neutron Diffraction — System Design

```
[Source] → [Forward (Neutron Diffraction)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: pattern (2θ × intensity, float32)  **Algorithms**: 15 — see `spec/neutron_diffraction.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/public/`

```python
from algorithm_base.neutron_diffraction.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
