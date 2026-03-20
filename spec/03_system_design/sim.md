# Structured Illumination Microscopy (SIM) — System Design

```
[488nm Laser] → [SLM: 3 orient. × 3 phases] → [Fluorescent Sample]
                                                        ↓
                                              [1.4 NA Objective (PSF)]
                                                        ↓
                                               [sCMOS Detector] → [16-bit ADC] → y
                                                        ↓
                                                [Poisson + readout σ=1.5 e⁻]
```

**Mismatch**: illumination phase offset `[0, 2π/3] rad`
**Input**: raw frames (9 × H × W: 3 angles × 3 phases)  **Algorithms**: 16 — see `spec/sim.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/`
**Paper**: `papers/system_design/outputs/sim_forward_v1_iter1.md`

```python
from algorithm_base.sim.solvers import run_solver
from pwm_core.mismatch.operators import sim_calibrate_phase
phase = sim_calibrate_phase(y)
calib_cfg = {"phase_offset": float(phase)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
