# Gear 4: Action Networks -- The Hardware

> Robotic labs and micro-factories giving AI "hands and feet."

**Status: PLANNED**

---

## The Principle

Action networks are the physical infrastructure that turns AI decisions into real-world changes. In manufacturing, this means robotic assembly lines. In imaging, this means automated calibration hardware: motorized stages, source controllers, detector adjustments -- anything that lets a software system physically modify an imaging instrument.

---

## PWM Implementation

PWM currently implements **software actuation**: the corrected operator parameters are fed back into the reconstruction pipeline, improving image quality without touching hardware. Hardware actuation (closed-loop instrument control) is on the roadmap.

### Software Actuation (Current)

The calibration pipeline estimates mismatch parameters (dx, dy, theta, etc.) and produces a corrected operator H_hat. This corrected operator replaces the nominal operator in the reconstruction pipeline:

1. **Measure**: Capture measurement y with the real (mismatched) instrument
2. **Infer**: Estimate mismatch parameters from y using Algorithm 1 (grid search) + Algorithm 2 (gradient refinement)
3. **Correct**: Build corrected operator H_hat with estimated parameters
4. **Reconstruct**: Solve inverse problem using H_hat instead of H_nom
5. **Verify**: Check re-projection consistency, log decision in DR-IS

### Validated Software Actuation

16 modalities have validated operator correction with measurable improvement:

| Modality | Parameters Corrected | Typical Gain |
|----------|---------------------|-------------|
| CASSI | dx, dy, theta, phi_d | +4.8 dB |
| CT | center of rotation | +13.0 dB |
| MRI | coil sensitivities | +48.3 dB |
| CACTI | mask timing | +12.6 dB |
| SPC | gain/bias | +24 dB |
| Ptychography | probe position | +7.1 dB |
| OCT | dispersion coefficients | +50.5 dB |
| ... | (16 total) | |

### Hardware Actuation (Roadmap)

Future hardware-in-the-loop capabilities:

| Stage | Capability | Timeline |
|-------|-----------|----------|
| Phase A | Software actuation only (current) | Now |
| Phase B | Instrument controller API specification | 6-12 months |
| Phase C | Closed-loop calibration on partner instruments | 12-24 months |
| Phase D | Live hardware scenarios in LIP-Arena | 24+ months |

---

## Key Files

| File | Description |
|------|-------------|
| `docs/purpose.md` Layer 6 | Actuation layer definition (software + hardware + reporting) |
| `packages/pwm_core/pwm_core/calibration/cassi_upwmi_alg12.py` | CASSI calibration algorithms (1032 lines) |
| `packages/pwm_core/pwm_core/calibration/cassi_torch_modules.py` | PyTorch differentiable forward model (542 lines) |
| `packages/pwm_core/pwm_core/calibration/cassi_mst_modules.py` | MST-L integration for gradient-based calibration |

---

## What's Built

- **Software actuation for 16 modalities**: Corrected operators fed into reconstruction pipelines
- **CASSI calibration pipeline**: Full Algorithm 1 (grid search) + Algorithm 2 (GPU gradient refinement), validated on 10 KAIST scenes
- **Differentiable forward models**: PyTorch modules enabling gradient-based operator correction
- **All actions logged**: Every correction decision recorded in DR-IS with evidence and compute consumed

---

## What's Next

- **Instrument controller API**: Define a protocol for PWM to send calibration commands to hardware (stage motors, source tuning, detector gain)
- **Hardware-in-the-loop LIP-Arena**: Evaluation scenarios using live instruments (Phase D)
- **Closed-loop calibration**: Iterative measure-correct-verify cycles on physical instruments
- **Multi-instrument orchestration**: Coordinate calibration across multiple instruments in a facility

---

## Connections

- **Gear 1 (Targeting System)**: The harness scores the outcomes of actuation (software or hardware)
- **Gear 3 (Compute Escrow)**: Calibration algorithms consume declared compute budgets
- **Gear 6 (Decision Logs)**: Every actuation is logged in DR-IS with full evidence chain
