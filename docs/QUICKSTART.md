# PWM Quick Start -- Your First Certified RunBundle

This guide walks you through running PWM locally and producing a machine-readable
Certificate for your imaging reconstruction algorithm.

## Prerequisites

- Python 3.10+
- pip install numpy scipy pyyaml

## 1. Install PWM

```bash
git clone https://github.com/integritynoble/Physics_World_Model.git
cd Physics_World_Model
pip install -e packages/pwm_core/
```

## 2. Run the evaluation harness

```bash
# Evaluate CT reconstruction with the default solver
pwm evaluate --modality ct --solver traditional_cpu --track correct --emit-certificate

# Or use the Python API directly
python3 -c "
from pwm_core.targeting.harness import Harness
from pwm_core.targeting.runbundle_emitter import emit_runbundle, issue_certificate

harness = Harness(modality='ct', solver='traditional_cpu', track='correct')
result = harness.run(n_scenes=5, seed=42)
bundle = emit_runbundle(result)
cert = issue_certificate(bundle)
print(f'Certificate: {cert}')
"
```

## 3. Inspect the RunBundle

```bash
ls run_ct_traditional_cpu_*/
# runbundle_manifest.json  -- spec, provenance, metrics, hashes
# certificate.json         -- trust verdict (R1-R4 gates, trust tier, domain flags)
# artifacts/               -- reconstruction outputs (x_hat, metrics, per_scene)
# logs/                    -- DR-IS decision records
```

## 4. Read the Certificate

```bash
python3 -c "
import json, glob
cert = json.load(open(glob.glob('run_ct_traditional_cpu_*/certificate.json')[0]))
print(f'Trust tier: {cert[\"trust_tier\"]}')
print(f'Gates: {list(cert[\"gate_verdicts\"].keys())}')
for gate, result in cert['gate_verdicts'].items():
    print(f'  {gate}: {result[\"verdict\"]}')
"
```

## 5. Use your own solver

```python
from pwm_core.targeting.harness import Harness
from pwm_core.targeting.runbundle_emitter import emit_runbundle, issue_certificate
import numpy as np

def my_solver(y, H_matrix, **kwargs):
    """Your reconstruction algorithm. Replace this."""
    # y = measurements, H_matrix = forward model
    # Return: reconstructed image (numpy array)
    x_hat = np.linalg.lstsq(H_matrix, y.ravel(), rcond=None)[0]
    return x_hat.reshape(kwargs.get('x_shape', (64, 64)))

harness = Harness(
    modality='ct',
    solver='traditional_cpu',
    track='correct',
    solver_fn=my_solver,  # Your custom solver
)
result = harness.run(n_scenes=5, seed=42)
bundle = emit_runbundle(result)
cert = issue_certificate(bundle)
```

## 6. Available modalities

12 Priority-1 modalities with golden reference bundles:

| Modality | Type | Default solver |
|----------|------|---------------|
| ct | Computed Tomography | FBP |
| mri | Magnetic Resonance Imaging | Zero-filled IFFT |
| pet | Positron Emission Tomography | FBP |
| spect | Single-Photon Emission CT | OSEM |
| ultrasound | Ultrasound Imaging | Delay-and-sum |
| oct | Optical Coherence Tomography | IFFT |
| mammography | X-ray Mammography | FBP |
| cbct | Cone-Beam CT | FBP |
| fundus | Retinal Fundus Photography | Wiener filter |
| endoscopy | Endoscopic Imaging | Richardson-Lucy |
| fmri | Functional MRI | GRAPPA |
| diffusion_mri | Diffusion-Weighted MRI | SENSE |

## 7. GitHub Action for CI

Add to your `.github/workflows/benchmark.yml`:

```yaml
name: PWM Benchmark
on: [pull_request]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install numpy scipy pyyaml
      - run: pip install -e packages/pwm_core/
      - run: pwm evaluate --modality ct --solver traditional_cpu --emit-certificate --scenes 3
```

## 8. Trust tiers

| Tier | Meaning |
|------|---------|
| Draft | Auto-generated, not yet verified |
| Author-confirmed | Authors reviewed the PWM result |
| Reproduced | Independent party confirmed the result |
| Certified | Full Judge pass + reviewer signoff |

## 9. Learn more

- Strategy: `pwm/notes/dyson_swarm_strategy.md`
- 172 modalities: `spec/` directory
- 2,732 algorithms: `algorithm_base/`
- Platform: https://pwm.platformai.org
