# Use Case 3: Imaging System Design — Index

> Design complete imaging systems: define physical elements, simulate forward model (with realistic mismatch), and reconstruct with mismatch correction.

## What Is System Design?

System design means specifying a **DAG** (directed acyclic graph) of physical elements:

```
[Source] → [Sample/Object] → [Optics/Geometry] → [Detector] → [ADC/Digitization] → y
   ↓              ↓                  ↓                ↓
[Noise]       [Mismatch]        [Mismatch]        [Noise]
```

Each element has:
- Physical parameters (energy, flux, NA, pixel size, etc.)
- Noise model (Poisson, Gaussian, shot noise)
- Mismatch sources (beam hardening, CoR offset, PSF error, etc.)

The output is a realistic simulated measurement `y` plus a reconstruction.

## Available System Design Specs

| Spec File | Imaging System | Elements | Key Mismatch |
|-----------|---------------|----------|--------------|
| [ct_system.md](ct_system.md) | Sparse-view CT | Tube → Phantom → Geometry → Detector → ADC | Beam hardening, scatter, CoR offset |
| [mri_system.md](mri_system.md) | Parallel MRI | RF coil → k-space → Coil array | B0 drift, coil sensitivity error |
| [lensless_system.md](lensless_system.md) | Lensless Camera | LED → Diffuser → Sensor | PSF shift, background |
| [sim_system.md](sim_system.md) | SIM Microscopy | Laser → SLM → Objective → Camera | Pattern mismatch, defocus |

## System Design Format

Each system design spec follows this structure (based on PWM system_design paper):

```markdown
## System DAG

[Element 1] → [Element 2] → ... → [y]

### Element: Name (id)
- Type: source / interaction / geometry / detector / digitization
- Parameters: {param: value, ...}
- Noise: {model: poisson/gaussian, params: ...}
- Mismatch sources: {source: [severity], correction: method}
- Connects to: next_element
```

## Running System Design

```python
import sys
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public')
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public/packages/pwm_core')

# Full pipeline: design + simulate + reconstruct
from papers.system_design.pipeline.orchestrator import run_pipeline

# Run CT forward design
plan = run_pipeline(
    modality='ct',
    period='forward',
    prompt='Sparse-view CT, 60 angles, low-dose I0=1e4, pediatric chest'
)
print(plan)

# Run CT reconstruction design
plan_recon = run_pipeline(
    modality='ct',
    period='reconstruction',
    prompt='TV-ADMM with beam hardening and scatter correction'
)
```

Or use the multi-agent system directly:
```bash
cd papers/system_design/
python main.py --modality ct --period forward \
  --prompt "sparse-view CT 60 angles low-dose pediatric"
python main.py --modality ct --period reconstruction \
  --prompt "TV-ADMM mismatch correction"
```
