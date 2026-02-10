# PWM QuickStart

Get running with Physics World Model in under 60 seconds.

## 1. Install

```bash
pip install -e packages/pwm_core
```

## 2. Verify environment

```bash
pwm doctor
```

You should see all green `[PASS]` lines. Fix any red `[FAIL]` items before continuing.

## 3. Run the 60-second demo

```bash
pwm demo cassi --run --export-sharepack
```

This will:
- Load the CASSI spectral imaging CasePack
- Simulate a coded-aperture measurement
- Reconstruct the spectral datacube
- Export a SharePack you can send to a colleague

## 4. Python API

```python
import yaml
import numpy as np
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec

# Load a graph template
with open("packages/pwm_core/contrib/graph_templates.yaml") as f:
    templates = yaml.safe_load(f)["templates"]

# Compile the CASSI operator graph
spec = OperatorGraphSpec.model_validate(templates["cassi_sd_graph_v1"])
compiler = GraphCompiler()
op = compiler.compile(spec)

# Forward pass: simulate measurement
x = np.random.rand(*op.x_shape).astype(np.float32)
y = op.forward(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
```

## 5. Explore further

- **Gallery**: Browse all 26 modalities with `pwm demo <modality>`
- **Challenges**: See `packages/pwm_core/contrib/challenges/` for community challenges
- **Full docs**: See `docs/plan.md` for the complete architecture reference
- **CasePacks**: Pre-configured experiments in `packages/pwm_core/contrib/casepacks/`
