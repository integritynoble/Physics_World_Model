# {MODALITY_NAME} — System Design Spec (Template)

> **Use Case 3: Simulate forward model (with mismatch) + reconstruct**

---

## System DAG

```
[Element 1] → [Element 2] → ... → [Detector] → y
      ↓              ↓
  [Mismatch]    [Mismatch]
```

---

## Element Definitions

### Element: Name (`element_id`)
- **Type**: source / interaction / geometry / detector / digitization
- **Parameters**:
  - `param1`: value
- **Noise**: type: model, params: values
- **Mismatch sources**:
  - `mismatch_name` [severity]: Description → Correction: method
- **Connects to**: `next_element`

---

## Noise Model

```
y = forward_model(x) + noise
```

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
# 1. Simulate forward model
# 2. Calibrate mismatch
# 3. Reconstruct
# 4. Evaluate PSNR/SSIM
```

---

## Using the Multi-Agent System

```bash
cd papers/system_design/
python main.py --modality {MODALITY_ID} --period forward --prompt "your system description"
python main.py --modality {MODALITY_ID} --period reconstruction --prompt "your algorithm"
```

---

## References

- `papers/system_design/` — three-agent pipeline (Plan + Judge + Performance)
- `packages/pwm_core/contrib/mismatch_db.yaml` — mismatch parameters
