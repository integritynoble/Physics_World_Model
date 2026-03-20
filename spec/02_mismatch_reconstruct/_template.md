# {MODALITY_NAME} — Mismatch Correction + Reconstruction (Template)

> **Use Case 2: Correct operator mismatch, then reconstruct**

---

## Mismatch Overview

*(Describe the common mismatch sources for this modality. Refer to `mismatch_db.yaml` for ranges.)*

| Mismatch Source | Parameter | Range | Typical Error | Correction Method |
|----------------|-----------|-------|---------------|-------------------|
| *Mismatch 1* | `param_name` | [min, max] | typical | correction_method |

---

## Mismatch Parameters (User-Provided)

```python
# Known values (provide if you know them)
mismatch_params = {
    'param_name': None,   # Set to float if known, None to auto-estimate
    'search_range': [min, max],
    'search_steps': 20,
}
```

---

## Run Button

```python
# ============================================================
# {MODALITY_NAME} Mismatch Correction — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from pwm_core.mismatch.operators import compute_psnr

# -------------------------------------------------------
# 1. Load your measurement
# -------------------------------------------------------
# y = np.load('your_measurement.npy').astype(np.float32)
# x_true = np.load('your_gt.npy').astype(np.float32)   # optional

# -------------------------------------------------------
# 2. Calibrate mismatch
# -------------------------------------------------------
# (Use appropriate calibration function from pwm_core.mismatch.operators)
# corrected_param = calibrate_{modality}_{param}(y, ...)

# -------------------------------------------------------
# 3. Reconstruct with corrected operator
# -------------------------------------------------------
# x_hat = run_solver(...)

# -------------------------------------------------------
# 4. Evaluate
# -------------------------------------------------------
# psnr = compute_psnr(x_true, x_hat)
# print(f"PSNR: {psnr:.2f} dB")
```

---

## References

- `packages/pwm_core/contrib/mismatch_db.yaml` — all mismatch parameters
- `packages/pwm_core/pwm_core/mismatch/operators.py` — calibration functions
