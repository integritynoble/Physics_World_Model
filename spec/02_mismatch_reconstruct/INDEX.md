# Use Case 2: Mismatch Correction + Reconstruct — Index

Mismatch occurs when the reconstruction algorithm uses an incorrect forward operator
(wrong PSF, wrong geometry, wrong calibration). This use case corrects the mismatch
before or during reconstruction.

## What Is Mismatch?

| Mismatch Type | Example | Physical Cause |
|---------------|---------|----------------|
| `spatial_shift` | PSF position error | Stage drift, thermal expansion |
| `rotation` | Angular misalignment | Encoder error, vibration |
| `scale` | Magnification error | Lens or camera calibration |
| `blur` | PSF width error | Wrong NA assumption, defocus |
| `offset` | Background / dark current | Calibration drift, autofluorescence |
| `center_of_rotation` | CoR offset in CT | Mechanical misalignment |
| `dispersion_step` | Prism step in CASSI | Manufacturing tolerance |
| `coil_sensitivity` | Coil map error in MRI | Temperature drift, patient motion |

## Available Mismatch Specs

| Spec File | Modality | Mismatch Type | Correction Method |
|-----------|----------|---------------|-------------------|
| [ct_mismatch.md](ct_mismatch.md) | CT | Center-of-rotation offset | Cross-correlation |
| [mri_mismatch.md](mri_mismatch.md) | MRI | Coil sensitivity, B0 drift | ESPIRiT |
| [cassi_mismatch.md](cassi_mismatch.md) | CASSI | Dispersion step error | Grid search |
| [lensless_mismatch.md](lensless_mismatch.md) | Lensless | PSF shift | Gradient calibration |
| [microscopy_mismatch.md](microscopy_mismatch.md) | Widefield/Confocal | PSF sigma, defocus | Grid search |

For other modalities, use `_template.md` and refer to `packages/pwm_core/contrib/mismatch_db.yaml`
for the full list of mismatch parameters.

## Common Correction Methods

| Method | Description | Best For |
|--------|-------------|---------|
| `grid_search` | Evaluate reconstruction quality over a grid of mismatch parameter values | PSF sigma, defocus |
| `cross_correlation` | Correlate 0°/180° projections or calibration targets | CT CoR, registration |
| `espirit` | Estimate coil sensitivity maps from ACS data | MRI parallel imaging |
| `gradient_calibration` | Gradient-based optimization of operator parameters | PSF shift, dispersion |
| `neural_calibration` | DL-based calibration from the measurement itself | General-purpose |

## Quick Start

```python
import sys
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public')
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public/packages/pwm_core')

# CT center-of-rotation calibration + reconstruction
from pwm_core.mismatch.operators import ct_calibrate_cor, ct_radon_forward, ct_sart_tv_recon
import numpy as np

# 1. Simulate CT with CoR mismatch
y_mismatch = ct_radon_forward(x_phantom, cor_offset=2.5)  # 2.5 pixel CoR error

# 2. Calibrate
cor_estimated = ct_calibrate_cor(y_mismatch)
print(f"Estimated CoR offset: {cor_estimated:.3f} px")

# 3. Reconstruct with corrected operator
x_hat = ct_sart_tv_recon(y_mismatch, cor_offset=cor_estimated)
```

For all mismatch correction functions, see `packages/pwm_core/pwm_core/mismatch/operators.py`.
