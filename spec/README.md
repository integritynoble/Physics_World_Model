# PWM Spec Database

4 use cases × 168+ modalities. Each spec has a run button (copy-paste Python), PSNR/SSIM evaluation, and visualization.

## Setup

```bash
# Local (Linux / macOS / Windows)
git clone https://github.com/integritynoble/Physics_World_Model
cd Physics_World_Model/pwm/public
pip install -e packages/pwm_core
```

```python
# Google Colab
!git clone https://github.com/integritynoble/Physics_World_Model
import sys; sys.path.insert(0, 'Physics_World_Model/pwm/public')
!pip install -e Physics_World_Model/pwm/public/packages/pwm_core -q
```

## Find a Spec

```bash
python3 spec/keyword_match.py "CT reconstruction"        # → spec/01_reconstruct/ct/
python3 spec/keyword_match.py "MRI mismatch"             # → spec/02_mismatch/mri/
python3 spec/keyword_match.py "lensless system design"   # → spec/03_system_design/lensless.md
python3 spec/keyword_match.py "Fresnel diffraction"      # → spec/04_simulation/09_optics/
python3 spec/keyword_match.py list                       # all 168 modalities
```

## Structure

| Folder | Use Case | Coverage |
|--------|----------|----------|
| `01_reconstruct/{modality}/{algorithm}.md` | Reconstruct | 168 modalities × all algorithms |
| `02_mismatch/{modality}/{algorithm}.md` | Mismatch + reconstruct | all benchmark modalities |
| `03_system_design/{modality}.md` | System DAG + reconstruct | all modalities |
| `04_simulation/{domain}/spec.md` | Physics simulation | from `papers/universal_simulation` |
| `{modality}.md` | Quick overview | all algorithms, one-page |

## GPU Note

Specs marked **GPU** require a CUDA-capable GPU. On CPU-only machines they raise `RuntimeError` — this does not affect other **CPU** specs. Skip GPU specs on CPU machines and run CPU ones directly.

## Regenerate

```bash
python3 spec/generators/generate_01_reconstruct.py
python3 spec/generators/generate_02_mismatch.py
python3 spec/generators/generate_03_system_design.py
python3 spec/generators/generate_04_simulation.py
python3 spec/generate_specs.py
```
