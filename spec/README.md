# PWM Spec Database

4 use cases × 168+ modalities. Each spec has a run button (copy-paste Python), PSNR/SSIM evaluation, and visualization.

## Setup

```bash
# Local / server
git clone https://github.com/integritynoble/Physics_World_Model
cd Physics_World_Model/pwm/public
pip install -e packages/pwm_core

# Google Colab
!git clone https://github.com/integritynoble/Physics_World_Model
import sys; sys.path.insert(0, 'Physics_World_Model/pwm/public')
```

## Find a Spec

```bash
python3 spec/keyword_match.py "CT reconstruction"        # → spec/01_reconstruct/ct/
python3 spec/keyword_match.py "MRI mismatch"             # → spec/02_mismatch/mri.md
python3 spec/keyword_match.py "lensless system design"   # → spec/03_system_design/lensless.md
python3 spec/keyword_match.py "Fresnel diffraction"      # → spec/04_simulation/09_optics/
python3 spec/keyword_match.py list                       # all 168 modalities
```

## Structure

| Folder | Use Case | Files | Coverage |
|--------|----------|-------|----------|
| `01_reconstruct/{modality}/{algorithm}.md` | Reconstruct with specific algorithm | 2571 | 168 modalities × all algorithms |
| `02_mismatch/{modality}.md` | Mismatch correction + reconstruct | 168 | all benchmark modalities |
| `03_system_design/{modality}.md` | Imaging system DAG + reconstruct | 168 | all modalities |
| `04_simulation/{domain}/spec.md` | Physics simulation examples | 12 | from `papers/universal_simulation` |
| `{modality}.md` | Quick overview | 168 | all algorithms, one-page |

## GPU Note

GPU solvers raise `RuntimeError` on CPU-only machines — they do not affect CPU solvers.

## Regenerate

```bash
python3 spec/generators/generate_01_reconstruct.py   # 2571 per-algorithm specs
python3 spec/generators/generate_02_mismatch.py      # 168 mismatch specs
python3 spec/generators/generate_03_system_design.py # 168 system design specs
python3 spec/generators/generate_04_simulation.py    # 12 simulation specs
python3 spec/generate_specs.py                       # 168 quick-overview specs
```
