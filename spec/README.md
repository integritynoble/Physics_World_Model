# PWM Spec Database

168 imaging modalities — each spec covers reconstruct, mismatch correction, system design, and simulation.

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

## Usage

```bash
# Find spec by keyword (no API key needed)
python3 spec/keyword_match.py "CT reconstruction"
python3 spec/keyword_match.py "MRI mismatch correction"
python3 spec/keyword_match.py "lensless imaging system"
python3 spec/keyword_match.py "list"              # all modalities
```

Then open the spec file shown, copy the **Run** code block, and execute.

## Spec Files

One `.md` per modality in `spec/`. Each contains:
- Input format + benchmark GCS path
- All algorithms with PSNR reference (CPU and GPU)
- Run button (copy-paste Python)
- Mismatch correction hint
- Links to papers

## Use Cases

| # | Description | How |
|---|-------------|-----|
| 1 | Reconstruct | `run_solver('key', y)` |
| 2 | Mismatch correction | `cfg={'calibrate': True}` |
| 3 | System design | See `papers/system_design/outputs/` |
| 4 | Physics simulation | See `papers/universal_simulation/benchmark/` |

## GPU Note

GPU solvers raise `RuntimeError` without a GPU — CPU solvers always work.
