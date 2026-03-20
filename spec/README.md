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

## Get a Spec

### Without API key — auto-match preset spec

```bash
python3 spec/autospec.py "CT reconstruction low-dose"
python3 spec/autospec.py "MRI mismatch correction"
python3 spec/autospec.py "lensless system design"
python3 spec/autospec.py list                          # all 168 modalities
```

Returns the closest preset spec.md. Copy-paste the run button and execute.

### With API key — auto-design a custom spec

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 spec/autospec.py "low-dose CT reconstruction with TV regularization"
python3 spec/autospec.py "MRI mismatch + ESPIRiT sensitivity calibration" --save my_spec.md
```

LLM reads the relevant preset specs as context, designs a new custom spec, then
enters a refinement loop:

```
You: change iterations to 50
You: add visualization code
You: save          ← saves to file
You: quit
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

Specs marked **GPU** require a CUDA-capable GPU. On CPU-only machines they raise
`RuntimeError` — this does not affect **CPU** specs. Skip GPU specs on CPU machines.

## Regenerate Presets

```bash
python3 spec/generators/generate_01_reconstruct.py
python3 spec/generators/generate_02_mismatch.py
python3 spec/generators/generate_03_system_design.py
python3 spec/generators/generate_04_simulation.py
python3 spec/generate_specs.py
```
