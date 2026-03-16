"""Update standard_state.md with GCS download instructions."""
from pathlib import Path

STATE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/standard_state.md")

text = STATE.read_text(encoding="utf-8")

# Add GCS section after the header line
gcs_section = """
## Cloud Storage (GCS)

Standard datasets are stored in Google Cloud Storage (NOT in the GitHub repo).

- **GCS bucket:** `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/`
- **Total size:** ~1 GB across 170 modalities (~5750 files)
- **File types:** `*.h5` (data), `*.json` (metadata), `images/*.png` (previews)

### Setup on a new server

```bash
# 1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
# 2. Authenticate:
gcloud auth login

# 3. Download all modalities (~1 GB):
python scripts/download_standard_from_gcs.py

# 4. Download specific modalities only:
python scripts/download_standard_from_gcs.py --modality ct,mri,pet

# 5. List available modalities:
python scripts/download_standard_from_gcs.py --list
```

### Upload (maintainer only)

```bash
gcloud auth login
python scripts/upload_standard_to_gcs.py
```
"""

# Insert after first header block (line 3)
lines = text.split("\n")
insert_idx = 3  # After "Last updated" line
new_lines = lines[:insert_idx] + gcs_section.split("\n") + lines[insert_idx:]
STATE.write_text("\n".join(new_lines), encoding="utf-8")
print("Updated standard_state.md with GCS instructions")
