# PLAN v4 — Test-First Report Contract (64 Modalities)

> **Status:** DRAFT v2 — awaiting user review before implementation.

---

## 0) Non-negotiable rules

| ID | Rule | Enforcement |
|----|------|-------------|
| R0.1 | **One modality at a time.** Implement + test + report + commit one modality fully before starting the next. | CLI guardrail (`pwm_cli next_modality` refuses unless previous DONE) |
| R0.2 | **Standard dataset required.** Each modality tested on a public benchmark (+ synthetic toy). Registered in `contrib/dataset_registry.yaml`. | CI test checks registry entry exists |
| R0.3 | **Both workflows mandatory.** W1 (prompt-driven simulate+reconstruct) AND W2 (operator correction with provided A + measured y). | Report contract CI test |
| R0.4 | **Save immediately.** After each modality: RunBundle exported (with node traces + PNGs), report written, scoreboard updated, commit created. | CLI guardrail |
| R0.5 | **Report contract is mandatory (flowchart required).** Every report must follow the template exactly, including ASCII flowchart matching Pattern A or Pattern B. CI fails if missing or malformed. | `tests/test_report_contract.py` |

---

## 1) Mandatory report template

### 1.0 Template file

Create `pwm/reports/templates/modality_report_template.md` with the exact structure below. Every modality report lives at `pwm/reports/<modality_key>.md` and must follow this template.

### 1.1 Required sections (exact headings — 11 total)

```markdown
# <Modality Display Name> — PWM Report

## Modality overview
- Modality key: `<key>`
- Category: <category>
- Forward model: <equation>
- Default solver: <solver_id>
- Pipeline linearity: <linear | nonlinear>

## Standard dataset
- Name: <dataset name>
- Source: <URL or citation>
- Size: <N images/volumes, dimensions>
- Registered in `dataset_registry.yaml` as `<dataset_id>`

## PWM pipeline flowchart (mandatory)

<Pattern A or Pattern B — see §1.2>

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength | — | — | — |
| 2 | <elem1> | <prim_id> | <transport\|interaction\|encoding\|transduction> | <param list> | <knob_name> | [lo, hi] | <prior_type> |
| … | … | … | … | … | … | … | … |
| N-1 | sensor | <sensor_prim> | sensor | QE, gain | qe_drift | [0.8, 1.0] | normal(0.9, 0.02) |
| N | noise | <noise_prim> | noise | read_sigma | read_sigma_drift | [0.001, 0.05] | log_normal |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for **one representative sample** from the standard dataset.
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (H, W) | float64 | [0.0, 1.0] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (H, W) | float64 | [0.0, 1.0] | `artifacts/trace/01_source.npy` |
| 2 | <elem1> | (M,) | float64 | [-0.5, 0.5] | `artifacts/trace/02_<elem1>.npy` |
| … | … | … | … | … | … |
| N-1 | sensor | (M,) | float64 | [0.0, 500.0] | `artifacts/trace/<N-1>_sensor.npy` |
| N | noise (y) | (M,) | float64 | [0.0, 510.3] | `artifacts/trace/<N>_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction
- **Prompt used:** `"<exact prompt text>"`
- **ExperimentSpec summary:**
  - modality: <key>
  - mode: simulate → invert
  - solver: <solver_id>
  - photon_budget: <value>
- **Mode S results (simulate y):**
  - y shape: <shape>
  - y range: [min, max]
  - SNR (if applicable): <value> dB
- **Mode I results (reconstruct x̂):**
  - x̂ shape: <shape>
  - Solver: <solver_id>, iterations: <N>
- **Dataset metrics:**
  | Metric | Value |
  |--------|-------|
  | PSNR   | XX.XX dB |
  | SSIM   | 0.XXXX |
  | NRMSE  | 0.XXXX |

## Workflow W2: Operator correction mode (measured y + operator A)
- **Operator definition:**
  - A_definition: <matrix | sparse | callable | LinearOperator>
  - A_extraction_method: <graph_stripped | provided | linearized>
  - Shape: <(M, N)>
  - A_sha256: <hash of A data, or "N/A" if callable>
  - Linearity: <linear | nonlinear>
  - Notes (if linearized): <"Jacobian at x₀ computed via ...", or N/A>
- **Mismatch specification:**
  - Mismatch type: <synthetic_injected | measured>
  - Parameters perturbed: <list with values>
  - Description: <what was changed and by how much>
- **Mode C fit results:**
  - Correction family: <Pre | Post | LowRank | Affine | FieldMap | Residual>
  - Parameters fitted: <list>
  - NLL before correction: <value>
  - NLL after correction: <value>
  - NLL decrease: <value> (<percentage>%)
- **Mode I recon using corrected operator A':**
  | Metric | A₀ (uncorrected) | A' (corrected) | Δ |
  |--------|-------------------|----------------|---|
  | PSNR   | XX.XX dB | XX.XX dB | +X.XX dB |
  | SSIM   | 0.XXXX | 0.XXXX | +0.XXXX |

## Test results summary

### Quick gate (pass/fail)
- [ ] W1 simulate: ✅ / ❌
- [ ] W1 reconstruct (PSNR ≥ threshold): ✅ / ❌
- [ ] W2 operator correction (NLL decreases): ✅ / ❌
- [ ] W2 corrected recon (beats uncorrected): ✅ / ❌
- [ ] Report contract (flowchart + all headings): ✅ / ❌
- [ ] Node-by-node trace saved (≥ N stages): ✅ / ❌
- [ ] RunBundle saved (with trace PNGs): ✅ / ❌
- [ ] Scoreboard updated: ✅ / ❌

### Full metrics
| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | XX.XX dB | ≥ YY dB | ✅/❌ |
| W1 SSIM | ssim | 0.XXXX | ≥ 0.YY | ✅/❌ |
| W1 NRMSE | nrmse | 0.XXXX | — | info |
| W2 NLL decrease | nll_decrease_pct | XX.X% | ≥ Y% | ✅/❌ |
| W2 PSNR gain | psnr_delta | +X.XX dB | > 0 | ✅/❌ |
| W2 SSIM gain | ssim_delta | +0.XXXX | > 0 | ✅/❌ |
| Trace stages | n_stages | N | ≥ 3 | ✅/❌ |
| Trace PNGs | n_pngs | N | ≥ 3 | ✅/❌ |
| W1 wall time | w1_seconds | X.XX s | — | info |
| W2 wall time | w2_seconds | X.XX s | — | info |

## Reproducibility
- **Seed:** <integer>
- **NumPy RNG state hash:** <sha256 hex[:16]>
- **PWM version:** <git SHA>
- **Python version:** <version>
- **Key package versions:** numpy=<v>, scipy=<v>
- **Platform:** <os + arch>
- **Deterministic reproduction command:**
  ```bash
  pwm_cli run --modality <key> --seed <seed> --mode simulate,invert,calibrate
  ```
- **Output hash (y):** <sha256 hex[:16]>
- **Output hash (x̂):** <sha256 hex[:16]>

## Saved artifacts
- RunBundle: `<path>`
- Report: `pwm/reports/<modality_key>.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `<runbundle>/artifacts/trace/*.npy`
- Node trace (.png): `<runbundle>/artifacts/trace/png/*.png`
- W2 operator metadata: `<runbundle>/artifacts/w2_operator_meta.json`

## Next actions
- <any follow-up items>
```

### 1.2 Flowchart patterns

The flowchart block under `## PWM pipeline flowchart (mandatory)` MUST match **exactly one** of the two patterns below.

#### Pattern A — Linear chain (most modalities)

For modalities where x enters at the top and flows straight through:

```
x (world)
  ↓
SourceNode: <source_primitive_id> — <description>
  ↓
Element 1 (subrole=<transport|interaction|encoding|transduction>): <primitive_id> — <description>
  ↓
Element 2 (subrole=<transport|interaction|encoding|transduction>): <primitive_id> — <description>
  ↓
...
  ↓
SensorNode: <sensor_primitive_id> — <description>
  ↓
NoiseNode: <noise_primitive_id> — <description>
  ↓
y
```

**Structural rules (Pattern A):**
1. First non-blank line: `x (world)`
2. Every stage separated by `↓`
3. Exactly one `SourceNode:` line — must be the first stage after `x (world)`
4. One or more `Element N (subrole=...)` lines — numbered sequentially starting from 1
5. Exactly one `SensorNode:` line — must come after all Elements
6. Exactly one `NoiseNode:` line — must come immediately after SensorNode
7. Last non-blank line: `y`

#### Pattern B — Branch entry (operator correction / nonlinear pipelines)

For modalities where the source illuminates but x (the object) enters at a specific interaction point — common in CT, ptychography, holography, MRI, etc.:

```
SourceNode: <source_primitive_id> — <description>
  ↓
  ↓ ← x (world) enters here
  ↓
Element 1 (subrole=interaction): <primitive_id> — <description: x modulates the beam/field>
  ↓
Element 2 (subrole=<transport|encoding|transduction>): <primitive_id> — <description>
  ↓
...
  ↓
SensorNode: <sensor_primitive_id> — <description>
  ↓
NoiseNode: <noise_primitive_id> — <description>
  ↓
y
```

**Structural rules (Pattern B):**
1. First non-blank line: `SourceNode:` (source emits carrier independently of x)
2. Contains exactly one branch-entry line matching: `← x (world) enters here`
3. The branch-entry line must appear **before** the first `Element` line
4. One or more `Element N (subrole=...)` lines — first Element must have `subrole=interaction` (where x modulates the carrier)
5. Exactly one `SensorNode:` line — must come after all Elements
6. Exactly one `NoiseNode:` line — must come immediately after SensorNode
7. Last non-blank line: `y`

#### Which pattern to use

| Pattern | When to use | Examples |
|---------|-------------|---------|
| A | x is the signal itself; source illuminates/excites x directly | widefield, confocal, SIM, SPC, CASSI, CACTI, lensless, lightsheet, FLIM, DOT, ultrasound, PET, SPECT, NeRF, 3DGS, matrix, panorama, light_field, integral, SEM, TEM, STEM, endoscopy, fundus, PALM/STORM, two_photon, STED, TIRF, polarization, doppler_ultrasound, elastography |
| B | Source emits carrier (X-ray/photon/spin/acoustic); x modulates it | CT, MRI, ptychography, holography, phase_retrieval, FPM, OCT, OCTA, photoacoustic, xray_radiography, fluoroscopy, mammography, DEXA, CBCT, angiography, fMRI, MRS, diffusion_MRI, electron_tomography, electron_diffraction, EBSD, EELS, electron_holography, SAR, sonar, neutron_tomo, proton_radiography, muon_tomo, ToF_camera, LiDAR, structured_light |

---

## 2) CI enforcement: `tests/test_report_contract.py`

### 2.1 What it checks

For every `pwm/reports/<modality_key>.md` file:

#### 2.1.1 Required headings (11 total, exact match)

1. `## Modality overview`
2. `## Standard dataset`
3. `## PWM pipeline flowchart (mandatory)`
4. `## Element inventory & mismatch parameters`
5. `## Node-by-node trace (one sample)`
6. `## Workflow W1: Prompt-driven simulation + reconstruction`
7. `## Workflow W2: Operator correction mode (measured y + operator A)`
8. `## Test results summary`
9. `## Reproducibility`
10. `## Saved artifacts`
11. `## Next actions`

All headings must be present. Test fails if any is missing.

#### 2.1.2 Flowchart structural validation (Pattern A OR Pattern B)

The validator extracts the flowchart block from `## PWM pipeline flowchart (mandatory)` and runs two checks. The flowchart **passes** if it matches **at least one** pattern:

**Pattern A check:**
```python
def check_pattern_a(lines: list[str]) -> bool:
    """Linear chain: x(world) → Source → Element(s) → Sensor → Noise → y"""
    # 1. First non-blank line == "x (world)"
    # 2. Exactly 1 line matching r"^SourceNode:"
    # 3. ≥1 lines matching r"^Element \d+ \(subrole=(transport|interaction|encoding|transduction)\):"
    # 4. Element numbers are sequential starting from 1
    # 5. Exactly 1 line matching r"^SensorNode:"
    # 6. Exactly 1 line matching r"^NoiseNode:"
    # 7. Order: x(world) < SourceNode < Element(s) < SensorNode < NoiseNode < y
    # 8. Last non-blank line == "y"
    # 9. ↓ arrows between every consecutive stage
```

**Pattern B check:**
```python
def check_pattern_b(lines: list[str]) -> bool:
    """Branch entry: Source → ← x enters → Element(s) → Sensor → Noise → y"""
    # 1. First non-blank line matches r"^SourceNode:"
    # 2. Exactly 1 line containing "← x (world) enters here"
    # 3. Branch-entry line comes BEFORE first Element line
    # 4. ≥1 lines matching r"^Element \d+ \(subrole=..."
    # 5. First Element has subrole=interaction
    # 6. Element numbers sequential from 1
    # 7. Exactly 1 SensorNode, exactly 1 NoiseNode
    # 8. Order: SourceNode < branch-entry < Element(s) < SensorNode < NoiseNode < y
    # 9. Last non-blank line == "y"
    # 10. ↓ arrows between consecutive stages
```

**Test logic:**
```python
def test_flowchart_structure(report_path):
    lines = extract_flowchart_block(report_path)
    assert check_pattern_a(lines) or check_pattern_b(lines), \
        f"Flowchart in {report_path} matches neither Pattern A nor Pattern B"
```

#### 2.1.3 Element inventory & mismatch parameters (non-empty)

- Section must contain a markdown table with at least 3 rows (header + separator + ≥1 data row)
- Table must have columns for: `node_id`, `primitive_id`, `subrole`
- At least one row must have a non-empty `mismatch knob` value (not `—`)

#### 2.1.4 Node-by-node trace validation

- Section must contain a markdown table
- Table must have ≥ 3 data rows (at least: input, one element, output)
- Every data row must have a non-empty `artifact_path` column
- Every `artifact_path` must match pattern `artifacts/trace/\d+_.*\.(npy|npz)`
- Section must mention PNG visualizations

```python
def test_node_trace(report_path):
    trace_table = extract_table_from_section(report_path, "## Node-by-node trace (one sample)")
    rows = parse_table_rows(trace_table)
    assert len(rows) >= 3, f"Trace must have ≥3 stages, got {len(rows)}"
    for row in rows:
        assert re.match(r"artifacts/trace/\d+_", row["artifact_path"].strip("`")), \
            f"Invalid artifact path: {row['artifact_path']}"
    assert "png" in trace_table.lower(), "Must reference PNG visualizations"
```

#### 2.1.5 Workflow W1 content checks

- Contains `**Prompt used:**` with non-empty prompt text
- Contains `Mode S results` and `Mode I results`
- Contains a `Dataset metrics` table with at least PSNR and SSIM

#### 2.1.6 Workflow W2 strengthened checks

- Contains `A_definition:` with value in {matrix, sparse, callable, LinearOperator}
- Contains `A_extraction_method:` with value in {graph_stripped, provided, linearized}
- If `A_extraction_method: linearized`, must also contain `Notes (if linearized):` with non-empty description
- Contains `A_sha256:` (hash or "N/A" for callables)
- Contains `NLL before correction:` and `NLL after correction:` with numeric values
- Contains a comparison table with A₀ vs A' columns

```python
VALID_A_DEFINITIONS = {"matrix", "sparse", "callable", "LinearOperator"}
VALID_A_METHODS = {"graph_stripped", "provided", "linearized"}

def test_w2_operator_spec(report_path):
    w2 = extract_section(report_path, "## Workflow W2")
    a_def = extract_field(w2, "A_definition")
    assert a_def in VALID_A_DEFINITIONS
    a_method = extract_field(w2, "A_extraction_method")
    assert a_method in VALID_A_METHODS
    assert "A_sha256:" in w2
    if a_method == "linearized":
        notes = extract_field(w2, "Notes (if linearized)")
        assert notes and notes != "N/A", \
            "linearized A must explain Jacobian computation"
    assert "NLL before correction:" in w2
    assert "NLL after correction:" in w2
```

#### 2.1.7 Test results summary — quick + full

- Must contain `### Quick gate` sub-heading with ≥ 6 checklist items
- Must contain `### Full metrics` sub-heading with a table containing ≥ 5 data rows
- Full metrics table must include columns: `Check`, `Metric`, `Value`, `Threshold`, `Status`

#### 2.1.8 Reproducibility section checks

- Must contain `Seed:` with integer value
- Must contain `PWM version:` with non-empty value
- Must contain `Output hash` with at least one sha256 fragment
- Must contain a `pwm_cli run` command block

#### 2.1.9 Dataset reference

- `## Standard dataset` section contains `dataset_registry.yaml` reference

### 2.2 Parametric test

```python
import glob
import pytest

def _collect_reports():
    return sorted(glob.glob("pwm/reports/*.md"))

@pytest.mark.parametrize("report_path", _collect_reports(), ids=lambda p: Path(p).stem)
class TestReportContract:
    def test_required_headings(self, report_path): ...
    def test_flowchart_pattern_a_or_b(self, report_path): ...
    def test_element_inventory(self, report_path): ...
    def test_node_trace(self, report_path): ...
    def test_w1_content(self, report_path): ...
    def test_w2_operator_spec(self, report_path): ...
    def test_w2_nll_fields(self, report_path): ...
    def test_results_summary_quick_and_full(self, report_path): ...
    def test_reproducibility(self, report_path): ...
    def test_saved_artifacts_section(self, report_path): ...
```

If no reports exist yet, the parametrize collects zero items and the test class is skipped (no false failures during bootstrap).

### 2.3 Scoreboard validation (in `tests/test_scoreboard.py`)

- Every modality listed has `status: DONE`, `PARTIAL`, or `PENDING`
- Every `DONE` modality has a corresponding `pwm/reports/<key>.md` that passes the contract
- Metrics fields (psnr, ssim, nll_decrease) are numeric when present
- `done_count + partial_count + pending_count == total_modalities == 64`

---

## 3) Dataset registry: `contrib/dataset_registry.yaml`

### 3.1 Schema

```yaml
version: "1.0"

datasets:
  <dataset_id>:
    display_name: "<Human-readable name>"
    modalities: [<modality_key>, ...]
    source_url: "<URL>"
    citation: "<BibTeX key or short ref>"
    license: "<license>"
    size:
      images: <N>
      dimensions: "<e.g. 256x256x31>"
    download:
      method: "<url | kaggle | zenodo | huggingface>"
      url: "<direct URL>"
      sha256: "<hash of archive>"
    notes: "<any special notes>"
```

### 3.2 Standard datasets for first 5 modalities

| # | Modality | Dataset | Source | Size |
|---|----------|---------|--------|------|
| 1 | SPC | Set11 | Standard CS test set | 11 images, 256×256 grayscale |
| 2 | CASSI | KAIST/MST 10 scenes | MST benchmark | 10 HSI scenes, 256×256×28 |
| 3 | CACTI | 6 grayscale videos | SCI benchmark | 6 videos, 256×256×8 frames |
| 4 | CT | LoDoPaB-CT | Zenodo (Leuschner et al.) | 35,820 slices, 362×362 |
| 5 | MRI | fastMRI knee singlecoil | NYU fastMRI | 973 volumes, 320×320 |

### 3.3 Standard datasets for remaining 59 modalities

| # | Modality | Dataset | Source |
|---|----------|---------|--------|
| 6 | widefield | BioSR (F-actin) | CNSB benchmark |
| 7 | widefield_lowdose | BioSR low-SNR subset | CNSB benchmark |
| 8 | confocal_livecell | DeepBacs fluorescence | zenodo |
| 9 | confocal_3d | CARE Tribolium | zenodo |
| 10 | sim | BioSR SIM dataset | CNSB benchmark |
| 11 | lensless | DiffuserCam | Waller Lab |
| 12 | lightsheet | OpenSPIM demo | openspim.org |
| 13 | ptychography | PtychoNN benchmark | ANL |
| 14 | holography | USC-SIPI off-axis | USC SIPI |
| 15 | nerf | Blender synthetic (NeRF) | NeRF paper |
| 16 | gaussian_splatting | Mip-NeRF 360 | Google Research |
| 17 | matrix | Shepp-Logan phantom (synth) | synthetic |
| 18 | panorama | DIV2K (multi-crop synth) | NTIRE |
| 19 | light_field | EPFL light-field dataset | EPFL |
| 20 | integral | Stanford Lytro | Stanford |
| 21 | phase_retrieval | CDP synthetic | synthetic (Candes) |
| 22 | flim | FLIM-FRET benchmark | IRB Barcelona |
| 23 | photoacoustic | IPASC phantom | IPASC consortium |
| 24 | oct | RETOUCH OCT | IEEE challenge |
| 25 | fpm | FPM LED-array benchmark | Zheng Lab |
| 26 | dot | TOAST/PMI phantom | UCL |
| 27 | xray_radiography | ChestX-ray14 | NIH CC |
| 28 | ultrasound | PICMUS challenge | IEEE IUS |
| 29 | pet | Brainweb PET | McConnell Brain Centre |
| 30 | spect | Zubal SPECT phantom | Yale |
| 31 | sem | SEM dataset (NIST) | NIST |
| 32 | tem | EMPIAR-10028 (ribosome) | EMPIAR |
| 33 | electron_tomography | EMPIAR-10045 | EMPIAR |
| 34 | stem | STEM-DL benchmark | ORNL |
| 35 | fluoroscopy | XCAT digital phantom | Duke |
| 36 | mammography | CBIS-DDSM | TCIA |
| 37 | dexa | Dual-energy synth phantom | synthetic |
| 38 | cbct | AAPM CBCT challenge | AAPM |
| 39 | angiography | XCAD angiography | Zenodo |
| 40 | doppler_ultrasound | Flow phantom (synth) | synthetic |
| 41 | elastography | CIRS phantom (synth) | synthetic |
| 42 | fmri | HCP resting-state fMRI | HCP / ConnectomeDB |
| 43 | mrs | ISMRMRD MRS phantom | ISMRMRD |
| 44 | diffusion_mri | HCP diffusion | HCP / ConnectomeDB |
| 45 | two_photon | Allen Brain two-photon | Allen Institute |
| 46 | sted | Abberior STED samples | Abberior |
| 47 | palm_storm | SMLM challenge 2016 | EPFL SMLM |
| 48 | tirf | TIRF-SIM benchmark | CNSB |
| 49 | polarization | Polarimetric test targets | synthetic |
| 50 | endoscopy | Hyper-Kvasir | SimulaRL |
| 51 | fundus | DRIVE retinal | DRIVE challenge |
| 52 | octa | ROSE-1 OCTA | ROSE challenge |
| 53 | tof_camera | NYU depth v2 (ToF subset) | NYU |
| 54 | lidar | KITTI LiDAR | KITTI benchmark |
| 55 | structured_light | Middlebury stereo (SL subset) | Middlebury |
| 56 | sar | MSTAR SAR | AFRL / Sandia |
| 57 | sonar | NSWC sonar benchmark | synthetic |
| 58 | electron_diffraction | 4D-STEM tutorial | py4DSTEM |
| 59 | ebsd | MTEX EBSD dataset | MTEX |
| 60 | eels | EELS-DB reference | EELS database |
| 61 | electron_holography | HolographyML | synthetic |
| 62 | neutron_tomo | NeXus neutron demo | NeXus |
| 63 | proton_radiography | pCT synth phantom | synthetic |
| 64 | muon_tomo | MuonTomography synth | synthetic |

---

## 4) Scoreboard: `pwm/reports/scoreboard.yaml`

### 4.1 Schema

```yaml
version: "1.0"
last_updated: "YYYY-MM-DDTHH:MM:SS"
total_modalities: 64
done_count: <N>
partial_count: <N>
pending_count: <N>

modalities:
  <modality_key>:
    status: "DONE" | "PARTIAL" | "PENDING"
    dataset_id: "<dataset_id>"
    w1_psnr: <float or null>
    w1_ssim: <float or null>
    w1_nrmse: <float or null>
    w2_nll_before: <float or null>
    w2_nll_after: <float or null>
    w2_nll_decrease_pct: <float or null>
    w2_psnr_corrected: <float or null>
    w2_a_extraction_method: <"graph_stripped" | "provided" | "linearized" | null>
    runbundle_path: "<path or null>"
    report_path: "<path or null>"
    commit_sha: "<sha or null>"
    completed_at: "<timestamp or null>"
```

### 4.2 Initial state

All 64 modalities start as `status: PENDING` with null metrics.

---

## 5) One-at-a-time guardrail

### 5.1 CLI command: `pwm_cli next_modality`

Implemented in `pwm_core/cli/modality_gate.py`:

```
pwm_cli next_modality
```

Checks for the *previous* modality (based on execution order):
1. RunBundle exists at the declared path
2. Report file exists at `pwm/reports/<prev_key>.md`
3. Report passes `test_report_contract.py` (runs pytest programmatically)
4. Scoreboard entry has `status: DONE`
5. A git commit exists that includes the report file

If ANY check fails → print which check failed → exit 1 (refuse).
If ALL pass → print the next modality key → exit 0.

### 5.2 Execution order

The full 64-modality execution order, grouped by priority tier:

**Tier 1 — Core compressive (1-5):**
1. `spc`
2. `cassi`
3. `cacti`
4. `ct`
5. `mri`

**Tier 2 — Microscopy fundamentals (6-13):**
6. `widefield`
7. `widefield_lowdose`
8. `confocal_livecell`
9. `confocal_3d`
10. `sim`
11. `lensless`
12. `lightsheet`
13. `flim`

**Tier 3 — Coherent imaging (14-18):**
14. `ptychography`
15. `holography`
16. `phase_retrieval`
17. `fpm`
18. `oct`

**Tier 4 — Medical imaging (19-28):**
19. `xray_radiography`
20. `ultrasound`
21. `photoacoustic`
22. `dot`
23. `pet`
24. `spect`
25. `fluoroscopy`
26. `mammography`
27. `dexa`
28. `cbct`

**Tier 5 — Neural rendering + computational (29-34):**
29. `nerf`
30. `gaussian_splatting`
31. `matrix`
32. `panorama`
33. `light_field`
34. `integral`

**Tier 6 — Electron microscopy (35-41):**
35. `sem`
36. `tem`
37. `stem`
38. `electron_tomography`
39. `electron_diffraction`
40. `ebsd`
41. `eels`

**Tier 7 — Advanced medical (42-47):**
42. `angiography`
43. `doppler_ultrasound`
44. `elastography`
45. `fmri`
46. `mrs`
47. `diffusion_mri`

**Tier 8 — Advanced microscopy (48-52):**
48. `two_photon`
49. `sted`
50. `palm_storm`
51. `tirf`
52. `polarization`

**Tier 9 — Clinical optics + depth (53-58):**
53. `endoscopy`
54. `fundus`
55. `octa`
56. `tof_camera`
57. `lidar`
58. `structured_light`

**Tier 10 — Remote sensing + exotic (59-64):**
59. `sar`
60. `sonar`
61. `electron_holography`
62. `neutron_tomo`
63. `proton_radiography`
64. `muon_tomo`

---

## 6) Per-modality implementation workflow

For each modality in order:

### Step 1: Dataset registration
- Add entry to `contrib/dataset_registry.yaml`
- Download or generate synthetic data
- Verify data loads correctly

### Step 2: W1 — Prompt-driven simulate + reconstruct
- Build/verify v2 graph template (canonical chain)
- Run Mode S (simulate): `GraphExecutor.execute(x=x_gt, config=ExecutionConfig(mode=simulate))`
- **Capture node-by-node trace** (intermediate tensor at each node, save as .npy)
- **Generate PNG visualization** for each intermediate state
- Run Mode I (invert): `GraphExecutor.execute(y=y, config=ExecutionConfig(mode=invert))`
- Compute metrics (PSNR, SSIM, NRMSE or modality-specific)
- Record results

### Step 3: W2 — Operator correction mode
- **Extract operator A** using one of three methods:
  - `graph_stripped`: strip noise node from compiled graph, use as linear operator
  - `provided`: user/test supplies explicit A (matrix, sparse, or LinearOperator)
  - `linearized`: for nonlinear pipelines, compute Jacobian at operating point x₀
- **Compute and record A metadata:**
  - `A_definition`: matrix | sparse | callable | LinearOperator
  - `A_extraction_method`: graph_stripped | provided | linearized
  - `A_sha256`: SHA-256 of A's data (or "N/A" for pure callables)
  - For `linearized`: document Jacobian computation method and operating point
- Inject synthetic mismatch (or use measured mismatch)
- Run Mode C (calibrate): `GraphExecutor.execute(x=x_gt, y=y_mismatch, config=ExecutionConfig(mode=calibrate))`
- Verify NLL decreases
- Run Mode I with corrected A' and compare to uncorrected A₀
- Record results + save `w2_operator_meta.json` in RunBundle

### Step 4: Export RunBundle + report
- Export RunBundle via `write_runbundle_skeleton` (extended layout, see §8)
- Save standard artifacts (y, x_hat, x_hat_corrected, metrics)
- **Save node-by-node trace artifacts** (`.npy` + `.png` per node)
- **Save W2 operator metadata** (`w2_operator_meta.json`)
- Write report `pwm/reports/<modality_key>.md` from template
  - Fill Pattern A or Pattern B flowchart
  - Fill element inventory table
  - Fill node-by-node trace table (referencing saved artifact paths)
  - Fill W1 and W2 results with A_definition / A_extraction_method
  - Fill quick gate + full metrics in test results summary
  - Fill reproducibility section (seed, hashes, versions)
- Update scoreboard `pwm/reports/scoreboard.yaml`

### Step 5: Validate + commit
- Run `pytest tests/test_report_contract.py -k <modality_key>` — must pass
- Run `pytest tests/test_modality_<key>.py` — must pass
- Commit: `git commit -m "Complete <modality_key>: W1+W2, report, RunBundle"`

### Step 6: Gate check
- `pwm_cli next_modality` must return exit 0 before proceeding

---

## 7) Per-modality acceptance tests

### 7.1 Test file per modality

Each modality gets tests in `tests/test_modality_<key>.py`:

```python
class TestW1Simulate:
    def test_mode_s_produces_finite_y(self): ...
    def test_mode_s_y_shape_correct(self): ...
    def test_mode_s_noise_present(self): ...
    def test_mode_s_trace_saved(self): ...       # NEW: trace artifacts exist
    def test_mode_s_trace_pngs_saved(self): ...  # NEW: PNG visualizations exist

class TestW1Reconstruct:
    def test_mode_i_produces_finite_xhat(self): ...
    def test_mode_i_psnr_above_threshold(self): ...
    def test_mode_i_ssim_above_threshold(self): ...

class TestW2OperatorCorrection:
    def test_mode_c_nll_decreases(self): ...
    def test_corrected_recon_better_than_uncorrected(self): ...
    def test_correction_family_registered(self): ...
    def test_a_extraction_method_valid(self): ...    # NEW
    def test_a_metadata_saved(self): ...             # NEW: w2_operator_meta.json exists
    def test_a_sha256_present(self): ...             # NEW: hash recorded

class TestReportContract:
    def test_report_exists(self): ...
    def test_report_passes_contract(self): ...
    def test_flowchart_matches_pattern(self): ...    # NEW: Pattern A or B
    def test_element_inventory_present(self): ...    # NEW
    def test_node_trace_present(self): ...           # NEW

class TestRunBundle:
    def test_runbundle_saved(self): ...
    def test_artifacts_present(self): ...
    def test_trace_npy_files_present(self): ...      # NEW
    def test_trace_png_files_present(self): ...      # NEW
    def test_w2_operator_meta_present(self): ...     # NEW
```

### 7.2 Metric thresholds (per-tier)

| Tier | Modalities | Min PSNR | Min SSIM | Min NLL decrease |
|------|-----------|----------|----------|-----------------|
| 1 (compressive) | spc, cassi, cacti, ct, mri | 25 dB | 0.80 | 5% |
| 2 (microscopy) | widefield…flim | 28 dB | 0.85 | 5% |
| 3 (coherent) | ptychography…oct | 22 dB | 0.75 | 3% |
| 4 (medical) | xray…cbct | 25 dB | 0.80 | 5% |
| 5 (neural) | nerf…integral | 22 dB | 0.75 | 3% |
| 6 (electron) | sem…eels | 20 dB | 0.70 | 3% |
| 7 (adv medical) | angiography…diffusion_mri | 22 dB | 0.75 | 3% |
| 8 (adv micro) | two_photon…polarization | 22 dB | 0.75 | 3% |
| 9 (clinical+depth) | endoscopy…structured_light | 20 dB | 0.70 | 3% |
| 10 (exotic) | sar…muon_tomo | 18 dB | 0.65 | 2% |

---

## 8) RunBundle extended layout

### 8.1 Directory structure (updated)

```
run_{spec_id}_{uuid}/
  artifacts/
    images/                     # Final reconstructions
    trace/                      # NEW: node-by-node intermediate states
      00_input_x.npy
      01_source.npy
      02_<element1>.npy
      ...
      <N-1>_sensor.npy
      <N>_noise_y.npy
      png/                      # NEW: PNG visualizations of each stage
        00_input_x.png
        01_source.png
        02_<element1>.png
        ...
    w2_operator_meta.json       # NEW: W2 operator metadata
  internal_state/
  agents/
  logs/
```

### 8.2 Node trace capture

During Mode S (simulate), the `GraphExecutor` records intermediate outputs:

```python
# In GraphExecutor._simulate():
trace = {}
for i, (node_id, prim) in enumerate(self._graph.forward_plan):
    # ... existing forward pass ...
    trace[f"{i:02d}_{node_id}"] = current_output.copy()
return ExecutionResult(..., diagnostics={"trace": trace})
```

The RunBundle writer saves each trace entry as:
- `.npy` file: `artifacts/trace/{key}.npy`
- `.png` file: `artifacts/trace/png/{key}.png` (auto-generated visualization)

PNG generation rules:
- 2D array → grayscale image (normalized to [0,255])
- 3D array (H,W,C) → RGB or first 3 channels
- 1D array → line plot
- Complex array → magnitude + phase side-by-side
- 4D+ → slice through first 2 spatial dims

### 8.3 W2 operator metadata file

`artifacts/w2_operator_meta.json`:
```json
{
  "a_definition": "matrix",
  "a_extraction_method": "graph_stripped",
  "a_shape": [614, 4096],
  "a_dtype": "float64",
  "a_sha256": "a1b2c3d4...",
  "a_nnz": 614000,
  "a_sparsity": 0.0,
  "linearity": "linear",
  "linearization_notes": null,
  "mismatch_type": "synthetic_injected",
  "mismatch_params": {"gain_drift": 0.05, "offset_shift": 0.02},
  "correction_family": "Affine",
  "nll_before": 1234.56,
  "nll_after": 987.65,
  "nll_decrease_pct": 20.0,
  "timestamp": "2026-02-12T10:30:00"
}
```

When `A_extraction_method == "provided"`:
- `a_sha256` is computed from the provided matrix/array data
- If A is a dense matrix: `sha256(A.tobytes())`
- If A is sparse: `sha256(A.data.tobytes() + A.indices.tobytes() + A.indptr.tobytes())`
- If A is a callable: `a_sha256 = "N/A (callable)"`, but `a_definition` is set to `"callable"`

When `A_extraction_method == "linearized"`:
- `linearization_notes` must be non-null, describing the Jacobian computation
- Example: `"Jacobian computed at x₀ = dataset mean via finite differences, ε=1e-6"`

---

## 9) New files to create

| # | File | Purpose | Lines (est) |
|---|------|---------|-------------|
| 1 | `pwm/reports/templates/modality_report_template.md` | Master report template (Pattern A + B) | ~120 |
| 2 | `tests/test_report_contract.py` | CI: validate reports (headings, flowchart pattern, trace, W2 operator) | ~350 |
| 3 | `contrib/dataset_registry.yaml` | Standard dataset registry | ~500 |
| 4 | `pwm/reports/scoreboard.yaml` | Modality completion scoreboard | ~400 |
| 5 | `pwm_core/cli/modality_gate.py` | CLI guardrail for one-at-a-time | ~150 |
| 6 | `tests/test_scoreboard.py` | CI: validate scoreboard consistency | ~100 |
| 7 | 64× `pwm/reports/<modality_key>.md` | Per-modality reports (created as each is done) | ~200 each |
| 8 | 64× `tests/test_modality_<key>.py` | Per-modality acceptance tests | ~150 each |

### Files to modify

| # | File | Changes |
|---|------|---------|
| 1 | `pwm_core/api/prompt_parser.py` | Add 30 missing modality keywords (v4+ modalities) |
| 2 | `pwm_core/cli/main.py` | Register `next_modality` CLI command |
| 3 | `pwm_core/graph/executor.py` | Add trace capture in `_simulate()`, expose trace in ExecutionResult |
| 4 | `pwm_core/core/runbundle/writer.py` | Add `trace/` and `trace/png/` dirs, save `.npy`+`.png`, save `w2_operator_meta.json` |
| 5 | `pwm_core/core/runbundle/artifacts.py` | Add `save_trace()` and `save_operator_meta()` helpers |

---

## 10) Implementation phases

### Phase A: Infrastructure (before any modality)
1. Create report template (`pwm/reports/templates/modality_report_template.md`) with Pattern A + B
2. Create `tests/test_report_contract.py` — full validation (headings, flowchart patterns, trace, W2 operator)
3. Create `contrib/dataset_registry.yaml` — empty skeleton
4. Create `pwm/reports/scoreboard.yaml` — all 64 as PENDING
5. Create `pwm_core/cli/modality_gate.py` — guardrail logic
6. Create `tests/test_scoreboard.py`
7. Update `prompt_parser.py` with all 64 modality keywords
8. Update `executor.py` — add trace capture to `_simulate()`
9. Update `runbundle/writer.py` — extended layout with trace + W2 metadata dirs
10. Update `runbundle/artifacts.py` — `save_trace()`, `save_operator_meta()`, PNG generator
11. Commit: "Add report contract infrastructure with Pattern A/B flowcharts, trace capture, W2 operator metadata"

### Phase B: First 5 modalities (one at a time, in order)
12. SPC → complete W1+W2+report+bundle+commit
13. CASSI → complete W1+W2+report+bundle+commit
14. CACTI → complete W1+W2+report+bundle+commit
15. CT → complete W1+W2+report+bundle+commit
16. MRI → complete W1+W2+report+bundle+commit

### Phase C: Remaining 59 modalities (one at a time, following §5.2 order)
17–75. Each modality follows the same Step 1-6 workflow from §6.

---

## 11) Definition of DONE

A modality is **✅ DONE** only if ALL of:
- [ ] Standard dataset registered in `dataset_registry.yaml` and used
- [ ] W1 passes: Mode S (simulate) + Mode I (reconstruct) with metrics above threshold
- [ ] W1 node-by-node trace saved (`.npy` + `.png` for each stage, ≥ 3 stages)
- [ ] W2 passes: operator correction improves NLL + corrected recon beats uncorrected
- [ ] W2 operator metadata saved (`A_definition`, `A_extraction_method`, `A_sha256`)
- [ ] Report at `pwm/reports/<key>.md` passes `test_report_contract.py`:
  - All 11 required headings present
  - Flowchart matches Pattern A or Pattern B structurally
  - Element inventory table present with mismatch parameters
  - Node-by-node trace table present with ≥ 3 stages and artifact paths
  - W2 section includes A_definition + A_extraction_method
  - Quick gate + full metrics tables in test results summary
  - Reproducibility section with seed, hashes, versions
- [ ] RunBundle exported with all artifacts (trace + PNGs + W2 meta)
- [ ] Scoreboard updated with metrics + status DONE
- [ ] Git commit exists containing report + artifacts

Otherwise it is **🔶 PARTIAL**.

---

## 12) Example: CT report (Pattern B)

```markdown
# Computed Tomography (CT) — PWM Report

## Modality overview
- Modality key: `ct`
- Category: medical
- Forward model: y = R·μ + n (R = Radon transform, μ = attenuation map)
- Default solver: fbp
- Pipeline linearity: linear

## Standard dataset
- Name: LoDoPaB-CT
- Source: Zenodo (Leuschner et al. 2021)
- Size: 35,820 slices, 362×362
- Registered in `dataset_registry.yaml` as `lodopab_ct`

## PWM pipeline flowchart (mandatory)

SourceNode: xray_source — Polychromatic X-ray tube (80 kVp)
  ↓
  ↓ ← x (world) enters here
  ↓
Element 1 (subrole=interaction): beer_lambert — X-ray attenuation through object μ(r)
  ↓
Element 2 (subrole=encoding): ct_radon — Radon transform (parallel beam, 180 angles)
  ↓
SensorNode: photon_sensor — Flat-panel detector (QE=0.85, gain=1.0)
  ↓
NoiseNode: poisson_only_sensor — Photon counting noise (I₀=10⁴ photons/ray)
  ↓
y

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | xray_source | source | keV=80, strength=1e4 | — | — | — |
| 2 | attenuation | beer_lambert | interaction | — | — | — | — |
| 3 | radon | ct_radon | encoding | n_angles=180 | angle_offset | [-2.0, 2.0] deg | normal(0, 0.5) |
| 4 | sensor | photon_sensor | sensor | QE=0.85, gain=1.0 | qe_drift | [0.7, 1.0] | normal(0.85, 0.03) |
| 5 | noise | poisson_only_sensor | noise | I0=1e4 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (362, 362) | float64 | [0.0, 0.04] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (362, 362) | float64 | [0.0, 10000.0] | `artifacts/trace/01_source.npy` |
| 2 | attenuation | (362, 362) | float64 | [0.0, 1.0] | `artifacts/trace/02_attenuation.npy` |
| 3 | radon | (180, 512) | float64 | [0.0, 15.2] | `artifacts/trace/03_radon.npy` |
| 4 | sensor | (180, 512) | float64 | [0.0, 12920.0] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (180, 512) | float64 | [0.0, 13105.3] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction
...

## Workflow W2: Operator correction mode (measured y + operator A)
- **Operator definition:**
  - A_definition: sparse
  - A_extraction_method: graph_stripped
  - Shape: (92160, 131044)
  - A_sha256: 7f3a2b1c...
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: angle_offset=+1.5 deg, qe_drift=-0.05
  - Description: Gantry angular miscalibration + detector sensitivity drift
- **Mode C fit results:**
  - Correction family: Affine
  - Parameters fitted: [angle_offset, qe_drift]
  - NLL before correction: 45230.1
  - NLL after correction: 38120.5
  - NLL decrease: 7109.6 (15.7%)
- **Mode I recon using corrected operator A':**
  | Metric | A₀ (uncorrected) | A' (corrected) | Δ |
  |--------|-------------------|----------------|---|
  | PSNR   | 27.3 dB | 31.1 dB | +3.8 dB |
  | SSIM   | 0.82 | 0.91 | +0.09 |

## Test results summary

### Quick gate
- [x] W1 simulate: ✅
- [x] W1 reconstruct (PSNR ≥ 25 dB): ✅
- [x] W2 operator correction (NLL decreases): ✅
- [x] W2 corrected recon (beats uncorrected): ✅
- [x] Report contract (flowchart Pattern B + all headings): ✅
- [x] Node-by-node trace saved (6 stages ≥ 3): ✅
- [x] RunBundle saved (with trace PNGs): ✅
- [x] Scoreboard updated: ✅

### Full metrics
| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 31.1 dB | ≥ 25 dB | ✅ |
| W1 SSIM | ssim | 0.91 | ≥ 0.80 | ✅ |
| W1 NRMSE | nrmse | 0.028 | — | info |
| W2 NLL decrease | nll_decrease_pct | 15.7% | ≥ 5% | ✅ |
| W2 PSNR gain | psnr_delta | +3.8 dB | > 0 | ✅ |
| W2 SSIM gain | ssim_delta | +0.09 | > 0 | ✅ |
| Trace stages | n_stages | 6 | ≥ 3 | ✅ |
| Trace PNGs | n_pngs | 6 | ≥ 3 | ✅ |
| W1 wall time | w1_seconds | 4.2 s | — | info |
| W2 wall time | w2_seconds | 12.8 s | — | info |

## Reproducibility
- **Seed:** 42
- **NumPy RNG state hash:** a1b2c3d4e5f6g7h8
- **PWM version:** f2d6da5
- **Python version:** 3.11.7
- **Key package versions:** numpy=1.26.4, scipy=1.12.0
- **Platform:** linux x86_64
- **Deterministic reproduction command:**
  ```bash
  pwm_cli run --modality ct --seed 42 --mode simulate,invert,calibrate
  ```
- **Output hash (y):** 3e4f5a6b7c8d9e0f
- **Output hash (x̂):** 1a2b3c4d5e6f7g8h

## Saved artifacts
- RunBundle: `runs/run_ct_exp_a1b2c3d4/`
- Report: `pwm/reports/ct.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_ct_exp_a1b2c3d4/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_ct_exp_a1b2c3d4/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_ct_exp_a1b2c3d4/artifacts/w2_operator_meta.json`

## Next actions
- None (modality complete)
```

---

## 13) Example: SPC report (Pattern A)

```markdown
# Single-Pixel Camera (SPC) — PWM Report

## Modality overview
- Modality key: `spc`
- Category: compressive
- Forward model: y = Φx + n (Φ = random binary mask, 15% sampling)
- Default solver: fista
- Pipeline linearity: linear

## Standard dataset
- Name: Set11
- Source: Standard compressive sensing test set (11 natural images)
- Size: 11 images, 256×256 grayscale
- Registered in `dataset_registry.yaml` as `set11`

## PWM pipeline flowchart (mandatory)

x (world)
  ↓
SourceNode: photon_source — Illumination source (strength=1.0)
  ↓
Element 1 (subrole=encoding): random_mask — Binary random measurement matrix (15% sampling rate)
  ↓
SensorNode: photon_sensor — Photodetector (QE=0.9, gain=1.0)
  ↓
NoiseNode: poisson_gaussian_sensor — Shot noise + read noise (σ_read=0.01)
  ↓
y

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | measure | random_mask | encoding | seed=42, rate=0.15 | mask_noise | [0, 0.1] | uniform |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain_drift | [0.8, 1.2] | normal(1.0, 0.05) |
| 4 | noise | poisson_gaussian_sensor | noise | σ_read=0.01 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (256, 256) | float64 | [0.0, 1.0] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (256, 256) | float64 | [0.0, 1.0] | `artifacts/trace/01_source.npy` |
| 2 | measure | (614,) | float64 | [-0.3, 0.8] | `artifacts/trace/02_measure.npy` |
| 3 | sensor | (614,) | float64 | [0.0, 480.2] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (614,) | float64 | [0.0, 495.1] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction
- **Prompt used:** `"Simulate single-pixel camera measurement of a natural image and reconstruct"`
- **ExperimentSpec summary:**
  - modality: spc
  - mode: simulate → invert
  - solver: fista
  - photon_budget: 10000
- **Mode S results (simulate y):**
  - y shape: (614,)
  - y range: [0.002, 0.998]
  - SNR: 25.3 dB
- **Mode I results (reconstruct x̂):**
  - x̂ shape: (256, 256)
  - Solver: fista, iterations: 100
- **Dataset metrics:**
  | Metric | Value |
  |--------|-------|
  | PSNR   | 28.45 dB |
  | SSIM   | 0.8723 |
  | NRMSE  | 0.0384 |

## Workflow W2: Operator correction mode (measured y + operator A)
- **Operator definition:**
  - A_definition: matrix
  - A_extraction_method: graph_stripped
  - Shape: (614, 65536)
  - A_sha256: e4f5a6b7...
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain_drift=+0.05
  - Description: Detector gain miscalibration (+5%)
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: [gain_drift]
  - NLL before correction: 8920.3
  - NLL after correction: 7650.1
  - NLL decrease: 1270.2 (14.2%)
- **Mode I recon using corrected operator A':**
  | Metric | A₀ (uncorrected) | A' (corrected) | Δ |
  |--------|-------------------|----------------|---|
  | PSNR   | 26.1 dB | 28.45 dB | +2.35 dB |
  | SSIM   | 0.83 | 0.8723 | +0.0423 |

## Test results summary

### Quick gate
- [x] W1 simulate: ✅
- [x] W1 reconstruct (PSNR ≥ 25 dB): ✅
- [x] W2 operator correction (NLL decreases): ✅
- [x] W2 corrected recon (beats uncorrected): ✅
- [x] Report contract (flowchart Pattern A + all headings): ✅
- [x] Node-by-node trace saved (5 stages ≥ 3): ✅
- [x] RunBundle saved (with trace PNGs): ✅
- [x] Scoreboard updated: ✅

### Full metrics
| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 28.45 dB | ≥ 25 dB | ✅ |
| W1 SSIM | ssim | 0.8723 | ≥ 0.80 | ✅ |
| W1 NRMSE | nrmse | 0.0384 | — | info |
| W2 NLL decrease | nll_decrease_pct | 14.2% | ≥ 5% | ✅ |
| W2 PSNR gain | psnr_delta | +2.35 dB | > 0 | ✅ |
| W2 SSIM gain | ssim_delta | +0.0423 | > 0 | ✅ |
| Trace stages | n_stages | 5 | ≥ 3 | ✅ |
| Trace PNGs | n_pngs | 5 | ≥ 3 | ✅ |
| W1 wall time | w1_seconds | 2.1 s | — | info |
| W2 wall time | w2_seconds | 5.3 s | — | info |

## Reproducibility
- **Seed:** 42
- **NumPy RNG state hash:** 9a8b7c6d5e4f3a2b
- **PWM version:** f2d6da5
- **Python version:** 3.11.7
- **Key package versions:** numpy=1.26.4, scipy=1.12.0
- **Platform:** linux x86_64
- **Deterministic reproduction command:**
  ```bash
  pwm_cli run --modality spc --seed 42 --mode simulate,invert,calibrate
  ```
- **Output hash (y):** b2c3d4e5f6a7b8c9
- **Output hash (x̂):** d4e5f6a7b8c9d0e1

## Saved artifacts
- RunBundle: `runs/run_spc_exp_f1a2b3c4/`
- Report: `pwm/reports/spc.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_spc_exp_f1a2b3c4/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_spc_exp_f1a2b3c4/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_spc_exp_f1a2b3c4/artifacts/w2_operator_meta.json`

## Next actions
- None (modality complete)
```

---

## 14) Summary

| What | Count / Detail |
|------|----------------|
| Total modalities | 64 |
| Required report headings | 11 (was 8) |
| Flowchart patterns | 2 (Pattern A linear + Pattern B branch-entry) |
| New infrastructure files | 8 |
| Files to modify | 5 |
| Per-modality deliverables | report + RunBundle (with trace+PNGs+W2 meta) + acceptance test + scoreboard entry + commit |
| Mandatory flowchart | Pattern A or B, structurally validated (not just line presence) |
| W2 operator fields | A_definition, A_extraction_method, A_sha256, linearity, linearization_notes |
| RunBundle extras | intermediate node .npy + .png per stage, w2_operator_meta.json |
| CI tests | `test_report_contract.py` (structural) + `test_scoreboard.py` + 64× `test_modality_<key>.py` |
| Workflows per modality | 2 (W1 prompt-driven + W2 operator correction) |
| Test results format | Quick gate (pass/fail checklist) + Full metrics (table with thresholds) |
| Guardrail | `pwm_cli next_modality` blocks until previous DONE |
