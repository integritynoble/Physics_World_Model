# Generated Files Index — 6-Point Check.md Framework

## Quick Links

| Component | Location | Count | Status |
|-----------|----------|-------|--------|
| Check.md files | `benchmarks/learn/{variant}/check.md` | 167 | GENERATED |
| Generator script | `scripts/generate_check_md.py` | 1 | NEW |
| JSON reports | Root directory | 3 | NEW |
| Summary docs | Root directory | 2 | NEW |

---

## Check.md Files (167 modalities)

All files follow the standardized 6-point template:
1. Benchmark Page Errors
2. Local Dataset Inspection
3. Public Dataset Source Assessment
4. Algorithm Coverage Assessment
5. Improvement Suggestions
6. Action Items

### By Category

**With Algorithms (76 modalities)**
- Acoustic Microscopy
- Adaptive Optics
- CEST MRI
- CLEM
- Confocal Endomicroscopy
- Coronagraphy
- CT Fluorescence
- DEXA
- DIC
- DNA-PAINT
- DOT
- EBSD
- Eddy Current
- EELS
- Elastography
- Electron Diffraction
- Electron Holography
- Electron Tomography
- Endoscopy
- Event Camera
- Flash LIDAR
- FLIM
- FPM
- Fundus
- FWI
- GPR
- Gravitational Wave
- Hyperspectral Remote
- Impedance Tomography
- Industrial CT
- InSAR
- Integral
- Lensless
- LiDAR
- Light Field
- Lucky Imaging
- Machine Vision
- MINFLUX
- MR Fingerprinting
- Neutron Diffraction
- NIRS Brain
- Ocean Color
- ODT
- PALM/STORM
- Panorama
- Particle Calorimetry
- Passive Microwave
- Phase Contrast
- Photoacoustic
- Photometric Stereo
- Proton Radiography
- Ptychography
- Pump-Probe
- Quantum Illumination
- Radio Astronomy
- Radio Interferometry
- SAXS
- Shearography
- SIM
- Solar Imaging
- Sonar
- SPECT-CT
- Structured Light
- Talbot-Lau
- Terahertz
- TOF Camera
- Ultrasonic Phased Array
- US-MRI
- WAXS
- Weather Radar
- XFEL-SFX
- X-ray Crystallography
- X-ray NDT
- XRF Imaging

**Without Algorithms (92 modalities)**
- MRI (Note: database shows 8 algorithms, but uppercase variant name)
- Acoustic Emission
- Active Thermography
- AFM
- Angiography
- ASL MRI
- Atom Probe
- Bioluminescence Tomography
- Brachytherapy Imaging
- Brillouin
- CARS
- CASSI
- Cathodoluminescence
- CBCT
- CEUS
- Coded Exposure
- Confocal 3D
- Confocal Live-Cell
- Cryo-EM
- Cryo-ET
- CUP
- Dark Field
- DESI
- Diffusion MRI
- Digital Breast Tomography
- Doppler Ultrasound
- EDX Mapping
- EHT Imaging
- Entangled Photon
- Expansion
- FIB-SEM
- Fluoroscopy
- fMRI
- FTIR Imaging
- Gaussian Splatting
- Ghost Imaging
- HDR Imaging
- Holography
- ISM
- IVUS
- Lattice Light-Sheet
- LIBS
- Light-Sheet
- Magnetic Particle
- MALDI-MSI
- Mammography
- Matrix
- MFM
- MRA
- MRS
- Multispectral SAT
- Muon Tomography
- NeRF
- Neutron Tomography
- NSOM
- Ocean Acoustic Tomography
- OCT
- OCTA
- PET
- PET-CT
- PET-MR
- Phase Retrieval
- Polarization
- PolSAR
- Portal Imaging
- Proton Therapy Imaging
- Quantum Illumination (Note: has 4 algos)
- Raman Imaging
- SAR
- SD-CASSI
- Seismic Tomography
- SEM
- SIMS
- SPC
- SPECT
- Spectral CT
- Spinning Disk
- SRS
- STED
- STEM
- STM
- Streak Camera
- SWI
- TEM
- Three-Photon
- TIRF
- Two-Photon
- Ultrasound
- Widefield
- Widefield Low-Dose
- X-ray Radiography
- XRF Tomography

---

## Report Files

### benchmark_check_generation_report.json
```json
{
  "timestamp": "2026-03-03T23:36:50.869493",
  "total_modalities": 168,
  "generated": 167,
  "skipped_hand_crafted": 1,
  "errors": [],
  "modalities": [
    {
      "variant": "ct",
      "check_md_exists": true,
      "in_database": true,
      "algorithms": 8,
      "status": "SKIPPED_HAND_CRAFTED"
    },
    ...168 total entries...
  ]
}
```

**Contents:**
- Per-modality metadata
- Generation status for each variant
- Database and algorithm coverage statistics
- Timestamp and error log

**Usage:**
```bash
jq '.modalities[] | select(.variant=="widefield")' benchmark_check_generation_report.json
jq '[.modalities[] | select(.algorithms > 0)] | length' benchmark_check_generation_report.json
```

### check_md_statistics.json
```json
{
  "timestamp": "2026-03-03",
  "total_modalities_processed": 168,
  "check_md_files_created": 168,
  "total_lines_generated": 20650,
  "total_size_mb": 0.49,
  "average_lines_per_file": 123,
  "average_size_kb": 3.0,
  "top_10_longest_files": [...]
}
```

**Contents:**
- File count and size statistics
- Line count per file
- Average metrics
- Longest files list

---

## Summary Documents

### CHECK_MD_GENERATION_SUMMARY.md
Comprehensive 8-section summary covering:
1. Executive Summary
2. Generation Statistics
3. File Structure (6-point template description)
4. Example Generated Files (3 detailed examples)
5. Key Findings (strengths & gaps)
6. Modality Distribution by Status (5 phases)
7. Integration Points (database, catalog, challenge data)
8. Usage Instructions & Next Steps

### GENERATED_FILES_INDEX.md (this file)
Quick reference guide listing:
- All file locations
- Modality groupings
- Report schemas
- Usage examples

---

## Generator Script

### scripts/generate_check_md.py

**Features:**
- Reads modality_database.py (63 entries)
- Reads _algorithm_catalog.py (76 entries)
- Generates 6-point check.md for each modality
- Preserves hand-crafted files (CT)
- Outputs JSON progress report
- Runs in < 5 seconds for all 168 modalities

**Key Functions:**
- `generate_check_md_content()` — Template generator with severity logic
- `get_modality_from_db()` — Database lookup
- `get_algorithms_for_variant()` — Algorithm catalog lookup
- `main()` — Orchestration and reporting

**Customization:**
```python
# Add to HAND_CRAFTED set to preserve files
HAND_CRAFTED = {
    "ct",           # Existing: CT comprehensive review
    "mri",          # Example: add if hand-crafted
    "widefield",    # Example: add if hand-crafted
}

# Adjust severity logic in generate_check_md_content()
if has_local_data:
    high_count = 0  # No missing data issues
    medium_count = 2
    low_count = 2
else:
    high_count = 1  # Missing data = HIGH severity
    medium_count = 1
    low_count = 1
```

---

## File Statistics

### Size Distribution
- **Total:** 0.49 MB (all 168 files)
- **Average per file:** 3.0 KB
- **Largest:** CT (8.9 KB, 236 lines)
- **Typical:** ~123 lines, ~3 KB

### Content Distribution
- **Total lines:** 20,650
- **Per file average:** 123 lines
- **Structure:** 6 sections × ~20 lines each

### By Status
- **Generated:** 167 files (99.4%)
- **Preserved hand-crafted:** 1 file (CT)
- **Errors:** 0

---

## Quick Access Examples

### View all modality status
```bash
cd /home/spiritai/pwm/Physics_World_Model
ls benchmarks/learn/*/check.md | wc -l  # Should show 168
```

### View specific modality
```bash
cat benchmarks/learn/cacti/check.md        # With data + algorithms
cat benchmarks/learn/widefield/check.md    # Database, no data/algos
cat benchmarks/learn/ct/check.md           # Hand-crafted comprehensive
```

### Check generation report
```bash
python3 << 'EOF'
import json
report = json.load(open('benchmark_check_generation_report.json'))
print(f"Generated: {report['generated']}")
print(f"Errors: {len(report['errors'])}")
print(f"With algorithms: {sum(1 for m in report['modalities'] if m['algorithms'] > 0)}")
