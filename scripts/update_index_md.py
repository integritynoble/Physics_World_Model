"""Generate updated index.md with comprehensive algorithm tracking from verification results."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "benchmark_results", "verification_8mod_5x.json")) as f:
    data = json.load(f)

MODALITY_NAMES = {
    "spc": ("SPC — Single-Pixel Camera", "spc"),
    "lensless": ("Lensless Imaging — Diffuser Camera", "lensless"),
    "holography": ("Digital Holographic Microscopy", "holography"),
    "ptychography": ("Ptychographic Imaging", "ptychography"),
    "cbct": ("CBCT — Cone-Beam CT", "cbct"),
    "ultrasound": ("Ultrasound B-mode", "ultrasound"),
    "cryo_em": ("Cryo-EM — Single-Particle Cryo-Electron Microscopy", "cryo_em"),
    "widefield": ("Widefield Fluorescence Microscopy", "widefield"),
}

# Also include the 4 modalities not in this batch (cassi, cacti, ct, mri)
OTHER_MODALITIES = ["cassi", "cacti", "ct", "mri"]

lines = []
lines.append("# Modality Templates Index\n")
lines.append("Comprehensive 7-step templates for all 168 imaging modalities in the PWM5 Benchmark.")
lines.append("Each template covers: (1) Verify Standard Dataset, (2) List All Algorithms, (3) Update Solvers, (4) Verify Each Algorithm, (5) Upload Checkpoints to GCS, (6) Upload Standard Dataset to GCS, (7) Push to GitHub.\n")
lines.append("---\n")
lines.append("## Implementation Tracking — 12 Flagship Paper Modalities\n")
lines.append("Each algorithm must be implemented at least **5 times** (5 independent verification runs on the standard dataset).")
lines.append("When all 5 runs are complete, the algorithm status is marked **done**.\n")

# Count totals
total_algos = 0
total_done = 0
for mod_id in MODALITY_NAMES:
    if mod_id in data and "n_solvers" in data[mod_id]:
        total_algos += data[mod_id]["n_solvers"]
        total_done += data[mod_id]["n_done"]

lines.append(f"Progress: **{total_done} / {total_algos} algorithms done** (8 modalities implemented) | Last updated: 2026-03-19\n")
lines.append("---\n")

# Generate tables for the 8 implemented modalities
section_num = 0
for mod_id, (display_name, _) in MODALITY_NAMES.items():
    section_num += 1
    r = data.get(mod_id, {})
    if "solvers" not in r:
        continue

    lines.append(f"### {section_num}. {display_name} (`{mod_id}`)\n")
    lines.append("| # | Algorithm | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean PSNR | Status |")
    lines.append("|---|-----------|-------|-------|-------|-------|-------|-----------|--------|")

    for i, (sk, sv) in enumerate(r["solvers"].items(), 1):
        runs = sv["runs"]
        run_marks = []
        for run in runs:
            if run["status"] == "ok":
                run_marks.append(f"{run['psnr']:.1f}")
            else:
                run_marks.append("ERR")

        status = sv["status"]
        status_mark = "done" if status == "done" else "partial" if status == "partial" else "fail"
        ref = sv.get("reference", "")
        name = sv["name"]
        if ref:
            name_display = f"{name}"
        else:
            name_display = name

        lines.append(
            f"| {i} | {name_display} | {run_marks[0]} | {run_marks[1]} | {run_marks[2]} | {run_marks[3]} | {run_marks[4]} | {sv['mean_psnr']:.1f} dB | {status_mark} |"
        )

    lines.append("")
    lines.append("---\n")

# Add placeholder sections for the other 4 modalities (cassi, cacti, ct, mri)
OTHER_INFO = {
    "cassi": ("CASSI — Coded Aperture Snapshot Spectral Imaging", 19),
    "cacti": ("CACTI — Coded Aperture Compressive Temporal Imaging", 7),
    "ct": ("CT — X-ray Computed Tomography", 40),
    "mri": ("MRI — Magnetic Resonance Imaging", 33),
}

for mod_id in OTHER_MODALITIES:
    section_num += 1
    display, n_algo = OTHER_INFO[mod_id]
    lines.append(f"### {section_num}. {display} (`{mod_id}`)\n")
    lines.append(f"*{n_algo} algorithms registered — verification pending*\n")
    lines.append("---\n")

# Summary table
lines.append("## Flagship Summary\n")
lines.append("| # | Modality | Algorithms | Done | Progress |")
lines.append("|---|----------|-----------|------|----------|")

all_mods = list(MODALITY_NAMES.items()) + [(k, (v[0], k)) for k, v in OTHER_INFO.items()]
grand_total = 0
grand_done = 0

for i, (mod_id, (display_name, _)) in enumerate(all_mods, 1):
    if mod_id in data and "n_solvers" in data[mod_id]:
        n = data[mod_id]["n_solvers"]
        d = data[mod_id]["n_done"]
        pct = int(100 * d / n) if n > 0 else 0
        lines.append(f"| {i} | {display_name.split('—')[0].strip()} | {n} | {d} | {pct}% |")
    elif mod_id in OTHER_INFO:
        n = OTHER_INFO[mod_id][1]
        lines.append(f"| {i} | {display_name.split('—')[0].strip()} | {n} | 0 | 0% |")
        d = 0
    else:
        n, d = 0, 0
    grand_total += n
    grand_done += d

lines.append(f"| | **Total** | **{grand_total}** | **{grand_done}** | **{int(100*grand_done/grand_total) if grand_total else 0}%** |")
lines.append("")
lines.append("---\n")

# Template files section
lines.append("## Template Files in This Folder (55 modalities)\n")
lines.append("These templates cover all modalities that were not already in `_templates_part1.md` through `_templates_part8.md`.\n")
lines.append("### [medical_core.md](medical_core.md) — Medical Imaging Core (7 modalities)\n")
lines.append("### [remote_sensing.md](remote_sensing.md) — Remote Sensing & Radar (10 modalities)\n")
lines.append("### [scanning_probe_spectroscopy.md](scanning_probe_spectroscopy.md) — Scanning Probe & Spectroscopic Imaging (13 modalities)\n")
lines.append("### [electron_nuclear_xray.md](electron_nuclear_xray.md) — Electron Microscopy, Nuclear & X-ray Techniques (13 modalities)\n")
lines.append("### [quantum_3d_compressive.md](quantum_3d_compressive.md) — Quantum, 3D Reconstruction & Compressive/Ultrafast (12 modalities)\n")
lines.append("")
lines.append("---\n")
lines.append("## Summary\n")
lines.append("| Location | Files | Modalities | Steps |")
lines.append("|----------|-------|------------|-------|")
lines.append("| `templates/` (this folder) | 5 | 55 | 385 |")
lines.append("| `_templates_part1-8.md` (parent) | 8 | 113 | 791 |")
lines.append("| **Total** | **13** | **168** | **1176** |")
lines.append("")
lines.append("All 168 modalities now have complete 7-step templates.")
lines.append("")

# Write
index_path = os.path.join(ROOT, "datasets", "templates", "index.md")
with open(index_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Updated {index_path}")
print(f"  {grand_done}/{grand_total} algorithms done across 12 flagship modalities")
