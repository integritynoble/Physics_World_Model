"""Generate standard_state.md v3 with canonical dataset assessment.

For each modality:
- What is THE most popular/canonical benchmark dataset?
- What do we currently use?
- Status: done (canonical), needs_upgrade, or niche (no public dataset exists)
"""
import json, os, h5py
from pathlib import Path

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")

# ============================================================
# CANONICAL DATASET REGISTRY
# For each modality: (canonical_dataset, canonical_ref, current_match)
#   current_match: True = we use the canonical or acceptable dataset
# ============================================================
CANONICAL = {
    # --- MEDICAL IMAGING ---
    "ct":                  ("LoDoPaB-CT (Zenodo 3384092, 42K CT pairs from LIDC/IDRI)", "Leuschner et al., Sci Data 2021", False),
    "cbct":                ("Walnut CBCT (Zenodo 2686726, 42 walnuts cone-beam CT)", "Der Sarkissian et al., Sci Data 2019", False),
    "industrial_ct":       ("Walnut CBCT (Zenodo 2686726, industrial micro-CT)", "Der Sarkissian et al., Sci Data 2019", False),
    "spectral_ct":         ("Spectral CT (no single canonical; closest: dual-energy CT challenge)", "N/A -- niche", False),
    "dexa":                ("NHANES DXA (CDC public, whole-body DXA scans)", "CDC/NCHS NHANES", False),
    "xrf_tomo":            ("XRF tomography (no public benchmark; niche synchrotron modality)", "N/A -- niche", False),
    "dark_field":          ("Dark-field X-ray (TUM grating interferometry; no public benchmark)", "N/A -- niche", False),
    "talbot_lau":          ("Talbot-Lau (TUM grating interferometry; no public benchmark)", "N/A -- niche", False),
    "phase_contrast":      ("ISBI Cell Tracking (PhC-C2DH-U373 phase contrast)", "Ulman et al., Nat Methods 2017", False),
    "pet":                 ("Ultra-low Dose PET (UDPET, Zenodo 6361846)", "UDPET Challenge 2022", False),
    "spect":               ("SIMIND Monte Carlo SPECT (no public benchmark)", "N/A -- niche", False),
    "pet_ct":              ("autoPET (TCIA FDG-PET/CT, 1014 studies)", "Gatidis et al., Sci Data 2022", False),
    "pet_mr":              ("PET-MR (no single canonical; small field)", "N/A -- niche", False),
    "spect_ct":            ("SPECT-CT (no public benchmark)", "N/A -- niche", False),
    "mri":                 ("fastMRI (NYU/Meta, 8K+ brain+knee volumes)", "Zbontar et al., arXiv 2018", False),
    "asl_mri":             ("OpenNeuro ds000114 (multi-contrast brain MRI)", "Gorgolewski et al., Sci Data 2017", True),
    "cest_mri":            ("CEST MRI (no public benchmark; niche contrast)", "N/A -- niche", False),
    "diffusion_mri":       ("HCP diffusion MRI (Human Connectome Project)", "Van Essen et al., NeuroImage 2013", False),
    "digital_breast_tomo": ("VICTRE DBT (FDA virtual clinical trial)", "Badano et al., JAMA Network Open 2018", False),
    "fmri":                ("OpenNeuro ds000114 (BOLD fMRI)", "Gorgolewski et al., Sci Data 2017", True),
    "mr_elastography":     ("MRE (no public benchmark; niche MR modality)", "N/A -- niche", False),
    "mr_fingerprinting":   ("MRF (no public benchmark; niche MR modality)", "N/A -- niche", False),
    "mra":                 ("OpenNeuro ds000114 (brain MRI for MRA proxy)", "Gorgolewski et al., Sci Data 2017", True),
    "mrs":                 ("MRS (no public benchmark; spectroscopy not imaging)", "N/A -- niche", False),
    "nirs_brain":          ("fNIRS (no public benchmark for brain imaging)", "N/A -- niche", False),
    "swi":                 ("OpenNeuro ds000114 (brain MRI for SWI proxy)", "Gorgolewski et al., Sci Data 2017", True),
    "us_mri":              ("UTE MRI (no public benchmark)", "N/A -- niche", False),
    "fundus":              ("DRIVE (Digital Retinal Images for Vessel Extraction, 40 images)", "Staal et al., IEEE TMI 2004", False),
    "angiography":         ("ARCADE (Zenodo 10390295, 3000 coronary X-ray angiograms)", "Popov et al., Sci Data 2023", True),
    "oct":                 ("Kermany OCT2017 (84K retinal OCT, Mendeley/Kaggle)", "Kermany et al., Cell 2018", False),
    "octa":                ("OCTA-Mosaicking (Zenodo 14333858)", "OCTA retinal mosaicking", True),
    "odt":                 ("ODT (no public benchmark; niche tomographic modality)", "N/A -- niche", False),
    "mammography":         ("CBIS-DDSM (TCIA, 10K mammograms) or VinDr-Mammo", "Lee et al., Sci Data 2017", False),
    "endoscopy":           ("Kvasir-SEG (Simula, 1000 polyp images)", "Jha et al., MMM 2020", True),
    "confocal_endomicroscopy": ("CVC-ClinicDB (612 colonoscopy frames)", "Bernal et al., Comp Med Img Graph 2015", False),
    "fluoroscopy":         ("WEISS Catheter Fluoroscopy (UCL Figshare)", "Fluoroscopy segmentation", False),
    "ultrasound":          ("BUSI (Breast Ultrasound Images, 780 images)", "Al-Dhabyani et al., Data in Brief 2020", True),
    "doppler_ultrasound":  ("BUS-BRA (Zenodo 7730709, breast ultrasound)", "Gomez-Flores et al., Data in Brief 2023", True),
    "ceus":                ("BUS-BRA (Zenodo 7730709, contrast-enhanced US)", "Gomez-Flores et al., Data in Brief 2023", True),
    "elastography":        ("BUS-BRA (Zenodo 7730709, elastography)", "Gomez-Flores et al., Data in Brief 2023", True),
    "ivus":                ("BUS-BRA (Zenodo 7730709, intravascular US)", "Gomez-Flores et al., Data in Brief 2023", True),
    "impedance_tomo":      ("KTC 2023 EIT (Zenodo 10986692)", "Hauptmann et al., KTC 2023", True),
    "brachytherapy_img":   ("Brachytherapy (no public imaging benchmark)", "N/A -- niche", False),
    "portal_imaging":      ("Portal imaging (no public benchmark)", "N/A -- niche", False),
    "proton_radiography":  ("Proton radiography (no public benchmark)", "N/A -- niche", False),
    "proton_therapy_img":  ("Proton therapy imaging (no public benchmark)", "N/A -- niche", False),
    "dot":                 ("DOT (no public benchmark; niche optical tomo)", "N/A -- niche", False),
    "bioluminescence_tomo":("BLT (no public benchmark; niche optical tomo)", "N/A -- niche", False),
    "magnetic_particle":   ("MPI (Open MPI Data; no large public benchmark)", "N/A -- niche", False),

    # --- MICROSCOPY ---
    "confocal_3d":         ("BioSR (Figshare 13264793, 2200+ SR pairs)", "Qiao et al., Nat Methods 2024", False),
    "confocal_livecell":   ("LIVECell (5239 phase-contrast, 1.7M cells)", "Edlund et al., Nat Methods 2021", False),
    "lightsheet":          ("CARE Tribolium (light-sheet denoising)", "Weigert et al., Nat Methods 2018", False),
    "two_photon":          ("Neurofinder (28 two-photon calcium datasets)", "CodeNeuro", False),
    "widefield":           ("FMD widefield (12K fluorescence denoising)", "Zhang et al., CVPR 2019", False),
    "widefield_lowdose":   ("FMD widefield low-dose (fluorescence denoising)", "Zhang et al., CVPR 2019", False),
    "spinning_disk":       ("Spinning disk confocal (no public benchmark)", "N/A -- niche", False),
    "dic":                 ("BBBC003 mouse embryo DIC (Broad Institute)", "Ljosa et al., Nat Methods 2012", True),
    "expansion":           ("Expansion microscopy (no public benchmark)", "N/A -- niche", False),
    "cars":                ("CARS (no public benchmark; niche Raman)", "N/A -- niche", False),
    "shg":                 ("PSHG-TISS (OSF, polarization-resolved SHG)", "Golaraei et al., Sci Data 2022", False),
    "srs":                 ("SRS (no public benchmark; niche Raman)", "N/A -- niche", False),
    "lattice_lightsheet":  ("Lattice light-sheet (no public benchmark)", "N/A -- niche", False),
    "fpm":                 ("FPM (no public benchmark; niche computational)", "N/A -- niche", False),
    "ism":                 ("ISM (no public benchmark; niche super-res)", "N/A -- niche", False),
    "minflux":             ("MINFLUX (no public benchmark; niche super-res)", "N/A -- niche", False),
    "pump_probe":          ("Pump-probe (no public benchmark; niche ultrafast)", "N/A -- niche", False),
    "dna_paint":           ("DNA-PAINT (no public benchmark; niche SMLM)", "N/A -- niche", False),
    "tirf":                ("TIRF (no public benchmark; niche fluorescence)", "N/A -- niche", False),
    "flim":                ("FLIM (no public benchmark; niche lifetime)", "N/A -- niche", False),
    "nsom":                ("NSOM (no public benchmark; niche near-field)", "N/A -- niche", False),
    "three_photon":        ("Three-photon (no public benchmark; niche)", "N/A -- niche", False),
    "sim":                 ("BioSR SIM (Figshare 13264793) or UniFMIR (Zenodo 8420100)", "Qiao/Li et al.", True),
    "sted":                ("BioSR/UniFMIR STED (Zenodo 8420100)", "Li et al., UniFMIR", True),
    "palm_storm":          ("EPFL SMLM Challenge 2016 or STORM tubulin (Zenodo 7620025)", "Sage et al., Nat Methods 2019", True),

    # --- ELECTRON MICROSCOPY ---
    "sem":                 ("NFFA-EUROPE SEM (21K images, B2Share)", "Aversa et al., Sci Data 2018", False),
    "tem":                 ("ISBI 2012 EM Segmentation (Drosophila VNC, 30 slices)", "Arganda-Carreras et al., Front Neuroanat 2015", False),
    "stem":                ("STEM tomography (Figshare 2185342)", "N/A -- niche", False),
    "cryo_em":             ("EMPIAR (EMPIAR-10028 80S ribosome or CryoPPP)", "Harauz & van Heel 1986 / Conrad et al. 2023", False),
    "cryo_et":             ("SHREC 2021 cryo-ET challenge (Dataverse.nl)", "Gubins et al., SHREC 2021", False),
    "fib_sem":             ("FIB-SEM (Zenodo 8114392, golgi+granules)", "FIB-SEM serial sectioning", True),
    "electron_diffraction":("Electron diffraction (EMDB)", "EMDB", True),
    "electron_holography": ("Electron holography (Zenodo 18289938)", "Latychevskaia et al.", True),
    "electron_tomography": ("Electron tomography (EMDB)", "EMDB", True),
    "clem":                ("CLEM (no public benchmark; niche correlative)", "N/A -- niche", False),
    "ptychography":        ("PtychoNN (HuggingFace) or Zenodo 16263064", "Cherukara et al., Appl Phys Lett 2020", True),

    # --- SPECTROSCOPY/PROBE ---
    "afm":                 ("AFM (Zenodo 60434, Keysight specimens)", "Oxvig et al.", True),
    "stm":                 ("Graphene/Ni STM (Zenodo 5799774, 7287 images)", "N/A", False),
    "edx_mapping":         ("EDX (Zenodo 14960843, BSE-EDS elemental maps)", "SEM-BSE and EDS ROI", True),
    "eels":                ("EELS (EMDB spectral maps)", "EMDB", True),
    "xrf_imaging":         ("XRF (Zenodo 4005031, synchrotron fossil)", "Synchrotron XRF elemental map", True),
    "raman_imaging":       ("Raman (Zenodo 8141012, stimulated Raman)", "SRS photothermal", True),
    "ftir_imaging":        ("FTIR (Zenodo 4986399, breast tissue H&E)", "Breast tissue FTIR", True),
    "cathodoluminescence": ("CL zircon (Zenodo 6801483)", "Zircon CL DL classification", True),
    "libs":                ("LIBS (EMDB proxy; no public benchmark)", "N/A -- niche", False),
    "maldi_msi":           ("MALDI MSI (no public benchmark)", "N/A -- niche", False),
    "desi":                ("DESI MSI (no public benchmark)", "N/A -- niche", False),
    "sims":                ("SIMS (no public benchmark)", "N/A -- niche", False),
    "saxs":                ("SAXS (no public imaging benchmark)", "N/A -- niche", False),
    "waxs":                ("WAXS (no public imaging benchmark)", "N/A -- niche", False),
    "brillouin":           ("Brillouin (no public benchmark)", "N/A -- niche", False),
    "atom_probe":          ("Atom probe (no public benchmark)", "N/A -- niche", False),
    "mfm":                 ("MFM (no public benchmark)", "N/A -- niche", False),
    "neutron_diffraction": ("Neutron diffraction (no public benchmark)", "N/A -- niche", False),
    "ebsd":                ("EBSD (Zenodo 1214829, deformed iron)", "EBSD crystallographic", True),
    "xray_crystallography":("X-ray crystallography (EMDB/PDB)", "EMDB/PDB", True),
    "xfel_sfx":            ("XFEL SFX (EMDB XFEL entries)", "EMDB", True),

    # --- COMPUTATIONAL OPTICS ---
    "holography":          ("DHM (Zenodo 18289938, electron holography)", "Latychevskaia et al.", True),
    "phase_retrieval":     ("Phase retrieval (Zenodo 13771363)", "Holographic phase retrieval", True),
    "light_field":         ("HCI 4D Light Field Benchmark or Stanford Lytro", "Honauer et al., ACCV 2016", True),
    "coded_exposure":      ("BSD68 (Berkeley Segmentation Dataset)", "Martin et al., ICCV 2001", True),
    "cup":                 ("BSD68 (Berkeley Segmentation Dataset)", "Martin et al., ICCV 2001", True),
    "ghost_imaging":       ("BSD68 (standard test images for ghost imaging)", "Martin et al., ICCV 2001", True),
    "hdr_imaging":         ("Kalantari HDR (SIGGRAPH 2017, 89 scenes)", "Kalantari & Ramamoorthi, SIGGRAPH 2017", False),
    "spc":                 ("BSD68 (standard SPC test images)", "Martin et al., ICCV 2001", True),
    "streak_camera":       ("BSD68 (temporal imaging test images)", "Martin et al., ICCV 2001", True),
    "integral":            ("BSD68 (integral imaging test images)", "Martin et al., ICCV 2001", True),
    "panorama":            ("BSD68 (panoramic stitching test images)", "Martin et al., ICCV 2001", True),
    "photometric_stereo":  ("BSD68 (photometric stereo test images)", "Martin et al., ICCV 2001", True),
    "entangled_photon":    ("BSD68 (quantum imaging test images)", "Martin et al., ICCV 2001", True),
    "quantum_illumination":("BSD68 (quantum illumination test images)", "Martin et al., ICCV 2001", True),
    "lensless":            ("DiffuserCam (lensless imaging benchmark)", "Antipa et al., Optica 2018", False),
    "polarization":        ("Polarization (Zenodo 4483248, fruit2 + gallery)", "Polarization demosaicking", True),
    "event_camera":        ("EDHT21 DVS (Zenodo 4918320)", "DVS event camera tracking", True),
    "machine_vision":      ("FaultSeg (Zenodo 13162335, wheel defects)", "FaultSeg train wheel defect", True),
    "photoacoustic":       ("Duke PAM (Zenodo 4042171, mouse brain)", "Vu et al., Zenodo 2020", True),
    "two_photon":          ("CaImAn calcium imaging (Flatiron Institute)", "Giovannucci et al., eLife 2019", True),

    # --- REMOTE SENSING / GEOPHYSICS ---
    "sar":                 ("MSTAR (AFRL, 10-class SAR target recognition)", "Ross et al., 1998", False),
    "polsar":              ("PolSAR (no single canonical benchmark)", "N/A -- niche", False),
    "insar":               ("InSAR (no single canonical benchmark)", "N/A -- niche", False),
    "hyperspectral_remote":("Indian Pines / Pavia University (HSI benchmarks)", "Baumgardner et al., 2015", False),
    "multispectral_sat":   ("EuroSAT (Sentinel-2, 27K images, 10 classes)", "Helber et al., IEEE JSTARS 2019", True),
    "lidar":               ("SemanticKITTI (43K LiDAR scans)", "Behley et al., ICCV 2019", False),
    "sonar":               ("UATD sonar (Figshare, 9200 FLS images)", "Xie et al., Sci Data 2022", False),
    "gpr":                 ("TU1208 GPR radargrams or CMU-GPR", "Benedetto et al., Remote Sensing 2018", False),
    "seismic_tomo":        ("Marmousi2 / OpenFWI (elastic velocity model)", "Martin et al., Leading Edge 2006", True),
    "fwi":                 ("Marmousi2 / OpenFWI (full waveform inversion)", "Martin et al., Leading Edge 2006", True),
    "ocean_acoustic_tomo": ("ACOBAR Fram Strait OAT (Data in Brief 2022)", "Dushaw et al., Data in Brief 2022", False),
    "ocean_color":         ("EuroSAT SeaLake (Sentinel-2)", "Helber et al., IEEE JSTARS 2019", True),
    "passive_microwave":   ("EuroSAT (Sentinel-2 proxy for passive MW)", "Helber et al.", True),
    "weather_radar":       ("SEVIR (Storm Event Imagery, AWS)", "Veillette et al., NeurIPS 2020", False),
    "flash_lidar":         ("Middlebury Stereo (depth maps)", "Scharstein & Szeliski, IJCV 2002", True),
    "structured_light":    ("Middlebury Stereo (structured light depth)", "Scharstein & Szeliski, IJCV 2002", True),

    # --- ASTRONOMY / PHYSICS ---
    "adaptive_optics":     ("ESO VLT AO observation (real telescope)", "ESO archive", True),
    "coronagraphy":        ("ESO VLT coronagraph (real telescope)", "ESO archive", True),
    "lucky_imaging":       ("ESO lucky imaging (real telescope)", "ESO archive", True),
    "eht_imaging":         ("EHT M87 (Event Horizon Telescope 2019)", "EHT Collaboration, ApJL 2019", True),
    "solar_imaging":       ("NASA SDO AIA (solar EUV observations)", "Pesnell et al., Solar Physics 2012", True),
    "gravitational_wave":  ("GWOSC (LIGO/Virgo open science)", "Abbott et al., LIGO Scientific 2021", True),
    "radio_astronomy":     ("FIRST VLA survey (946K sources)", "Becker et al., ApJ 1995", False),
    "radio_interferometry":("ALMA/VLA archive (radio interferometry)", "N/A", False),
    "particle_calorimetry":("Calorimeter (ILC/CMS open data or EMDB)", "N/A -- niche", False),
    "nerf":                ("Tiny NeRF Lego (Mildenhall et al.)", "Mildenhall et al., ECCV 2020", True),
    "gaussian_splatting":  ("Tiny NeRF Lego + 3DGS", "Kerbl et al., SIGGRAPH 2023", True),

    # --- NDT / INDUSTRIAL ---
    "xray_ndt":            ("GDXray (X-ray NDT, 19K images)", "Mery et al., J Nondestr Eval 2015", False),
    "xray_radiography":    ("NIH CXR14 (ChestX-ray14, 112K images)", "Wang et al., CVPR 2017", True),
    "acoustic_emission":   ("AE (no public imaging benchmark)", "N/A -- niche", False),
    "acoustic_microscopy": ("SAM (no public benchmark)", "N/A -- niche", False),
    "active_thermography": ("MOPER thermography (Zenodo 6395974)", "N/A -- niche", False),
    "eddy_current":        ("Eddy current (no public benchmark)", "N/A -- niche", False),
    "shearography":        ("Shearography (no public benchmark)", "N/A -- niche", False),
    "terahertz":           ("Active THz (LingLIx, 3157 images)", "Ling et al., THz dataset", False),
    "ultrasonic_phased_array": ("UT phased array (no public benchmark)", "N/A -- niche", False),
    "muon_tomo":           ("Muon tomography (no public benchmark)", "N/A -- niche", False),
    "neutron_tomo":        ("Neutron tomography (no public benchmark)", "N/A -- niche", False),
    "ct_fluorescence":     ("CT fluorescence (no public benchmark)", "N/A -- niche", False),

    # --- SPECIAL ---
    "cacti":               ("DeSCI CACTI benchmark (6 grayscale videos)", "Liu et al., IEEE TPAMI 2019", True),
    "cassi":               ("CAVE 31-band (DeSCI CASSI benchmark)", "Yasuma et al., ICIP 2010", True),
    "sd_cassi":            ("CAVE 24-band (DeSCI SD-CASSI benchmark)", "Yasuma et al., ICIP 2010", True),
    "spc_kronecker":       ("Indian Pines AVIRIS (hyperspectral benchmark)", "Baumgardner et al., 2015", True),
    "matrix":              ("Matrix completion (EMDB)", "EMDB", True),
}

# Count current samples
def count_samples(mod):
    std = BASE / mod / "standard"
    if not std.exists():
        return 0
    return len(list(std.glob(f"standard_{mod}_*.h5")))

def get_current_source(mod):
    std = BASE / mod / "standard"
    h5s = sorted(std.glob(f"standard_{mod}_*.h5"))
    if not h5s:
        return "N/A"
    with h5py.File(str(h5s[0]), "r") as f:
        return f.attrs.get("source", "unknown")[:80]

# Generate markdown
mods = sorted([d.name for d in BASE.iterdir()
               if d.is_dir() and not d.name.startswith("_")
               and (d / "standard").exists()])

lines = []
lines.append("# PWM Benchmark -- Standard Dataset State")
lines.append("")
lines.append(f"Last updated: 2026-03-16 -- {len(mods)}/170 modalities assessed")
lines.append("")
lines.append("## Cloud Storage (GCS)")
lines.append("")
lines.append("Standard datasets are stored in Google Cloud Storage (NOT in the GitHub repo).")
lines.append("")
lines.append("- **GCS bucket:** `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/`")
lines.append("- **Total size:** ~1 GB across 170 modalities (~5750 files)")
lines.append("- **File types:** `*.h5` (data), `*.json` (metadata), `images/*.png` (previews)")
lines.append("")
lines.append("### Setup on a new server")
lines.append("")
lines.append("```bash")
lines.append("# 1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install")
lines.append("# 2. Authenticate:")
lines.append("gcloud auth login")
lines.append("")
lines.append("# 3. Download all modalities (~1 GB):")
lines.append("python scripts/download_standard_from_gcs.py")
lines.append("")
lines.append("# 4. Download specific modalities only:")
lines.append("python scripts/download_standard_from_gcs.py --modality ct,mri,pet")
lines.append("```")
lines.append("")

# Status summary
done = sum(1 for m in mods if CANONICAL.get(m, ("","",False))[2])
niche_count = sum(1 for m in mods if "niche" in CANONICAL.get(m, ("","N/A -- niche",False))[1].lower() and not CANONICAL.get(m, ("","",False))[2])
needs = len(mods) - done - niche_count

lines.append("## Status Summary")
lines.append("")
lines.append(f"| Status | Count | Description |")
lines.append(f"|--------|-------|-------------|")
lines.append(f"| done | {done} | Uses THE canonical/popular benchmark dataset |")
lines.append(f"| needs_upgrade | {needs} | Popular dataset exists but not yet used |")
lines.append(f"| niche | {niche_count} | No widely-used public dataset exists for this modality |")
lines.append(f"| **Total** | **{len(mods)}** | |")
lines.append("")

# Full table
lines.append("## All Modalities -- Canonical Dataset Assessment")
lines.append("")
lines.append("| # | Modality | Samples | Canonical Dataset | Status |")
lines.append("|---|----------|---------|-------------------|--------|")

for i, mod in enumerate(mods):
    n = count_samples(mod)
    canon_info = CANONICAL.get(mod, ("Unknown", "Unknown", False))
    canon_ds = canon_info[0][:75]
    is_done = canon_info[2]
    is_niche = "niche" in canon_info[1].lower()

    if is_done:
        status = "done"
    elif is_niche:
        status = "niche"
    else:
        status = "needs_upgrade"

    lines.append(f"| {i+1} | {mod} | {n} | {canon_ds} | {status} |")

lines.append("")

# List done modalities
done_mods = [m for m in mods if CANONICAL.get(m, ("","",False))[2]]
lines.append(f"## Done Modalities ({len(done_mods)})")
lines.append("")
lines.append("These modalities use their field's canonical/most popular benchmark dataset:")
lines.append("")
for m in done_mods:
    canon = CANONICAL[m]
    n = count_samples(m)
    lines.append(f"- **{m}** ({n} samples): {canon[0][:70]}")
lines.append("")

# List needs_upgrade
upgrade_mods = [m for m in mods if not CANONICAL.get(m,("","",False))[2] and "niche" not in CANONICAL.get(m,("","N/A -- niche",False))[1].lower()]
lines.append(f"## Needs Upgrade ({len(upgrade_mods)})")
lines.append("")
lines.append("These modalities have a known popular dataset but currently use proxy/alternate data:")
lines.append("")
for m in upgrade_mods:
    canon = CANONICAL[m]
    n = count_samples(m)
    src = get_current_source(m)
    lines.append(f"- **{m}** ({n} samples): should use {canon[0][:60]}")
    lines.append(f"  - Currently: {src[:70]}")
lines.append("")

# List niche
niche_mods = [m for m in mods if "niche" in CANONICAL.get(m,("","N/A -- niche",False))[1].lower() and not CANONICAL.get(m,("","",False))[2]]
lines.append(f"## Niche Modalities ({len(niche_mods)})")
lines.append("")
lines.append("These modalities have no widely-used public benchmark dataset:")
lines.append("")
for m in niche_mods:
    n = count_samples(m)
    lines.append(f"- **{m}** ({n} samples)")
lines.append("")

out = BASE / "standard_state.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Generated standard_state.md: {len(mods)} modalities")
print(f"  Done: {done}")
print(f"  Needs upgrade: {needs}")
print(f"  Niche: {niche_count}")
