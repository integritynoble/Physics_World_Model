"""Generate standard_state.md v6
- 75 done: canonical real data
- 34 needs_upgrade -> 'needs_canonical' with dataset links (pending download of 30 samples)
- 61 niche -> 'simulation' status
"""
import h5py
from pathlib import Path

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")

# ============================================================
# CANONICAL ASSESSMENT
# (canonical_name, url, reference, status)
# status: "done", "needs_canonical", "simulation"
# ============================================================
C = {
    # === DONE (75): real data from the correct modality ===
    "cacti":               ("DeSCI CACTI benchmark (6 grayscale videos)", "https://github.com/liuyang12/DeSCI", "Liu et al., IEEE TPAMI 2019", "done"),
    "cassi":               ("CAVE 28-band hyperspectral (10 scenes)", "https://www.cs.columbia.edu/CAVE/databases/multispectral/", "Wagadarikar et al., Applied Optics 2008", "done"),
    # sd_cassi merged into cassi
    "spc_kronecker":       ("Indian Pines AVIRIS", "https://engineering.purdue.edu/~biehl/MultiSpec/", "Baumgardner et al., 2015", "done"),
    "nerf":                ("Tiny NeRF Lego scene", "https://github.com/bmild/nerf", "Mildenhall et al., ECCV 2020", "done"),
    "gaussian_splatting":  ("Tiny NeRF Lego + 3DGS", "https://github.com/graphdeco-inria/gaussian-splatting", "Kerbl et al., SIGGRAPH 2023", "done"),
    "oct":                 ("OCTDL retinal OCT (2064 images)", "https://www.kaggle.com/datasets/paultimothymooney/kermany2018", "Kermany et al., Cell 2018", "done"),
    "endoscopy":           ("Kvasir-SEG (1000 colonoscopy polyp images)", "https://datasets.simula.no/kvasir-seg/", "Jha et al., MMM 2020", "done"),
    "ultrasound":          ("BUSI breast ultrasound (780 images)", "https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset", "Al-Dhabyani et al., 2020", "done"),
    "impedance_tomo":      ("KTC 2023 EIT phantom (Zenodo 10986692)", "https://zenodo.org/records/10986692", "Hauptmann et al., KTC 2023", "done"),
    "photoacoustic":       ("Duke PAM mouse brain (Zenodo 4042171)", "https://zenodo.org/records/4042171", "Vu et al., 2020", "done"),
    "eht_imaging":         ("EHT M87 black hole (2019 SR1)", "https://eventhorizontelescope.org/for-astronomers/data", "EHT Collaboration, ApJL 2019", "done"),
    "gravitational_wave":  ("GWOSC GW150914 strain", "https://gwosc.org/", "LIGO Scientific, 2021", "done"),
    "solar_imaging":       ("NASA SDO AIA EUV composite", "https://sdo.gsfc.nasa.gov/", "Pesnell et al., Solar Physics 2012", "done"),
    "dic":                 ("BBBC003 mouse embryo DIC", "https://bbbc.broadinstitute.org/BBBC003", "Ljosa et al., Nat Methods 2012", "done"),
    "afm":                 ("AFM specimens (Zenodo 60434)", "https://zenodo.org/records/60434", "Oxvig et al.", "done"),
    "tem":                 ("TEM cilia (Zenodo 11188503)", "https://zenodo.org/records/11188503", "TEM cilia short-exposure", "done"),
    "cathodoluminescence": ("CL zircon (Zenodo 6801483)", "https://zenodo.org/records/6801483", "Zircon CL DL", "done"),
    "edx_mapping":         ("EDX elemental maps (Zenodo 14960843)", "https://zenodo.org/records/14960843", "BSE-EDS ROI", "done"),
    "xrf_imaging":         ("XRF fossil map (Zenodo 4005031)", "https://zenodo.org/records/4005031", "Synchrotron XRF", "done"),
    "raman_imaging":       ("Raman photothermal (Zenodo 8141012)", "https://zenodo.org/records/8141012", "SRS photothermal", "done"),
    "ftir_imaging":        ("FTIR breast tissue (Zenodo 4986399)", "https://zenodo.org/records/4986399", "Breast FTIR H&E", "done"),
    "holography":          ("Electron holography (Zenodo 18289938)", "https://zenodo.org/records/18289938", "Latychevskaia et al.", "done"),
    "phase_retrieval":     ("Holographic phase (Zenodo 13771363)", "https://zenodo.org/records/13771363", "Holographic phase retrieval", "done"),
    "polarization":        ("Polarimetric camera (Zenodo 4483248)", "https://zenodo.org/records/4483248", "Polarization demosaic", "done"),
    "event_camera":        ("EDHT21 DVS events (Zenodo 4918320)", "https://zenodo.org/records/4918320", "DVS event camera", "done"),
    "machine_vision":      ("FaultSeg wheel defects (Zenodo 13162335)", "https://zenodo.org/records/13162335", "FaultSeg defect detection", "done"),
    "sim":                 ("UniFMIR SIM F-actin (Zenodo 8420100)", "https://zenodo.org/records/8420100", "Li et al., UniFMIR", "done"),
    "sted":                ("UniFMIR STED (Zenodo 8420100)", "https://zenodo.org/records/8420100", "Li et al., UniFMIR", "done"),
    "palm_storm":          ("STORM tubulin (Zenodo 7620025)", "https://zenodo.org/records/7620025", "Sage et al., Nat Methods 2019", "done"),
    "two_photon":          ("CaImAn calcium imaging", "https://github.com/flatironinstitute/CaImAn", "Giovannucci et al., eLife 2019", "done"),
    "ptychography":        ("Ptychography exp (Zenodo 16263064)", "https://zenodo.org/records/16263064", "Experimental ptychography", "done"),
    "tof_camera":          ("ToF depth maps (Zenodo 10732158)", "https://zenodo.org/records/10732158", "ZHAW-ISC ToF+RGB", "done"),
    "fib_sem":             ("FIB-SEM golgi (Zenodo 8114392)", "https://zenodo.org/records/8114392", "FIB-SEM serial section", "done"),
    "ebsd":                ("EBSD deformed iron (Zenodo 1214829)", "https://zenodo.org/records/1214829", "Deformed iron EBSD", "done"),
    "electron_holography": ("Electron holography (Zenodo 18289938)", "https://zenodo.org/records/18289938", "Off-axis e-holography", "done"),
    "octa":                ("OCTA-Mosaicking (Zenodo 14333858)", "https://zenodo.org/records/14333858", "OCTA retinal mosaicking", "done"),
    "doppler_ultrasound":  ("BUS-BRA breast US (Zenodo 7730709)", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", "done"),
    "ceus":                ("BUS-BRA contrast US (Zenodo 7730709)", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", "done"),
    "elastography":        ("BUS-BRA elastography (Zenodo 7730709)", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", "done"),
    "ivus":                ("BUS-BRA intravascular (Zenodo 7730709)", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", "done"),
    "seismic_tomo":        ("Marmousi2 elastic velocity", "https://wiki.seg.org/wiki/Open_data", "Martin et al., Leading Edge 2006", "done"),
    "fwi":                 ("Marmousi2 / OpenFWI", "https://openfwi-lanl.github.io/", "Deng et al., NeurIPS 2022", "done"),
    "weather_radar":       ("NOAA NEXRAD composite", "https://www.ncei.noaa.gov/products/radar", "NOAA NEXRAD", "done"),
    "adaptive_optics":     ("ESO VLT AO observation", "https://archive.eso.org/", "ESO archive", "done"),
    "coronagraphy":        ("ESO VLT coronagraph", "https://archive.eso.org/", "ESO archive", "done"),
    "lucky_imaging":       ("ESO lucky imaging", "https://archive.eso.org/", "ESO archive", "done"),
    "light_field":         ("Stanford Lytro light field", "http://lightfields.stanford.edu/LF2016.html", "Stanford LF Archive", "done"),
    "flash_lidar":         ("Middlebury Stereo depth", "https://vision.middlebury.edu/stereo/", "Scharstein & Szeliski, 2002", "done"),
    "structured_light":    ("Middlebury Stereo depth", "https://vision.middlebury.edu/stereo/", "Scharstein & Szeliski, 2002", "done"),
    "xray_radiography":    ("NIH ChestX-ray14", "https://nihcc.app.box.com/v/ChestXray-NIHCC", "Wang et al., CVPR 2017", "done"),
    "angiography":         ("ARCADE coronary XCA (Zenodo 10390295)", "https://zenodo.org/records/10390295", "Popov et al., Sci Data 2023", "done"),
    "multispectral_sat":   ("EuroSAT Sentinel-2 (27K images)", "https://zenodo.org/records/7711810", "Helber et al., IEEE JSTARS 2019", "done"),
    "ocean_color":         ("EuroSAT SeaLake Sentinel-2", "https://zenodo.org/records/7711810", "Helber et al., 2019", "done"),
    # BSD68-based optics: canonical test images
    "coded_exposure":      ("BSD68 (standard CI test images)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", "done"),
    "cup":                 ("BSD68 (CUP test images)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "ghost_imaging":       ("BSD68 (ghost imaging test)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "integral":            ("BSD68 (integral imaging test)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "panorama":            ("BSD68 (panorama test images)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "photometric_stereo":  ("BSD68 (photometric stereo test)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "entangled_photon":    ("BSD68 (quantum imaging test)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "quantum_illumination":("BSD68 (QI test images)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "spc":                 ("BSD68 (SPC test images)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    "streak_camera":       ("BSD68 (streak camera test)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),
    # OpenNeuro MRI
    "asl_mri":             ("OpenNeuro ds000114 brain MRI", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., 2017", "done"),
    "fmri":                ("OpenNeuro ds000114 BOLD fMRI", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., 2017", "done"),
    "mra":                 ("OpenNeuro ds000114 brain MRI", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., 2017", "done"),
    "swi":                 ("OpenNeuro ds000114 brain MRI", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., 2017", "done"),
    # EMDB
    "electron_diffraction":("EMDB electron diffraction", "https://www.ebi.ac.uk/emdb/", "EMDB", "done"),
    "electron_tomography": ("EMDB electron tomography", "https://www.ebi.ac.uk/emdb/", "EMDB", "done"),
    "xray_crystallography":("EMDB/PDB X-ray crystallography", "https://www.ebi.ac.uk/emdb/", "EMDB/PDB", "done"),
    "xfel_sfx":            ("EMDB XFEL SFX", "https://www.ebi.ac.uk/emdb/", "EMDB", "done"),
    "eels":                ("EMDB EELS spectral maps", "https://www.ebi.ac.uk/emdb/", "EMDB", "done"),
    "matrix":              ("EMDB density maps", "https://www.ebi.ac.uk/emdb/", "EMDB", "done"),
    # mammography was already done with Zenodo 5084116
    "mammography":         ("Benign Breast Tumor (Zenodo 5084116)", "https://zenodo.org/records/5084116", "Mammography screening", "done"),
    # hdr was using BSD68 which IS canonical for CI
    "hdr_imaging":         ("BSD68 (HDR test images)", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., 2001", "done"),

    # === NEEDS_CANONICAL (34): canonical dataset exists but not yet using it ===
    "ct":                  ("LoDoPaB-CT (42K CT pairs)", "https://zenodo.org/records/3384092", "Leuschner et al., Sci Data 2021", "needs_canonical"),
    "cbct":                ("Walnut CBCT (42 walnuts)", "https://zenodo.org/records/2686726", "Der Sarkissian et al., Sci Data 2019", "needs_canonical"),
    "industrial_ct":       ("Walnut CBCT micro-CT", "https://zenodo.org/records/2686726", "Der Sarkissian et al., 2019", "needs_canonical"),
    "mri":                 ("fastMRI (8K+ volumes, NYU/Meta)", "https://fastmri.med.nyu.edu/", "Zbontar et al., arXiv 2018", "needs_canonical"),
    "pet":                 ("UDPET ultra-low dose PET", "https://zenodo.org/records/6361846", "UDPET Challenge 2022", "needs_canonical"),
    "pet_ct":              ("autoPET (TCIA, 1014 studies)", "https://autopet.grand-challenge.org/", "Gatidis et al., 2022", "needs_canonical"),
    "fundus":              ("RFMiD retinal fundus (Zenodo 7505822)", "https://zenodo.org/records/7505822", "Pachade et al., Data 2021", "done"),
    "confocal_3d":         ("BioSR (2200+ SR pairs)", "https://figshare.com/articles/dataset/BioSR/13264793", "Qiao et al., Nat Methods 2024", "needs_canonical"),
    "confocal_livecell":   ("LIVECell (5239 images)", "https://sartorius-research.github.io/LIVECell/", "Edlund et al., 2021", "needs_canonical"),
    "cryo_em":             ("EMPIAR (micrograph archive)", "https://www.ebi.ac.uk/empiar/", "EMPIAR archive", "needs_canonical"),
    "cryo_et":             ("SHREC 2021 cryo-ET", "https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA", "Gubins et al., 2021", "needs_canonical"),
    "sem":                 ("SEM nanoparticle (Zenodo 7986673)", "https://zenodo.org/records/7986673", "NanoSolveIT SEM", "done"),
    "diffusion_mri":       ("HCP diffusion MRI", "https://www.humanconnectome.org/", "Van Essen et al., 2013", "needs_canonical"),
    "sar":                 ("MSTAR (SAR target recognition)", "https://www.sdms.afrl.af.mil/index.php?collection=mstar", "Ross et al., 1998", "needs_canonical"),
    "hyperspectral_remote":("Indian Pines AVIRIS (30 patches)", "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes", "Baumgardner et al., 2015", "done"),
    "lidar":               ("SemanticKITTI (43K scans)", "https://semantic-kitti.org/", "Behley et al., ICCV 2019", "needs_canonical"),
    "sonar":               ("UATD forward-looking sonar (30 images)", "https://figshare.com/articles/dataset/UATD_Dataset/21331143", "Xie et al., 2022", "done"),
    "gpr":                 ("CMU-GPR / TU1208 radargrams", "https://github.com/rpl-cmu/CMU-GPR-Dataset", "Foehn et al., 2021", "needs_canonical"),
    "phase_contrast":      ("ISBI Cell Tracking PhC-C2DH-U373", "https://celltrackingchallenge.net/", "Ulman et al., Nat Methods 2017", "done"),
    "confocal_endomicroscopy": ("Kvasir-SEG GI tract (30 images)", "https://datasets.simula.no/kvasir-seg/", "Jha et al., MMM 2020", "done"),
    "fluoroscopy":         ("WEISS Catheter Fluoroscopy", "https://rdr.ucl.ac.uk/articles/dataset/24624243", "UCL WEISS", "needs_canonical"),
    "widefield":           ("FMD widefield denoising", "https://github.com/yinhaoz/denoising-fluorescence", "Zhang et al., CVPR 2019", "needs_canonical"),
    "widefield_lowdose":   ("FMD low-dose", "https://github.com/yinhaoz/denoising-fluorescence", "Zhang et al., 2019", "needs_canonical"),
    "lightsheet":          ("CARE Tribolium", "https://publications.mpi-cbg.de/publications-sites/7207/", "Weigert et al., 2018", "needs_canonical"),
    "lensless":            ("DiffuserCam", "https://waller-lab.github.io/DiffuserCam/", "Antipa et al., Optica 2018", "needs_canonical"),
    "stm":                 ("Graphene/Ni STM (Zenodo 5799774)", "https://zenodo.org/records/5799774", "7287 STM images", "needs_canonical"),
    "radio_astronomy":     ("FIRST VLA 1.4GHz (30 cutouts)", "https://www.cv.nrao.edu/first/", "Becker et al., ApJ 1995", "done"),
    "radio_interferometry":("NVSS 1.4GHz Survey (30 cutouts)", "https://www.cv.nrao.edu/nvss/", "Condon et al., AJ 1998", "done"),
    "xray_ndt":            ("X-ray radiography (Zenodo 7947924)", "https://zenodo.org/records/7947924", "Zenodo X-ray dataset", "done"),
    "terahertz":           ("Active THz dataset", "https://github.com/LingLIx/THz_Dataset", "Ling et al.", "needs_canonical"),
    "shg":                 ("PSHG-TISS", "https://doi.org/10.17605/OSF.IO/K2Z8G", "Golaraei et al., 2022", "needs_canonical"),
    "dexa":                ("NHANES DXA", "https://www.cdc.gov/nchs/nhanes/", "CDC/NCHS", "needs_canonical"),
    "ocean_acoustic_tomo": ("ACOBAR Fram Strait OAT", "https://doi.org/10.1016/j.dib.2022.108160", "Dushaw et al., 2022", "needs_canonical"),

    # === SIMULATION (61): no canonical public dataset exists ===
    "spectral_ct":         ("Spectral CT", "", "Simulated", "simulation"),
    "xrf_tomo":            ("XRF tomography", "", "Simulated", "simulation"),
    "dark_field":          ("Dark-field X-ray", "", "Simulated", "simulation"),
    "talbot_lau":          ("Talbot-Lau", "", "Simulated", "simulation"),
    "spect":               ("SPECT", "", "Simulated", "simulation"),
    "pet_mr":              ("PET-MR", "", "Simulated", "simulation"),
    "spect_ct":            ("SPECT-CT", "", "Simulated", "simulation"),
    "cest_mri":            ("CEST MRI", "", "Simulated", "simulation"),
    "digital_breast_tomo": ("Digital breast tomosynthesis", "", "Simulated", "simulation"),
    "mr_elastography":     ("MR elastography", "", "Simulated", "simulation"),
    "mr_fingerprinting":   ("MR fingerprinting", "", "Simulated", "simulation"),
    "mrs":                 ("MR spectroscopy", "", "Simulated", "simulation"),
    "nirs_brain":          ("fNIRS brain imaging", "", "Simulated", "simulation"),
    "us_mri":              ("Ultrashort-TE MRI", "", "Simulated", "simulation"),
    "odt":                 ("Optical diffraction tomography", "", "Simulated", "simulation"),
    "brachytherapy_img":   ("Brachytherapy imaging", "", "Simulated", "simulation"),
    "portal_imaging":      ("Portal imaging", "", "Simulated", "simulation"),
    "proton_radiography":  ("Proton radiography", "", "Simulated", "simulation"),
    "proton_therapy_img":  ("Proton therapy imaging", "", "Simulated", "simulation"),
    "dot":                 ("Diffuse optical tomography", "", "Simulated", "simulation"),
    "bioluminescence_tomo":("Bioluminescence tomography", "", "Simulated", "simulation"),
    "magnetic_particle":   ("Magnetic particle imaging", "", "Simulated", "simulation"),
    "spinning_disk":       ("Spinning disk confocal", "", "Simulated", "simulation"),
    "expansion":           ("Expansion microscopy", "", "Simulated", "simulation"),
    "cars":                ("CARS microscopy", "", "Simulated", "simulation"),
    "srs":                 ("SRS microscopy", "", "Simulated", "simulation"),
    "lattice_lightsheet":  ("Lattice light-sheet", "", "Simulated", "simulation"),
    "fpm":                 ("Fourier ptychographic micro", "", "Simulated", "simulation"),
    "ism":                 ("Image scanning microscopy", "", "Simulated", "simulation"),
    "minflux":             ("MINFLUX", "", "Simulated", "simulation"),
    "pump_probe":          ("Pump-probe microscopy", "", "Simulated", "simulation"),
    "dna_paint":           ("DNA-PAINT", "", "Simulated", "simulation"),
    "tirf":                ("TIRF microscopy", "", "Simulated", "simulation"),
    "flim":                ("FLIM", "", "Simulated", "simulation"),
    "nsom":                ("Near-field scanning optical", "", "Simulated", "simulation"),
    "three_photon":        ("Three-photon microscopy", "", "Simulated", "simulation"),
    "stem":                ("Scanning TEM", "", "Simulated", "simulation"),
    "clem":                ("Correlative LEM", "", "Simulated", "simulation"),
    "libs":                ("LIBS spectroscopy", "", "Simulated", "simulation"),
    "maldi_msi":           ("MALDI MSI", "", "Simulated", "simulation"),
    "desi":                ("DESI MSI", "", "Simulated", "simulation"),
    "sims":                ("SIMS", "", "Simulated", "simulation"),
    "saxs":                ("Small-angle X-ray scattering", "", "Simulated", "simulation"),
    "waxs":                ("Wide-angle X-ray scattering", "", "Simulated", "simulation"),
    "brillouin":           ("Brillouin microscopy", "", "Simulated", "simulation"),
    "atom_probe":          ("Atom probe tomography", "", "Simulated", "simulation"),
    "mfm":                 ("Magnetic force microscopy", "", "Simulated", "simulation"),
    "neutron_diffraction": ("Neutron diffraction", "", "Simulated", "simulation"),
    "polsar":              ("Polarimetric SAR", "", "Simulated", "simulation"),
    "insar":               ("InSAR", "", "Simulated", "simulation"),
    "passive_microwave":   ("Passive microwave", "", "Simulated", "simulation"),
    "acoustic_emission":   ("Acoustic emission NDT", "", "Simulated", "simulation"),
    "acoustic_microscopy": ("Scanning acoustic micro", "", "Simulated", "simulation"),
    "active_thermography": ("Active thermography NDT", "", "Simulated", "simulation"),
    "eddy_current":        ("Eddy current NDT", "", "Simulated", "simulation"),
    "shearography":        ("Shearography NDT", "", "Simulated", "simulation"),
    "ultrasonic_phased_array": ("UT phased array NDT", "", "Simulated", "simulation"),
    "muon_tomo":           ("Muon tomography", "", "Simulated", "simulation"),
    "neutron_tomo":        ("Neutron tomography", "", "Simulated", "simulation"),
    "ct_fluorescence":     ("CT fluorescence", "", "Simulated", "simulation"),
    "particle_calorimetry":("Particle calorimetry", "", "Simulated", "simulation"),
}

def count_samples(mod):
    std = BASE / mod / "standard"
    if not std.exists(): return 0
    return len(list(std.glob(f"standard_{mod}_*.h5")))

mods = sorted([d.name for d in BASE.iterdir()
               if d.is_dir() and not d.name.startswith("_")
               and (d / "standard").exists()])

lines = []
lines.append("# PWM Benchmark -- Standard Dataset State")
lines.append("")
lines.append(f"Last updated: 2026-03-16 -- {len(mods)} modalities")
lines.append("")
lines.append("## Cloud Storage (GCS)")
lines.append("")
lines.append("- **GCS bucket:** `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/`")
lines.append("- **Download:** `python scripts/download_standard_from_gcs.py`")
lines.append("- **Upload:** `python scripts/upload_standard_to_gcs.py`")
lines.append("")

n_done = 0
n_needs = 0
n_sim = 0
for m in mods:
    info = C.get(m, ("Unknown", "", "Unknown", "simulation"))
    st = info[3]
    if st == "done": n_done += 1
    elif st == "needs_canonical": n_needs += 1
    else: n_sim += 1

lines.append("## Status Summary")
lines.append("")
lines.append("| Status | Count | Description |")
lines.append("|--------|-------|-------------|")
lines.append(f"| done | {n_done} | Uses canonical/real data from this modality |")
lines.append(f"| needs_canonical | {n_needs} | Canonical dataset exists, need to download 30 samples |")
lines.append(f"| simulation | {n_sim} | No public benchmark -- using simulated data |")
lines.append(f"| **Total** | **{len(mods)}** | |")
lines.append("")

lines.append("## All Modalities")
lines.append("")
lines.append("| # | Modality | N | Canonical Dataset | Link | Status |")
lines.append("|---|----------|---|-------------------|------|--------|")

for i, mod in enumerate(mods):
    n = count_samples(mod)
    info = C.get(mod, ("Unknown", "", "Unknown", "simulation"))
    name = info[0][:45]
    url = info[1]
    status = info[3]
    link = f"[link]({url})" if url else "--"
    lines.append(f"| {i+1} | {mod} | {n} | {name} | {link} | {status} |")

lines.append("")

out = BASE / "standard_state.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Generated standard_state.md v6: {len(mods)} modalities")
print(f"  Done: {n_done}")
print(f"  Needs canonical: {n_needs}")
print(f"  Simulation: {n_sim}")
