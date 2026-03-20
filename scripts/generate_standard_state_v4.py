"""Generate standard_state.md v4 with canonical dataset links and status."""
import json, os, h5py
from pathlib import Path

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")

# ============================================================
# CANONICAL DATASET REGISTRY with LINKS
# (canonical_name, canonical_url, reference, current_match)
# ============================================================
CANONICAL = {
    # === MEDICAL ===
    "ct": ("LoDoPaB-CT", "https://zenodo.org/records/3384092", "Leuschner et al., Sci Data 2021", False),
    "cbct": ("Walnut CBCT", "https://zenodo.org/records/2686726", "Der Sarkissian et al., Sci Data 2019", False),
    "industrial_ct": ("Walnut CBCT", "https://zenodo.org/records/2686726", "Der Sarkissian et al., Sci Data 2019", False),
    "spectral_ct": ("Spectral CT (niche)", "", "N/A -- niche", False),
    "dexa": ("NHANES DXA", "https://www.cdc.gov/nchs/nhanes/", "CDC/NCHS", False),
    "xrf_tomo": ("XRF tomography (niche)", "", "N/A -- niche", False),
    "dark_field": ("Dark-field X-ray (niche)", "", "N/A -- niche", False),
    "talbot_lau": ("Talbot-Lau (niche)", "", "N/A -- niche", False),
    "phase_contrast": ("ISBI Cell Tracking", "https://celltrackingchallenge.net/", "Ulman et al., Nat Methods 2017", False),
    "pet": ("UDPET", "https://zenodo.org/records/6361846", "UDPET Challenge 2022", False),
    "spect": ("SIMIND SPECT (niche)", "", "N/A -- niche", False),
    "pet_ct": ("autoPET", "https://autopet.grand-challenge.org/", "Gatidis et al., Sci Data 2022", False),
    "pet_mr": ("PET-MR (niche)", "", "N/A -- niche", False),
    "spect_ct": ("SPECT-CT (niche)", "", "N/A -- niche", False),
    "mri": ("fastMRI", "https://fastmri.med.nyu.edu/", "Zbontar et al., arXiv 2018", False),
    "asl_mri": ("OpenNeuro ds000114", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., Sci Data 2017", True),
    "cest_mri": ("CEST MRI (niche)", "", "N/A -- niche", False),
    "diffusion_mri": ("HCP Diffusion", "https://www.humanconnectome.org/", "Van Essen et al., NeuroImage 2013", False),
    "digital_breast_tomo": ("VICTRE DBT (niche)", "", "N/A -- niche", False),
    "fmri": ("OpenNeuro ds000114", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., Sci Data 2017", True),
    "mr_elastography": ("MRE (niche)", "", "N/A -- niche", False),
    "mr_fingerprinting": ("MRF (niche)", "", "N/A -- niche", False),
    "mra": ("OpenNeuro ds000114", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., Sci Data 2017", True),
    "mrs": ("MRS (niche)", "", "N/A -- niche", False),
    "nirs_brain": ("fNIRS (niche)", "", "N/A -- niche", False),
    "swi": ("OpenNeuro ds000114", "https://openneuro.org/datasets/ds000114", "Gorgolewski et al., Sci Data 2017", True),
    "us_mri": ("UTE MRI (niche)", "", "N/A -- niche", False),
    "fundus": ("DRIVE", "https://drive.grand-challenge.org/", "Staal et al., IEEE TMI 2004", False),
    "angiography": ("ARCADE", "https://zenodo.org/records/10390295", "Popov et al., Sci Data 2023", True),
    "oct": ("OCTDL / Kermany OCT2017", "https://www.kaggle.com/datasets/paultimothymooney/kermany2018", "Kermany et al., Cell 2018", True),
    "octa": ("OCTA-Mosaicking", "https://zenodo.org/records/14333858", "OCTA retinal mosaicking", True),
    "odt": ("ODT (niche)", "", "N/A -- niche", False),
    "mammography": ("CBIS-DDSM", "https://www.cancerimagingarchive.net/collection/cbis-ddsm/", "Lee et al., Sci Data 2017", False),
    "endoscopy": ("Kvasir-SEG", "https://datasets.simula.no/kvasir-seg/", "Jha et al., MMM 2020", True),
    "confocal_endomicroscopy": ("CVC-ClinicDB", "https://polyp.grand-challenge.org/CVCClinicDB/", "Bernal et al., 2015", False),
    "fluoroscopy": ("WEISS Catheter Fluoroscopy", "https://rdr.ucl.ac.uk/articles/dataset/24624243", "UCL WEISS", False),
    "ultrasound": ("BUSI", "https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset", "Al-Dhabyani et al., Data in Brief 2020", True),
    "doppler_ultrasound": ("BUS-BRA", "https://zenodo.org/records/7730709", "Gomez-Flores et al., Data in Brief 2023", True),
    "ceus": ("BUS-BRA", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", True),
    "elastography": ("BUS-BRA", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", True),
    "ivus": ("BUS-BRA", "https://zenodo.org/records/7730709", "Gomez-Flores et al., 2023", True),
    "impedance_tomo": ("KTC 2023 EIT", "https://zenodo.org/records/10986692", "Hauptmann et al., KTC 2023", True),
    "brachytherapy_img": ("Brachytherapy (niche)", "", "N/A -- niche", False),
    "portal_imaging": ("Portal imaging (niche)", "", "N/A -- niche", False),
    "proton_radiography": ("Proton radiography (niche)", "", "N/A -- niche", False),
    "proton_therapy_img": ("Proton therapy (niche)", "", "N/A -- niche", False),
    "dot": ("DOT (niche)", "", "N/A -- niche", False),
    "bioluminescence_tomo": ("BLT (niche)", "", "N/A -- niche", False),
    "magnetic_particle": ("MPI (niche)", "", "N/A -- niche", False),

    # === MICROSCOPY ===
    "confocal_3d": ("BioSR", "https://figshare.com/articles/dataset/BioSR/13264793", "Qiao et al., Nat Methods 2024", False),
    "confocal_livecell": ("LIVECell", "https://sartorius-research.github.io/LIVECell/", "Edlund et al., Nat Methods 2021", False),
    "lightsheet": ("CARE Tribolium", "https://publications.mpi-cbg.de/publications-sites/7207/", "Weigert et al., Nat Methods 2018", False),
    "two_photon": ("CaImAn / Neurofinder", "https://github.com/flatironinstitute/CaImAn", "Giovannucci et al., eLife 2019", True),
    "widefield": ("FMD", "https://github.com/yinhaoz/denoising-fluorescence", "Zhang et al., CVPR 2019", False),
    "widefield_lowdose": ("FMD low-dose", "https://github.com/yinhaoz/denoising-fluorescence", "Zhang et al., CVPR 2019", False),
    "spinning_disk": ("Spinning disk (niche)", "", "N/A -- niche", False),
    "dic": ("BBBC003", "https://bbbc.broadinstitute.org/BBBC003", "Ljosa et al., Nat Methods 2012", True),
    "expansion": ("Expansion microscopy (niche)", "", "N/A -- niche", False),
    "cars": ("CARS (niche)", "", "N/A -- niche", False),
    "shg": ("PSHG-TISS", "https://doi.org/10.17605/OSF.IO/K2Z8G", "Golaraei et al., Sci Data 2022", False),
    "srs": ("SRS (niche)", "", "N/A -- niche", False),
    "lattice_lightsheet": ("Lattice LS (niche)", "", "N/A -- niche", False),
    "fpm": ("FPM (niche)", "", "N/A -- niche", False),
    "ism": ("ISM (niche)", "", "N/A -- niche", False),
    "minflux": ("MINFLUX (niche)", "", "N/A -- niche", False),
    "pump_probe": ("Pump-probe (niche)", "", "N/A -- niche", False),
    "dna_paint": ("DNA-PAINT (niche)", "", "N/A -- niche", False),
    "tirf": ("TIRF (niche)", "", "N/A -- niche", False),
    "flim": ("FLIM (niche)", "", "N/A -- niche", False),
    "nsom": ("NSOM (niche)", "", "N/A -- niche", False),
    "three_photon": ("Three-photon (niche)", "", "N/A -- niche", False),
    "sim": ("BioSR / UniFMIR", "https://zenodo.org/records/8420100", "Li et al., UniFMIR", True),
    "sted": ("UniFMIR", "https://zenodo.org/records/8420100", "Li et al., UniFMIR", True),
    "palm_storm": ("SMLM / STORM tubulin", "https://zenodo.org/records/7620025", "Sage et al., Nat Methods 2019", True),

    # === ELECTRON MICROSCOPY ===
    "sem": ("NFFA-EUROPE SEM", "https://b2share.eudat.eu/records/f1aa0f5ad38c456eaf7b04d47a65af53", "Aversa et al., Sci Data 2018", False),
    "tem": ("ISBI 2012 EM / Zenodo 11188503", "https://zenodo.org/records/11188503", "Arganda-Carreras et al., 2015", True),
    "stem": ("STEM (niche)", "", "N/A -- niche", False),
    "cryo_em": ("EMPIAR / CryoPPP", "https://www.ebi.ac.uk/empiar/", "EMPIAR archive", False),
    "cryo_et": ("SHREC 2021", "https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA", "Gubins et al., 2021", False),
    "fib_sem": ("FIB-SEM (Zenodo 8114392)", "https://zenodo.org/records/8114392", "FIB-SEM golgi+granules", True),
    "electron_diffraction": ("EMDB electron diffraction", "https://www.ebi.ac.uk/emdb/", "EMDB", True),
    "electron_holography": ("Zenodo 18289938", "https://zenodo.org/records/18289938", "Latychevskaia et al.", True),
    "electron_tomography": ("EMDB ET", "https://www.ebi.ac.uk/emdb/", "EMDB", True),
    "clem": ("CLEM (niche)", "", "N/A -- niche", False),
    "ptychography": ("PtychoNN / Zenodo 16263064", "https://zenodo.org/records/16263064", "Cherukara et al., 2020", True),

    # === SPECTROSCOPY/PROBE ===
    "afm": ("AFM Zenodo 60434", "https://zenodo.org/records/60434", "Oxvig et al.", True),
    "stm": ("Graphene/Ni STM", "https://zenodo.org/records/5799774", "Zenodo 5799774", False),
    "edx_mapping": ("EDX Zenodo 14960843", "https://zenodo.org/records/14960843", "BSE-EDS ROI", True),
    "eels": ("EMDB EELS", "https://www.ebi.ac.uk/emdb/", "EMDB", True),
    "xrf_imaging": ("XRF Zenodo 4005031", "https://zenodo.org/records/4005031", "Synchrotron XRF", True),
    "raman_imaging": ("Raman Zenodo 8141012", "https://zenodo.org/records/8141012", "SRS photothermal", True),
    "ftir_imaging": ("FTIR Zenodo 4986399", "https://zenodo.org/records/4986399", "Breast tissue FTIR", True),
    "cathodoluminescence": ("CL Zenodo 6801483", "https://zenodo.org/records/6801483", "Zircon CL", True),
    "libs": ("LIBS (niche)", "", "N/A -- niche", False),
    "maldi_msi": ("MALDI MSI (niche)", "", "N/A -- niche", False),
    "desi": ("DESI MSI (niche)", "", "N/A -- niche", False),
    "sims": ("SIMS (niche)", "", "N/A -- niche", False),
    "saxs": ("SAXS (niche)", "", "N/A -- niche", False),
    "waxs": ("WAXS (niche)", "", "N/A -- niche", False),
    "brillouin": ("Brillouin (niche)", "", "N/A -- niche", False),
    "atom_probe": ("Atom probe (niche)", "", "N/A -- niche", False),
    "mfm": ("MFM (niche)", "", "N/A -- niche", False),
    "neutron_diffraction": ("Neutron diff (niche)", "", "N/A -- niche", False),
    "ebsd": ("EBSD Zenodo 1214829", "https://zenodo.org/records/1214829", "Deformed iron EBSD", True),
    "xray_crystallography": ("EMDB/PDB", "https://www.ebi.ac.uk/emdb/", "EMDB/PDB", True),
    "xfel_sfx": ("EMDB XFEL", "https://www.ebi.ac.uk/emdb/", "EMDB", True),

    # === COMPUTATIONAL OPTICS ===
    "holography": ("DHM Zenodo 18289938", "https://zenodo.org/records/18289938", "Latychevskaia et al.", True),
    "phase_retrieval": ("Phase Zenodo 13771363", "https://zenodo.org/records/13771363", "Holographic phase", True),
    "light_field": ("Stanford Lytro / HCI", "http://lightfields.stanford.edu/LF2016.html", "Honauer et al., ACCV 2016", True),
    "coded_exposure": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "cup": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "ghost_imaging": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "hdr_imaging": ("Kalantari HDR", "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/", "Kalantari, SIGGRAPH 2017", False),
    "integral": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "panorama": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "photometric_stereo": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "entangled_photon": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "quantum_illumination": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "spc": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "streak_camera": ("BSD68", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/", "Martin et al., ICCV 2001", True),
    "lensless": ("DiffuserCam", "https://waller-lab.github.io/DiffuserCam/", "Antipa et al., Optica 2018", False),
    "polarization": ("Zenodo 4483248", "https://zenodo.org/records/4483248", "Polarization demosaic", True),
    "event_camera": ("EDHT21 DVS", "https://zenodo.org/records/4918320", "DVS event camera", True),
    "machine_vision": ("FaultSeg", "https://zenodo.org/records/13162335", "FaultSeg wheel defect", True),
    "photoacoustic": ("Duke PAM", "https://zenodo.org/records/4042171", "Vu et al., 2020", True),

    # === REMOTE SENSING ===
    "sar": ("MSTAR", "https://www.sdms.afrl.af.mil/index.php?collection=mstar", "Ross et al., 1998", False),
    "polsar": ("PolSAR (niche)", "", "N/A -- niche", False),
    "insar": ("InSAR (niche)", "", "N/A -- niche", False),
    "hyperspectral_remote": ("Indian Pines", "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes", "Baumgardner et al., 2015", False),
    "multispectral_sat": ("EuroSAT", "https://zenodo.org/records/7711810", "Helber et al., IEEE JSTARS 2019", True),
    "lidar": ("SemanticKITTI", "https://semantic-kitti.org/", "Behley et al., ICCV 2019", False),
    "sonar": ("UATD", "https://figshare.com/articles/dataset/UATD_Dataset/21331143", "Xie et al., Sci Data 2022", False),
    "gpr": ("CMU-GPR", "https://github.com/rpl-cmu/CMU-GPR-Dataset", "Foehn et al., 2021", False),
    "seismic_tomo": ("Marmousi2 / OpenFWI", "https://wiki.seg.org/wiki/Open_data", "Martin et al., Leading Edge 2006", True),
    "fwi": ("OpenFWI", "https://openfwi-lanl.github.io/", "Deng et al., NeurIPS 2022", True),
    "ocean_acoustic_tomo": ("ACOBAR Fram Strait", "https://doi.org/10.1016/j.dib.2022.108160", "Dushaw et al., 2022", False),
    "ocean_color": ("EuroSAT SeaLake", "https://zenodo.org/records/7711810", "Helber et al., 2019", True),
    "passive_microwave": ("EuroSAT proxy", "https://zenodo.org/records/7711810", "Helber et al., 2019", True),
    "flash_lidar": ("Middlebury Stereo", "https://vision.middlebury.edu/stereo/", "Scharstein & Szeliski, 2002", True),
    "structured_light": ("Middlebury Stereo", "https://vision.middlebury.edu/stereo/", "Scharstein & Szeliski, 2002", True),
    "weather_radar": ("SEVIR", "https://registry.opendata.aws/sevir/", "Veillette et al., NeurIPS 2020", False),

    # === ASTRONOMY ===
    "adaptive_optics": ("ESO VLT AO", "https://archive.eso.org/", "ESO archive", True),
    "coronagraphy": ("ESO VLT coronagraph", "https://archive.eso.org/", "ESO archive", True),
    "lucky_imaging": ("ESO lucky imaging", "https://archive.eso.org/", "ESO archive", True),
    "eht_imaging": ("EHT M87", "https://eventhorizontelescope.org/for-astronomers/data", "EHT Collaboration, ApJL 2019", True),
    "solar_imaging": ("NASA SDO AIA", "https://sdo.gsfc.nasa.gov/", "Pesnell et al., Solar Physics 2012", True),
    "gravitational_wave": ("GWOSC", "https://gwosc.org/", "LIGO Scientific, 2021", True),
    "radio_astronomy": ("FIRST VLA", "https://www.cv.nrao.edu/first/", "Becker et al., ApJ 1995", False),
    "radio_interferometry": ("ALMA/VLA archive", "https://almascience.eso.org/", "N/A", False),
    "particle_calorimetry": ("Calorimeter (niche)", "", "N/A -- niche", False),
    "nerf": ("Tiny NeRF Lego", "https://github.com/bmild/nerf", "Mildenhall et al., ECCV 2020", True),
    "gaussian_splatting": ("3DGS Lego", "https://github.com/graphdeco-inria/gaussian-splatting", "Kerbl et al., SIGGRAPH 2023", True),

    # === NDT ===
    "xray_ndt": ("GDXray", "https://domingomery.ing.puc.cl/material/gdxray/", "Mery et al., 2015", False),
    "xray_radiography": ("NIH ChestX-ray14", "https://nihcc.app.box.com/v/ChestXray-NIHCC", "Wang et al., CVPR 2017", True),
    "acoustic_emission": ("AE (niche)", "", "N/A -- niche", False),
    "acoustic_microscopy": ("SAM (niche)", "", "N/A -- niche", False),
    "active_thermography": ("MOPER (niche)", "", "N/A -- niche", False),
    "eddy_current": ("Eddy current (niche)", "", "N/A -- niche", False),
    "shearography": ("Shearography (niche)", "", "N/A -- niche", False),
    "terahertz": ("Active THz", "https://github.com/LingLIx/THz_Dataset", "Ling et al.", False),
    "ultrasonic_phased_array": ("UT phased array (niche)", "", "N/A -- niche", False),
    "muon_tomo": ("Muon tomo (niche)", "", "N/A -- niche", False),
    "neutron_tomo": ("Neutron tomo (niche)", "", "N/A -- niche", False),
    "ct_fluorescence": ("CT fluorescence (niche)", "", "N/A -- niche", False),

    # === SPECIAL ===
    "cacti": ("DeSCI CACTI", "https://github.com/liuyang12/DeSCI", "Liu et al., IEEE TPAMI 2019", True),
    "cassi": ("CAVE 31-band", "https://www.cs.columbia.edu/CAVE/databases/multispectral/", "Yasuma et al., ICIP 2010", True),
    "sd_cassi": ("CAVE 24-band", "https://www.cs.columbia.edu/CAVE/databases/multispectral/", "Yasuma et al., ICIP 2010", True),
    "spc_kronecker": ("Indian Pines AVIRIS", "https://engineering.purdue.edu/~biehl/MultiSpec/", "Baumgardner et al., 2015", True),
    "matrix": ("EMDB", "https://www.ebi.ac.uk/emdb/", "EMDB", True),
}

def count_samples(mod):
    std = BASE / mod / "standard"
    if not std.exists(): return 0
    return len(list(std.glob(f"standard_{mod}_*.h5")))

def get_current_source(mod):
    std = BASE / mod / "standard"
    h5s = sorted(std.glob(f"standard_{mod}_*.h5"))
    if not h5s: return "N/A"
    with h5py.File(str(h5s[0]), "r") as f:
        return f.attrs.get("source", "unknown")[:80]

mods = sorted([d.name for d in BASE.iterdir()
               if d.is_dir() and not d.name.startswith("_")
               and (d / "standard").exists()])

lines = []
lines.append("# PWM Benchmark -- Standard Dataset State")
lines.append("")
lines.append(f"Last updated: 2026-03-16 -- {len(mods)} modalities, 20 samples each")
lines.append("")
lines.append("## Cloud Storage (GCS)")
lines.append("")
lines.append("Standard datasets are stored in Google Cloud Storage (NOT in the GitHub repo).")
lines.append("")
lines.append("- **GCS bucket:** `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/`")
lines.append("- **File types:** `*.h5` (data), `*.json` (metadata), `images/*.png` (previews)")
lines.append("")
lines.append("### Setup on a new server")
lines.append("")
lines.append("```bash")
lines.append("gcloud auth login")
lines.append("python scripts/download_standard_from_gcs.py")
lines.append("python scripts/download_standard_from_gcs.py --modality ct,mri,pet  # specific modalities")
lines.append("```")
lines.append("")

# Status
done_count = sum(1 for m in mods if CANONICAL.get(m, ("","","",False))[3])
niche_count = sum(1 for m in mods if "niche" in str(CANONICAL.get(m, ("","","N/A -- niche",False))[2]).lower() and not CANONICAL.get(m, ("","","",False))[3])
needs = len(mods) - done_count - niche_count

lines.append("## Status Summary")
lines.append("")
lines.append(f"| Status | Count | Description |")
lines.append(f"|--------|-------|-------------|")
lines.append(f"| done | {done_count} | Uses canonical/popular benchmark dataset |")
lines.append(f"| needs_upgrade | {needs} | Popular dataset exists but not yet used |")
lines.append(f"| niche | {niche_count} | No widely-used public dataset for this modality |")
lines.append(f"| **Total** | **{len(mods)}** | |")
lines.append("")

# Full table
lines.append("## All Modalities")
lines.append("")
lines.append("| # | Modality | Samples | Canonical Dataset | Link | Status |")
lines.append("|---|----------|---------|-------------------|------|--------|")

for i, mod in enumerate(mods):
    n = count_samples(mod)
    info = CANONICAL.get(mod, ("Unknown", "", "Unknown", False))
    canon_name = info[0][:40]
    canon_url = info[1]
    is_done = info[3]
    is_niche = "niche" in str(info[2]).lower()

    if is_done:
        status = "done"
    elif is_niche:
        status = "niche"
    else:
        status = "needs_upgrade"

    link_md = f"[link]({canon_url})" if canon_url else "--"
    lines.append(f"| {i+1} | {mod} | {n} | {canon_name} | {link_md} | {status} |")

lines.append("")

# Done list
done_mods = [m for m in mods if CANONICAL.get(m, ("","","",False))[3]]
lines.append(f"## Done ({len(done_mods)} modalities)")
lines.append("")
for m in done_mods:
    info = CANONICAL[m]
    n = count_samples(m)
    link = f" -- [{info[0]}]({info[1]})" if info[1] else f" -- {info[0]}"
    lines.append(f"- **{m}** ({n} samples){link}")

lines.append("")

# Needs upgrade
upgrade_mods = [m for m in mods if not CANONICAL.get(m,("","","",False))[3] and "niche" not in str(CANONICAL.get(m,("","","N/A -- niche",False))[2]).lower()]
lines.append(f"## Needs Upgrade ({len(upgrade_mods)} modalities)")
lines.append("")
for m in upgrade_mods:
    info = CANONICAL[m]
    n = count_samples(m)
    src = get_current_source(m)
    link = f"[{info[0]}]({info[1]})" if info[1] else info[0]
    lines.append(f"- **{m}** ({n} samples): should use {link}")

lines.append("")

# Niche
niche_mods = [m for m in mods if "niche" in str(CANONICAL.get(m,("","","N/A -- niche",False))[2]).lower() and not CANONICAL.get(m,("","","",False))[3]]
lines.append(f"## Niche ({len(niche_mods)} modalities)")
lines.append("")
lines.append("No widely-used public benchmark dataset exists for these modalities:")
lines.append("")
for m in niche_mods:
    n = count_samples(m)
    lines.append(f"- **{m}** ({n} samples)")

out = BASE / "standard_state.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Generated standard_state.md v4: {len(mods)} modalities")
print(f"  Done: {done_count}, Needs upgrade: {needs}, Niche: {niche_count}")
