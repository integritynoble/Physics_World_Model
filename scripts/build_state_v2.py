#!/usr/bin/env python3
"""Build comprehensive state.md with 4-state tracking for all 168 modalities.

States:
  1. Dataset: public (≥10), dev (20), hidden (20) created locally
  2. Benchmark: modality page updated at pwm.platformai.org/benchmark
  3. GPU Tests: algorithm tests run on GPU server (from JSON)
  4. SpecLab: full reconstruction suite on main server

This script reads:
  - benchmark_results/comprehensive_algorithm_test.json
  - datasets/benchmark/{mod}/public/ (sample counts)
  - benchmarks/configs/{mod}.yaml (solver definitions)
  - benchmark_results/benchmark_leaderboard_reference.json (leaderboard data)
"""

import json
import os
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALGO_JSON = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"
LEADERBOARD_JSON = ROOT / "benchmark_results" / "benchmark_leaderboard_reference.json"
CONFIGS_DIR = ROOT / "benchmarks" / "configs"
DATASETS_DIR = ROOT / "datasets" / "benchmark"
OUTPUT = ROOT / "datasets" / "benchmark" / "state.md"

# Canonical public dataset for each modality (most popular accepted datasets)
# Based on check.md section 5 and benchmarks literature
CANONICAL_DATASETS = {
    "acoustic_emission":      "CAME (Component Analysis for Micro-Earthquake) / AEDB",
    "acoustic_microscopy":    "Zenodo AM benchmark / custom lab datasets",
    "active_thermography":    "IR-thermography IRT benchmark / EXTRACT dataset",
    "adaptive_optics":        "SciPy/WFS simulation data / SAXO+ datasets",
    "afm":                    "OpenAFM benchmark / NIST AFM standard",
    "angiography":            "CTA/DSA datasets — VESSEL12 / CAT08",
    "asl_mri":                "Human Connectome Project ASL / OpenNeuro",
    "atom_probe":             "IVAS dataset / FAU Erlangen dataset",
    "bioluminescence_tomo":   "BLT simulation benchmark / Caltech BLT",
    "brachytherapy_img":      "TG-43 phantom / AAPM TG-186 dataset",
    "brillouin":              "Zenodo Brillouin benchmark / stimulated data",
    "cacti":                  "Six Scenes / DAVIS / CODED dataset (Hitomi et al. 2011)",
    "cars":                   "CARS spectral benchmark / SRS-CARS dataset",
    "cassi":                  "CAVE hyperspectral / KAIST scene dataset",
    "cathodoluminescence":    "CL-EM benchmark / Zeiss CL dataset",
    "cbct":                   "TCIA Head-Neck CBCT / CBCT-AAPM challenge",
    "cest_mri":               "CEST simulation benchmark / CEST MRI Challenge",
    "ceus":                   "CAMUS ultrasound / MICCAI CEUS challenge",
    "clem":                   "OpenOrganelle CLEM / CryoET Data Portal CLEM",
    "coded_exposure":         "Deblurring dataset (Levin 2009) / Kohler 2012",
    "confocal_3d":            "OpenCell 3D confocal / HPA 3D confocal dataset",
    "confocal_endomicroscopy":"UCL pCLE dataset / Mauna Kea nCLE data",
    "confocal_livecell":      "LiveCell dataset / CTC Cell Tracking Challenge",
    "coronagraphy":           "HST coronagraph archive / GPI GPIES dataset",
    "cryo_em":                "EMPIAR-10028 (TRPV1) / EMDB GroEL / SHREC19",
    "cryo_et":                "SHREC 2021 / EMPIAR-10045 / IsoNet dataset",
    "ct":                     "LoDoPaB-CT (Leuschner 2021, Zenodo 3384092)",
    "ct_fluorescence":        "CT-FMT simulation benchmark / FLECT dataset",
    "cup":                    "CUP benchmark / STAMP dataset",
    "dark_field":             "Talbot-Lau dark-field benchmark / Munich DFI data",
    "desi":                   "MetaboLights DESI dataset / METLIN-DESI",
    "dexa":                   "OsteoArthritis Initiative (OAI) DXA / NHANES DXA",
    "dic":                    "SciPy phase benchmark / ACPA DIC dataset",
    "diffusion_mri":          "Human Connectome Project dMRI / Sherbrooke-3T",
    "digital_breast_tomo":    "INBreast / VDM-100 DBT dataset",
    "dna_paint":              "DNA-PAINT sim benchmark / Jungmann lab data",
    "doppler_ultrasound":     "PHANTOMNET / EchoNet-Dynamic / MICCAI Doppler",
    "dot":                    "DOT simulation benchmark / UCL DOT dataset",
    "ebsd":                   "DREAM.3D synthetic / Neper EBSD benchmark",
    "eddy_current":           "ECT benchmark / Rolls-Royce NDT dataset",
    "edx_mapping":            "NIST EDX SRM / Hyperspy EDX demo dataset",
    "eels":                   "EELS.info database / Cornell EELS dataset",
    "eht_imaging":            "EHT 2019 M87 data / ngEHT simulated dataset",
    "elastography":           "MRE NIST phantom / MICCAI Elastography dataset",
    "electron_diffraction":   "CIF/ICSD simulation / RRUFF electron diffraction",
    "electron_holography":    "EMDB holo dataset / FZJ Juelich holography data",
    "electron_tomography":    "EMPIAR-10005 / EMPIAR-10045 / EMDB tilt series",
    "endoscopy":              "Kvasir-SEG / CholecT50 / Hyper-Kvasir",
    "entangled_photon":       "NIST quantum imaging data / simulation benchmark",
    "event_camera":           "DAVIS 240C / N-MNIST / MVSEC event dataset",
    "expansion":              "ExPath benchmark / Allen Institute ExM dataset",
    "fib_sem":                "OpenOrganelle FIB-SEM / Janelia FIB-SEM (H01)",
    "flash_lidar":            "KITTI LiDAR / Middlebury flash dataset",
    "flim":                   "FLIM-FRET benchmark / FLUTE dataset",
    "fluoroscopy":            "TCIA Fluoroscopy / CVC-ClinicDB fluoroscopy",
    "fmri":                   "Human Connectome Project fMRI / OpenNeuro BOLD",
    "fpm":                    "FPM benchmark (Tian 2014) / UCB FPM dataset",
    "ftir_imaging":           "USGS spectral library / SFDB FTIR benchmark",
    "fundus":                 "DRIVE / STARE / CHASE_DB1 fundus dataset",
    "fwi":                    "OpenFWI / SEG-SALT / SEAM dataset",
    "gaussian_splatting":     "Tanks & Temples / Mip-NeRF360 / Blender NeRF",
    "ghost_imaging":          "Ghost imaging simulation / NIST quantum dataset",
    "gpr":                    "GPR simulation benchmark / ISAP GPR dataset",
    "gravitational_wave":     "LIGO O3 public data / GW event catalog GWTC-3",
    "hdr_imaging":            "HDR-DB (Fairchild) / HDREye / EMPA HDR dataset",
    "holography":             "HoloPy dataset / DHM benchmark dataset",
    "hyperspectral_remote":   "AVIRIS Indian Pines / ROSIS Pavia / HSRS-MT",
    "impedance_tomo":         "EIDORS simulation / Finnish EIT challenge",
    "industrial_ct":          "GCPD industrial CT / Zeiss Xradia dataset",
    "insar":                  "Sentinel-1 SAR archive / COSAR benchmark",
    "integral":               "EPFL integral imaging / Stanford LF archive",
    "ism":                    "ISM benchmark / Oxford ISM simulation data",
    "ivus":                   "IVUS segmentation challenge / MICCAI 2011 IVUS",
    "lattice_lightsheet":     "Allen Institute lattice LS / Janelia LLC data",
    "lensless":               "DiffuserCam (Monakhova 2019) / PhlatCam dataset",
    "libs":                   "LIBS spectral database / NIST LIBS database",
    "lidar":                  "KITTI LiDAR / nuScenes LiDAR / SemanticKITTI",
    "light_field":            "Stanford Light Field Archive / INRIA LF dataset",
    "lightsheet":             "Allen Brain Atlas LS / Zebrafish SPIM dataset",
    "lucky_imaging":          "Lucky imaging benchmark / Palomar speckle data",
    "machine_vision":         "MVTec Anomaly Detection / BSDS500",
    "magnetic_particle":      "MPI reconstruction challenge / OpenMPIData",
    "maldi_msi":              "MetaboLights MSI / METLIN imaging dataset",
    "mammography":            "VinDr-Mammo / CBIS-DDSM / INBreast",
    "matrix":                 "Matrix completion benchmark / CASC dataset",
    "mfm":                    "MFM simulation / NanoWorld MFM calibration",
    "minflux":                "MINFLUX benchmark / Gottingen MINFLUX data",
    "mr_elastography":        "MRE-NIST phantom / RSNA MRE challenge",
    "mr_fingerprinting":      "MRF simulation (Ma 2013) / CPMG relaxometry",
    "mra":                    "TOF-MRA dataset / MICCAI vessel challenge",
    "mri":                    "FastMRI (Zbontar 2018) multi-coil k-space",
    "mrs":                    "MRSHUB dataset / BIG-PRESS simulation",
    "multispectral_sat":      "Sentinel-2 / WorldView-3 / DESIS hyperspectral",
    "muon_tomo":              "Muon tomography simulation / CERN muon data",
    "nerf":                   "NeRF Blender / LLFF / DTU MVS dataset",
    "neutron_diffraction":    "SINQ / ILL neutron diffraction / CrysAlis",
    "neutron_tomo":           "ILL ICON neutron CT / PSI NEUTRA dataset",
    "nirs_brain":             "fNIRS benchmark / LABBRAIN fNIRS dataset",
    "nsom":                   "NSOM simulation / Witec NSOM benchmark data",
    "ocean_acoustic_tomo":    "OAT simulation benchmark / NOAA acoustic data",
    "ocean_color":            "NASA MODIS ocean color / SeaWiFS dataset",
    "oct":                    "RETOUCH / Duke OCT / OPTIMA OCT dataset",
    "octa":                   "ROSE dataset / CAVF OCTA benchmark",
    "odt":                    "2D DIC/ODT benchmark / Toulouse ODT dataset",
    "palm_storm":             "SMLM Challenge 2016 / Thunderstorm benchmark",
    "panorama":               "SUN360 / LAVAL HDR Panorama dataset",
    "particle_calorimetry":   "GEANT4 simulation / CERN CaloChallenge 2022",
    "passive_microwave":      "AMSR2 / SSMIS passive microwave NASA",
    "pet":                    "TCIA-PET LIDC / OpenPET simulation data",
    "pet_ct":                 "TCIA PET-CT / MAASTRO PET-CT dataset",
    "pet_mr":                 "MICCAI PET-MR / BrainPET dataset",
    "phase_contrast":         "CXLS phase contrast / APS phase contrast data",
    "phase_retrieval":        "CDI benchmark / Phase retrieval algorithm tests",
    "photoacoustic":          "MICCAI PATATO / PAT-Public dataset",
    "photometric_stereo":     "DiLiGenT-MV / CyclesPS benchmark",
    "polarization":           "Polarization benchmark / AOLP dataset",
    "polsar":                 "UAVSAR / SIR-C PolSAR / AIRSAR dataset",
    "portal_imaging":         "EPID benchmark / AAPM portal imaging TG58",
    "proton_radiography":     "FLASH proton CT simulation / GSI proton data",
    "proton_therapy_img":     "Proton CT simulation / TOPAS benchmark",
    "ptychography":           "CDI / FXI ptychography benchmark / CXLS data",
    "pump_probe":             "Ultrafast spectroscopy simulation data",
    "quantum_illumination":   "Quantum imaging simulation benchmark",
    "radio_astronomy":        "LOFAR / VLA FIRST / ALMA calibration data",
    "radio_interferometry":   "MeerKAT / VLBI imaging challenge 2022",
    "raman_imaging":          "RRUFF Raman database / NIST Raman benchmark",
    "sar":                    "Sentinel-1 GRD / UAVSAR / ERS-2 SAR archive",
    "saxs":                   "cSAXS synchrotron / ALS SAXS dataset",
    "seismic_tomo":           "IRIS seismic / SEG-Y NCEDC dataset",
    "sem":                    "SEM-CIFA dataset / NIST SEM calibration",
    "shearography":           "Shearography simulation / LTI lab dataset",
    "shg":                    "SHG microscopy benchmark / collagen data",
    "sim":                    "SIMbench dataset / Allen SIM data",
    "sims":                   "SIMS imaging database / IFM-Stuttgart SIMS",
    "solar_imaging":          "SDO AIA EUV / SOHO EIT / TRACE solar data",
    "sonar":                  "NOAA sonar archive / ARIS multibeam sonar",
    "spc":                    "SPC simulation benchmark / Rice SPC dataset",
    "spect":                  "SIMIND simulation / GATE SPECT benchmark",
    "spect_ct":               "TCIA SPECT-CT / Philips IQ-SPECT dataset",
    "spectral_ct":            "AAPM Spectral CT / Medipix spectral CT data",
    "spinning_disk":          "Spinning disk benchmark / Zeiss LSM dataset",
    "srs":                    "SRS benchmark / coherent Raman dataset",
    "sted":                   "STED benchmark / Leica Abberior STED data",
    "stem":                   "AAEM STEM benchmark / NIST STEM dataset",
    "stm":                    "STM database / NIST surface topography data",
    "streak_camera":          "Streak camera simulation benchmark",
    "structured_light":       "SL benchmark (Gupta 2012) / CAVE SL dataset",
    "swi":                    "SWI benchmark / OpenNeuro SWI dataset",
    "talbot_lau":             "Munich Talbot-Lau grating dataset / PSI data",
    "tem":                    "EMPIAR TEM benchmark / JEOL TEM data",
    "terahertz":              "THz-TDS benchmark / NIST THz dataset",
    "three_photon":           "3PM simulation / Kleinfeld lab 3PM dataset",
    "tirf":                   "TIRF benchmark / Cell-TIRF dataset",
    "tof_camera":             "ETH3D / Middlebury 3D ToF dataset",
    "two_photon":             "Allen Brain 2P / Carandini-Harris 2P dataset",
    "ultrasonic_phased_array":"PAUT benchmark / NDT phased array dataset",
    "ultrasound":             "CAMUS / Echonet-Dynamic / CARDIAC US dataset",
    "us_mri":                 "Ultrashort TE MRI benchmark / PETRA dataset",
    "waxs":                   "SAXS/WAXS synchrotron / ESRF WAXS archive",
    "weather_radar":          "NEXRAD WSR-88D / MetOffice C-band radar data",
    "widefield":              "BSDS / MitoCheck widefield benchmark",
    "widefield_lowdose":      "Low-dose fluorescence benchmark / CARE dataset",
    "xfel_sfx":               "CFEL SFX benchmark / LCLS SFX dataset",
    "xray_crystallography":   "CIF / PDB (Protein Data Bank) / CSD dataset",
    "xray_ndt":               "ASTM NDT benchmark / Zeiss Xradia NDT data",
    "xray_radiography":       "RSNA Bone Age / Chest X-ray14 / PadChest",
    "xrf_imaging":            "XRF benchmark / ESRF XRF dataset",
    "xrf_tomo":               "XRF-CT benchmark / APS XRF-CT dataset",
}


def get_dataset_status(mod):
    """Check local dataset status for a modality."""
    base = DATASETS_DIR / mod
    if not base.exists():
        return {"pub": 0, "dev": 0, "hid": 0}
    result = {}
    for tier, key in [("public", "pub"), ("dev", "dev"), ("hidden", "hid")]:
        tp = base / tier
        if tp.is_dir():
            result[key] = len([d for d in os.listdir(tp) if d.startswith("sample")])
        else:
            result[key] = 0
    return result


def get_algo_results(mod, algo_data):
    """Get best GPU test result for a modality."""
    # Handle alias mapping
    aliases = {"cassi": "sd_cassi", "spc": "spc_kronecker"}
    key = aliases.get(mod, mod)
    info = algo_data.get(key, algo_data.get(mod, {}))
    if not info:
        return None, 0

    best_psnr = None
    n_completed = 0
    for solver, sdata in info.get("solvers", {}).items():
        status = sdata.get("status", "")
        if status in ("completed", "done"):
            n_completed += 1
            p = sdata.get("psnr_db", sdata.get("psnr", None))
            if p is not None and (best_psnr is None or p > best_psnr):
                best_psnr = p
    return best_psnr, n_completed


def load_yaml_solver_count(mod):
    """Count solvers defined in YAML config."""
    cfg = CONFIGS_DIR / f"{mod}.yaml"
    if not cfg.exists():
        return 0
    with open(cfg) as f:
        data = yaml.safe_load(f)
    solvers = data.get("solvers", {}) or {}
    return len(solvers)


def load_leaderboard_ref(mod, lb_data):
    """Get leaderboard reference count for a modality."""
    if mod in lb_data:
        return len(lb_data[mod].get("algorithms", []))
    return 0


def fmt_psnr(psnr):
    if psnr is None:
        return "—"
    if psnr == float("inf"):
        return "inf"
    return f"{psnr:.1f}"


def dataset_state_icon(pub, dev, hid):
    """Return state icon for dataset creation."""
    if pub >= 10 and dev >= 20 and hid >= 20:
        return "✅"
    elif pub > 0 or dev > 0 or hid > 0:
        return f"🔄 pub={pub}/10 dev={dev}/20 hid={hid}/20"
    else:
        return "❌"


def algo_state_icon(n_completed, best_psnr):
    if n_completed > 0:
        return f"✅ {n_completed} solvers, best={fmt_psnr(best_psnr)} dB"
    return "❌"


def main():
    # Load algo test results
    with open(ALGO_JSON) as f:
        algo_data_full = json.load(f)
    algo_data = algo_data_full.get("modalities", {})

    # Load leaderboard reference
    lb_data = {}
    if LEADERBOARD_JSON.exists():
        with open(LEADERBOARD_JSON) as f:
            lb_data = json.load(f)

    # Get sorted modality list from YAML configs
    mods = sorted(
        [f[:-5] for f in os.listdir(CONFIGS_DIR)
         if f.endswith(".yaml") and not f.startswith("_")]
    )

    lines = []
    lines.append("# PWM Benchmark — Dataset & Pipeline State\n\n")
    lines.append(f"Last updated: 2026-03-11 — {len(mods)} modalities\n\n")
    lines.append("## Pipeline States\n\n")
    lines.append("Each modality tracks 4 pipeline stages:\n\n")
    lines.append("1. **Dataset** — public (≥10 samples), dev (20), hidden (20) created\n")
    lines.append("2. **Benchmark** — modality page live at https://pwm.platformai.org/benchmark\n")
    lines.append("3. **GPU Tests** — all YAML-defined solvers tested on GPU server\n")
    lines.append("4. **SpecLab** — full reconstruction suite running on main server\n\n")
    lines.append("Icons: ✅ done | 🔄 in progress | ❌ pending\n\n")
    lines.append("---\n\n")
    lines.append("## Quick Status Table\n\n")
    lines.append("| Modality | Public Dataset | Stage 1: Dataset | Stage 2: Benchmark | Stage 3: GPU Tests | Stage 4: SpecLab |\n")
    lines.append("|----------|---------------|------------------|-------------------|--------------------|-----------------|\n")

    summary = {
        "dataset_done": 0,
        "benchmark_done": 0,
        "gpu_done": 0,
        "speclab_done": 0,
    }

    mod_details = []
    for mod in mods:
        ds = get_dataset_status(mod)
        pub, dev, hid = ds["pub"], ds["dev"], ds["hid"]
        ds_icon = dataset_state_icon(pub, dev, hid)
        ds_done = pub >= 10 and dev >= 20 and hid >= 20

        best_psnr, n_completed = get_algo_results(mod, algo_data)
        gpu_icon = algo_state_icon(n_completed, best_psnr)
        gpu_done = n_completed > 0

        # Benchmark page — currently pending for all (no verification mechanism)
        bench_icon = "❌"
        # SpecLab — pending for all
        speclab_icon = "❌"

        canonical = CANONICAL_DATASETS.get(mod, "—")

        if ds_done:
            summary["dataset_done"] += 1
        if gpu_done:
            summary["gpu_done"] += 1

        lines.append(
            f"| {mod} | {canonical} | {ds_icon} | {bench_icon} | {gpu_icon} | {speclab_icon} |\n"
        )
        mod_details.append((mod, ds, best_psnr, n_completed, canonical))

    lines.append("\n")
    lines.append(f"**Summary:** {summary['dataset_done']}/168 datasets done | "
                 f"{summary['gpu_done']}/168 GPU tests done\n\n")
    lines.append("---\n\n")

    # Detailed section by category
    lines.append("## CT Dataset (Reference Implementation)\n\n")
    ct_ds = get_dataset_status("ct")
    lines.append(f"- Public: {ct_ds['pub']} samples (LoDoPaB-CT Shepp-Logan fallback — need real zips)\n")
    lines.append(f"- Dev: {ct_ds['dev']} samples\n")
    lines.append(f"- Hidden: {ct_ds['hid']} samples\n")
    lines.append(f"- Structure: per-sample dirs with groundtruth.npy, measurement.npy, images/\n")
    lines.append(f"- GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ct/\n\n")
    lines.append("**NOTE:** Public tier currently uses Shepp-Logan synthetic fallback.\n")
    lines.append("To use real LoDoPaB-CT data, download to `datasets/benchmark/ct/lodopab_src/`:\n")
    lines.append("```bash\n")
    lines.append("wget 'https://zenodo.org/api/records/3384092/files/ground_truth_test.zip/content' \\\n")
    lines.append("     -O datasets/benchmark/ct/lodopab_src/ground_truth_test.zip\n")
    lines.append("python datasets/benchmark/ct/generate_dataset.py\n")
    lines.append("```\n\n")
    lines.append("---\n\n")

    # Modalities needing datasets (167 pending)
    lines.append("## Modalities Needing Datasets (167 pending)\n\n")
    lines.append("All modalities except CT need public/dev/hidden datasets generated.\n")
    lines.append("Dataset generation scripts: `scripts/generate_batch{1-12}_datasets.py`\n\n")
    lines.append("| Modality | Public Dataset (Canonical) | Notes |\n")
    lines.append("|----------|---------------------------|-------|\n")
    for mod, ds, best_psnr, n_completed, canonical in mod_details:
        if mod == "ct":
            continue
        pub = ds["pub"]
        if pub < 10:
            lines.append(f"| {mod} | {canonical} | needs generation |\n")
    lines.append("\n---\n\n")

    # GPU algo test results
    lines.append("## GPU Algorithm Test Results\n\n")
    lines.append("Tests run: 2026-03-11 | GPU: NVIDIA GTX 1660 Ti, CUDA 12.4\n\n")
    lines.append("| Modality | Solvers Completed | Best PSNR (dB) |\n")
    lines.append("|----------|------------------|---------------|\n")
    for mod, ds, best_psnr, n_completed, canonical in mod_details:
        psnr_str = fmt_psnr(best_psnr)
        lines.append(f"| {mod} | {n_completed} | {psnr_str} |\n")
    lines.append("\n---\n\n")
    lines.append("*Generated by scripts/build_state_v2.py*\n")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Written: {OUTPUT}")
    print(f"Summary: {summary['dataset_done']}/168 datasets done, "
          f"{summary['gpu_done']}/168 GPU tests done")


if __name__ == "__main__":
    main()
