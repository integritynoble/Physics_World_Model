"""Fix standard_state.md to reflect actual data sources (real vs synthetic)."""
import re, pathlib

STATE = pathlib.Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark\standard_state.md")

text = STATE.read_text(encoding="utf-8")

# ── 1. Fix header ──
text = text.replace(
    "Last updated: 2026-03-13 — 168 modalities (cassi ✅, cacti ✅, sd_cassi ✅ built)",
    "Last updated: 2026-03-13 — 168 modalities | 16 ✅ real-public | 13 🔧 built-synthetic | 101 🔄 source-id | 6 ⚠️ weak | 32 ❌ pending"
)

# ── 2. Add 🔧 to legend ──
text = text.replace(
    "| ✅ done | Famous verified public source · dataset built · uploaded to GCS |\n| 🔄 building |",
    "| ✅ done | Real data from famous verified public source · dataset built · ready for GCS |\n| 🔧 built-syn | Dataset built with synthetic/simulated data — needs real public data to be ✅ |\n| 🔄 building |"
)

# ── 3. Fix statuses: ✅ → 🔧 for synthetic datasets ──
# Each tuple: (old_snippet, new_snippet) — unique substring per row

fixes = [
    # coded_exposure: actually uses skimage test images, not GoPro
    (
        "| 20 | **coded_exposure** | GoPro Deblurring dataset | Nah CVPR 2017; github.com/SeungjunNah/DeepDeblur-PyTorch | 10 | (720,1280,3) | ✅ done |",
        "| 20 | **coded_exposure** | GoPro Deblurring (target) | **Built:** skimage test images + flutter-shutter forward model (synthetic) | 10 | (720,1280,3) | 🔧 built-syn |"
    ),
    # ct: actually uses Shepp-Logan phantom
    (
        "| 27 | **ct** | LoDoPaB-CT (validation split) | Leuschner Sci. Data 2021; doi:10.1038/s41597-021-00893-z; Zenodo 3384092 | 10 | (362,362) sinogram→image | ✅ done |",
        "| 27 | **ct** | LoDoPaB-CT (target) | **Built:** Shepp-Logan phantom + Radon transform (synthetic); need real LoDoPaB-CT | 10 | (362,362) sinogram→image | 🔧 built-syn |"
    ),
    # mri: actually uses Shepp-Logan multi-coil
    (
        "| 91 | **mri** | fastMRI multi-coil k-space (knee) | fastmri.med.nyu.edu; doi:10.1148/ryai.2020190007 | 10 | (320,320,15,C) k-space | ✅ done |",
        "| 91 | **mri** | fastMRI knee (target) | **Built:** Shepp-Logan phantom + multi-coil FFT (synthetic); need real fastMRI | 10 | (320,320,15,C) k-space | 🔧 built-syn |"
    ),
    # oct: synthetic retinal layers
    (
        "| 102 | **oct** | RETOUCH OCT challenge | Bogunovic IVCM 2019; doi:10.1109/TMI.2019.2901970 | 10 | (496,512) B-scan | ✅ done |",
        "| 102 | **oct** | RETOUCH OCT (target) | **Built:** Synthetic retinal layer model + axial PSF (synthetic); need real RETOUCH | 10 | (496,512) B-scan | 🔧 built-syn |"
    ),
    # photoacoustic: synthetic vessels
    (
        "| 114 | **photoacoustic** | OADAT photoacoustic benchmark | Jaeger OADAT 2022; doi:10.1088/1361-6420/acad6c | 10 | (28,2030) PA sinogram | ✅ done |",
        "| 114 | **photoacoustic** | OADAT (target) | **Built:** Synthetic vessel structures + PA forward model (synthetic); need real OADAT | 10 | (28,2030) PA sinogram | 🔧 built-syn |"
    ),
    # solar_imaging: synthetic EUV
    (
        "| 136 | **solar_imaging** | SDO AIA public archive | lmsal.com/sdo; doi:10.1007/s11207-011-9834-2 | 10 | (4096,4096) EUV (crop 512×512) | ✅ done |",
        "| 136 | **solar_imaging** | SDO AIA (target) | **Built:** Synthetic EUV solar disk (synthetic); need real NASA SDO/AIA data | 10 | (4096,4096) EUV (crop 512×512) | 🔧 built-syn |"
    ),
    # ultrasound: synthetic PICMUS phantom
    (
        "| 158 | **ultrasound** | PICMUS / CAMUS | Liebgott IUS 2016; doi:10.1109/TUFFC.2019.2917338 | 10 | (2030,128) IQ | ✅ done |",
        "| 158 | **ultrasound** | PICMUS (target) | **Built:** Synthetic tissue phantom + plane-wave PSF (synthetic); need real PICMUS | 10 | (2030,128) IQ | 🔧 built-syn |"
    ),
]

# Fix 🔄 → 🔧 for modalities that are BUILT but with synthetic data
fixes += [
    (
        "| 55 | **fluoroscopy** | CVC-ClinicDB (Bernal CMIG 2015) | polyp.cv.uab.es; doi:10.1016/j.compmedimag.2015.02.007 | 10 | (288,384) X-ray | 🔄 building |",
        "| 55 | **fluoroscopy** | CVC-ClinicDB (target) | **Built:** Synthetic chest/abdomen phantom + Beer-Lambert (synthetic) | 10 | (288,384) X-ray | 🔧 built-syn |"
    ),
    (
        "| 60 | **fwi** | OpenFWI benchmark | Deng IEEE TGRS 2021; doi:10.1109/TGRS.2021.3124783 | 10 | (70,70) velocity | 🔄 building |",
        "| 60 | **fwi** | OpenFWI (target) | **Built:** Synthetic velocity models + straight-ray traveltime (synthetic) | 10 | (70,70) velocity | 🔧 built-syn |"
    ),
    (
        "| 66 | **holography** | HoloPy benchmark | holopy.readthedocs.io; DHM simulation | 10 | (256,256) hologram | 🔄 building |",
        "| 66 | **holography** | HoloPy (target) | **Built:** Synthetic scatterers + Fraunhofer diffraction (synthetic) | 10 | (256,256) hologram | 🔧 built-syn |"
    ),
    (
        "| 106 | **panorama** | SUN360 (Xiao CVPR 2012) | vision.cs.princeton.edu/projects/2012/SUN360; doi:10.1109/CVPR.2012.6247971 | 10 | (512,1024,3) equirect | 🔄 building |",
        "| 106 | **panorama** | SUN360 (target) | **Built:** Synthetic UV-sphere equirectangular + perspective projection (synthetic) | 10 | (512,1024,3) equirect | 🔧 built-syn |"
    ),
    (
        "| 109 | **pet** | TCIA LIDC-IDRI PET | Clark Sci. Data 2013; doi:10.1038/sdata.2015.17 | 10 | (128,128,L) PET | 🔄 building |",
        "| 109 | **pet** | TCIA PET (target) | **Built:** Shepp-Logan + FDG hot-spots + Radon per-slice (synthetic) | 10 | (128,128,L) PET | 🔧 built-syn |"
    ),
    (
        "| 130 | **seismic_tomo** | IRIS SEED / Marmousi-2 model | iris.edu; SEG-Y open data; doi:10.1190/1.9781560801528.ch2 | 10 | (737,3000) velocity | 🔄 building |",
        "| 130 | **seismic_tomo** | Marmousi-2 (target) | **Built:** Synthetic Marmousi-style velocity + ray-based Radon (synthetic) | 10 | (737,3000) velocity | 🔧 built-syn |"
    ),
]

# Fix fundus source (DRIVE listed but actually CHASE_DB1)
fixes += [
    (
        "| 59 | **fundus** | DRIVE retinal vessel dataset | Staal IEEE TMI 2004; doi:10.1109/TMI.2004.825627 | 10 | (584,565,3) fundus | ✅ done |",
        "| 59 | **fundus** | CHASE_DB1 retinal vessel dataset | Owen IEEE TBE 2009; doi:10.1109/TMI.2012.2205687; kingston.ac.uk/CHASE_DB1 | 10 | (584,565,3) fundus | ✅ done |"
    ),
]

# Add source info to correctly-marked ✅ done ones that were built
# (endoscopy, lensless, machine_vision, radio_astronomy are already ✅ and real — just add "Built:" prefix)

for old, new in fixes:
    if old in text:
        text = text.replace(old, new)
        print(f"  Fixed: {new.split('**')[1]}")
    else:
        # Try to find the modality row
        mod = old.split('**')[1] if '**' in old else "?"
        print(f"  WARNING: Could not find row for {mod}")

# ── 4. Fix summary ──
old_summary = """| Status | Count | Description |
|--------|-------|-------------|
| ✅ done | 10 | cassi, cacti, sd_cassi, gravitational_wave, matrix, nerf, hyperspectral_remote, eht_imaging, particle_calorimetry, cryo_em |
| 🔄 building | 114 | Famous/verified source identified, pending build |
| ⚠️ weak-src | 6 | Source identified but not from major benchmark |
| ❌ pending | 38 | No dominant public dataset; simulation only |

**Coverage:** 130/168 (77%) have a verifiable famous public source"""

new_summary = """| Status | Count | Description |
|--------|-------|-------------|
| ✅ done | 16 | cassi, cacti, sd_cassi, spc_kronecker, cryo_em, endoscopy, fundus, eht_imaging, gravitational_wave, hyperspectral_remote, lensless, machine_vision, matrix, nerf, particle_calorimetry, radio_astronomy |
| 🔧 built-syn | 13 | coded_exposure, ct, fluoroscopy, fwi, holography, mri, oct, panorama, pet, photoacoustic, seismic_tomo, solar_imaging, ultrasound |
| 🔄 building | 101 | Famous/verified source identified, pending build + download |
| ⚠️ weak-src | 6 | Source identified but not from major benchmark |
| ❌ pending | 32 | No dominant public dataset; simulation only |

**Coverage:** 16/168 (9.5%) verified done with real public data · 29/168 (17%) have built datasets"""

text = text.replace(old_summary, new_summary)
print("  Fixed: summary section")

# ── 5. Update footer date ──
text = text.replace(
    "*Standard dataset tracking — PWM Benchmark — 2026-03-13*",
    "*Standard dataset tracking — PWM Benchmark — Chengshuai Yang — 2026-03-13*"
)

STATE.write_text(text, encoding="utf-8")
print("\nDone — standard_state.md updated.")
