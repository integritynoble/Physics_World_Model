# Category 4: Coherent Imaging — Full Case Catalog

> 5 modalities, 20 system variants, 5,120 total test instances

---

## 4.1 Ptychographic Imaging

### System Variants (5)

| # | Variant | DAG | Physical Elements | Key Difference |
|---|---------|-----|-------------------|----------------|
| 1 | **Far-field X-ray Ptychography** | Src(synchrotron_10keV) -> M(probe_aperture) -> P(Fresnel_propagation) -> D(CCD_2048x2048) | Focused X-ray probe + area detector | Most common; PtychoNN benchmark |
| 2 | **Near-field Ptychography** | Src(synchrotron) -> M(probe) -> P(Fresnel_short_distance) -> D(CCD) | Short propagation; Fresnel zone | Relaxed overlap requirement |
| 3 | **Bragg Ptychography** | Src(synchrotron) -> M(crystal_probe) -> R(Bragg_diffraction) -> D(CCD) | Near Bragg peak; crystalline samples | 3D strain/displacement mapping |
| 4 | **Electron Ptychography (4D-STEM)** | Src(electron_gun_200keV) -> M(probe_STEM) -> P(elastic_scattering) -> D(pixelated_256x256) | Electron beam + pixelated detector | 4D dataset: scan(x,y) + diffraction(kx,ky) |
| 5 | **Fourier Ptychography (FPM)** | Src(LED_array_15x15) -> M(thin_sample) -> P(objective_pupil) -> D(sCMOS) | LED array illumination | Wide-field + high-resolution synthesis |

### Sizes: 128x128, 256x256, 512x512 (3)
### Scan positions: 16, 64, 225 (3)

### Mismatch Parameters (5)

| # | Parameter | Range | Unit |
|---|-----------|-------|------|
| 1 | Probe position error | [0, 5] | px |
| 2 | Probe intensity variation | [0.8, 1.2] | relative |
| 3 | Background scatter | [0, 0.1] | relative |
| 4 | Detector saturation | [0, 1%] | fraction |
| 5 | Coherence degradation | [0, 0.3] | partial coherence |

### Case Count: 5 x 3 x 3 x 4 x 5 = **900 per benchmark**

### Data Sources

| Source | Label | URL |
|--------|-------|-----|
| PtychoNN tungsten test sample | `WEB` | https://github.com/mcherukara/PtychoNN/tree/master/data |
| CXIDB (Coherent X-ray Imaging Data Bank) | `WEB` | http://cxidb.org |
| Ptychography 4.0 datasets | `WEB` | https://ptychography-4-0.github.io/ptychography/datasets.html |
| Waller Lab FPM | `WEB` | https://github.com/Waller-Lab/FPM |
| Generated (Shepp-Logan + CDI simulation) | `GEN` | Custom diffraction simulation |

---

## 4.2 Digital Holographic Microscopy

### System Variants (4)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Off-axis DHM** | Src(HeNe_633nm) -> P(sample_path) + P(reference_tilted_3deg) -> Sigma(interference) -> D(CCD) | Tilted reference beam; single-shot |
| 2 | **In-line (Gabor)** | Src(laser) -> P(sample_path) -> D(CCD) | Coaxial reference; twin image artifact |
| 3 | **Lensless DHM (DLHM)** | Src(point_source) -> P(sample_Fresnel) -> D(CCD_no_lens) | No objective lens; long working distance |
| 4 | **Electron Holography** | Src(electron_gun) -> P(biprism_electrostatic) -> D(CCD) | TEM + electrostatic biprism |

### Sizes: 256x256, 512x512, 1024x1024 (3)
### Case Count: 4 x 3 x 4 x 5 = **240 per benchmark**

### Data Sources

| Source | Label | URL |
|--------|-------|-----|
| DHM RBC dataset (760+609 holograms) | `WEB` | https://www.nature.com/articles/s41597-023-02818-4 |
| USC-SIPI Image Database | `SYN-WEB` | https://sipi.usc.edu/database/ |
| Generated (angular spectrum propagation) | `GEN` | Custom code |

---

## 4.3 Coherent Diffractive Imaging / Phase Retrieval

### System Variants (4): Plane-wave CDI, Bragg CDI, Fresnel CDI, Sparsity CDI
### Sizes: 128x128, 256x256, 512x512 (3)
### Case Count: 4 x 3 x 4 x 5 = **240 per benchmark**

### Data Sources

| Source | Label | URL |
|--------|-------|-----|
| CXIDB | `WEB` | http://cxidb.org |
| PyNX simulation | `SYN-WEB` | https://gitlab.esrf.fr/favre/pynx |
| Generated HIO/ER simulation | `GEN` | Custom iterative phase retrieval |

---

## 4.4 Optical Diffraction Tomography (ODT)

### System Variants (3): Rotating illumination, Rotating sample, Mixed
### Sizes: 128x128x128, 256x256x256 (2)
### Case Count: 3 x 2 x 4 x 5 = **120 per benchmark**

---

## 4.5 Talbot-Lau X-ray Grating Interferometry

### System Variants (3): Analyzer-based, Talbot (2-grating), Talbot-Lau (3-grating)
### Case Count: 3 x 2 x 4 x 5 = **120 per benchmark**

---

## Category 4 Summary

| Modality | Variants | Per-Benchmark | Total (4 benchmarks) |
|----------|----------|---------------|----------------------|
| Ptychography | 5 | 900 | ~3,700 |
| Holography | 4 | 240 | ~1,000 |
| CDI/Phase Retrieval | 4 | 240 | ~1,000 |
| ODT | 3 | 120 | ~500 |
| Talbot-Lau | 3 | 120 | ~500 |
| **TOTAL** | **19** | **1,620** | **~6,700** |
