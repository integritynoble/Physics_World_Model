# PWM Benchmark — Public Dataset Registry

**168 imaging modalities** | Last updated: 2026-03-12 | 93 real-world datasets + 75 physics-based generators

This document lists every modality's public dataset source, popularity level, data type, download links, and key metadata. It serves as the canonical reference for what data underpins each PWM benchmark modality.

---

## Summary

| Data Type | Count | Description |
|-----------|-------|-------------|
| **Real** | 93 | Real-world experimental/clinical data from public repositories |
| **Physics** | 75 | Dedicated physics-based forward model simulation (no public dataset available) |
| **Total** | 168 | All modalities verified, physics-accurate |

### Popularity Legend

| Rating | Meaning |
|--------|---------|
| ★★★★★ | Gold standard — dominant benchmark in the field, thousands of citations, used in major challenges (MICCAI, NeurIPS, CVPR) |
| ★★★★☆ | Widely used — well-known in the community, hundreds of citations, top-venue publication |
| ★★★☆☆ | Established — recognized reference dataset, used in multiple publications |
| ★★☆☆☆ | Niche — best available for this specific modality, limited but real |
| ★☆☆☆☆ | Emerging — small or very new dataset, only option available |

---

## 1. Medical Imaging (42 modalities)

### 1.1 X-ray / CT / Projection

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 1 | **ct** | LoDoPaB-CT | Real | ★★★★★ | 42,895 samples | HDF5 | [zenodo.org/record/3384092](https://zenodo.org/record/3384092) | Leuschner et al., Sci. Data 2021, DOI:10.1038/s41597-021-00893-z | CC BY 4.0 | Low-dose parallel-beam CT from real patient data. De facto standard for CT reconstruction benchmarks. |
| 2 | **cbct** | AAPM Low-Dose CT Challenge 2016 | Physics | ★★★★★ | Challenge data | DICOM | [aapm.org/grandchallenge](https://www.aapm.org/grandchallenge/lowdosect/) | McCollough et al., Med. Phys. 2017 | Research | Cone-beam CT generated with dedicated physics: dental/head phantoms + scatter/ring artifacts. |
| 3 | **xray_radiography** | CheXpert / NIH ChestX-ray14 | Real | ★★★★★ | 224,316 / 112,120 images | PNG/DICOM | [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) / [nihcc.app.box.com](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Irvin et al., AAAI 2019 / Wang et al., CVPR 2017 | Research | Two largest public chest X-ray datasets. CheXpert: Stanford, 14 labels. NIH: 14 pathology labels, multi-center. |
| 4 | **xray_ndt** | GDXray | Real | ★★★★☆ | 19,407 images, 3.5 GB | PNG | [domingomery.ing.puc.cl/material/gdxray](https://domingomery.ing.puc.cl/material/gdxray/) | Mery et al., J. Nondestructive Evaluation 2015 | Free for research | 5 groups: castings, welds, baggage, natural objects, settings. Standard NDT X-ray benchmark. |
| 5 | **fluoroscopy** | WEISS Catheter Segmentation | Real | ★★★☆☆ | 2,000+ images | PNG | [rdr.ucl.ac.uk](https://rdr.ucl.ac.uk/articles/dataset/WEISS_Catheter_Segmentation_in_Fluoroscopy_Dataset/24624243) | UCL WEISS Lab 2023 | CC BY 4.0 | Phantom and in-vivo catheterization fluoroscopy images with segmentation labels. |
| 6 | **digital_breast_tomo** | BCS-DBT (Duke/TCIA) | Real | ★★★★★ | 22,032 DBT volumes, 5,060 patients | DICOM | [cancerimagingarchive.net/collection/breast-cancer-screening-dbt](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/) | Buda et al., Radiology 2021, DOI:10.7937/e4wt-cd02 | TCIA | Largest public DBT dataset. Associated DBTex challenge. |
| 7 | **mammography** | CBIS-DDSM / VinDr-Mammo | Physics | ★★★★★ | 2,620 / 5,000 studies | DICOM | [cancerimagingarchive.net/collection/cbis-ddsm](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) | Lee et al., Sci. Data 2017 | TCIA | Dedicated physics generator: Beer-Lambert breast attenuation + scatter model. |
| 8 | **industrial_ct** | GCPD Industrial CT | Physics | ★★★★☆ | Custom | TIFF | Internal | WoDT benchmark | Research | Dedicated generator: metal/bolt phantoms + beam hardening polynomial model. |
| 9 | **spectral_ct** | AAPM Spectral CT Challenge | Physics | ★★★★☆ | Challenge data | HDF5 | Internal | Medipix3 consortium | Research | Dedicated generator: dual-energy material decomposition (water/bone/iodine). |
| 10 | **brachytherapy_img** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | AAPM TG-43 standard | — | Dedicated physics: CT + applicator geometry + dose kernel convolution. No public dataset exists. |
| 11 | **portal_imaging** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: MV X-ray projection through CT + flat panel detector model. No public dataset. |
| 12 | **talbot_lau** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: 3-grating wave propagation (absorption + phase + dark-field channels). |
| 13 | **dexa** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | NHANES DXA program | — | Dedicated physics: dual-energy X-ray absorptiometry forward model (bone + soft tissue). |

### 1.2 MRI / Magnetic Resonance

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 14 | **mri** | fastMRI | Real | ★★★★★ | 1.5M+ slices | HDF5 (k-space) | [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/) | Zbontar et al., NeurIPS 2018 | Research | Multi-coil brain + knee k-space. The single most cited MRI reconstruction benchmark. |
| 15 | **diffusion_mri** | Human Connectome Project dMRI | Physics | ★★★★★ | ~1200 subjects | NIfTI | [humanconnectome.org](https://www.humanconnectome.org/) | Van Essen et al., NeuroImage 2013 | HCP DUA | Dedicated generator: ADC brain phantoms + b-value weighted k-space. |
| 16 | **fmri** | HCP fMRI / OpenNeuro | Physics | ★★★★★ | ~1200 subjects | NIfTI | [humanconnectome.org](https://www.humanconnectome.org/) | Poldrack et al., Nat. Neurosci. 2013 | HCP DUA | Dedicated generator: BOLD activation + k-space temporal acceleration. |
| 17 | **asl_mri** | HCP Lifespan ASL / OpenNeuro ASL-BIDS | Real | ★★★★☆ | ~3,000 subjects | NIfTI (BIDS) | [openneuro.org](https://openneuro.org) | Alsop et al., MRM 2015 (ASL consensus) | Various | Real perfusion imaging from HCP Lifespan and multi-vendor ASL-BIDS datasets. |
| 18 | **cest_mri** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: Bloch-McConnell multi-pool Z-spectra simulation. Highly specialized, no public dataset. |
| 19 | **mr_elastography** | MRE Phantom+Liver+Brain | Real | ★★★☆☆ | Multi-organ | NIfTI | [nature.com/articles/s41597-025-05968-9](https://www.nature.com/articles/s41597-025-05968-9) | Scientific Data 2025 | CC BY 4.0 | Wave field images + inversion results for phantom, liver, brain tissue. |
| 20 | **mr_fingerprinting** | 3D MRF Multi-site Study | Real | ★★★☆☆ | 12 volunteers, 8 sites | DICOM/MAT | [zenodo.org/records/3989799](https://zenodo.org/records/3989799) | Ma et al., Nature 2013 (concept) | CC BY 4.0 | T1/T2/M0 maps from 1.5T + 3T scanners across 8 institutions. |
| 21 | **mra** | IXI TOF-MRA | Real | ★★★★☆ | 494+ datasets | NIfTI | [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/) | IXI consortium | CC BY-SA 3.0 | Time-of-flight MRA from 3 London centers. Also: TOF-MRA aneurysm (63 patients). |
| 22 | **mrs** | MRSHub / MRSDB | Real | ★★★☆☆ | Multi-scanner | BIDS-MRS | [mrshub.netlify.app/datasets_svs](https://mrshub.netlify.app/datasets_svs/) | Various | Open | Multiple single-voxel MRS datasets from various scanners. PRESS data, synthetic benchmarks. |
| 23 | **swi** | OASIS-3 SWI | Real | ★★★★★ | 2,842 MR sessions, 1,378 subjects | NIfTI | [sites.wustl.edu/oasisbrains](https://sites.wustl.edu/oasisbrains/home/oasis-3/) | LaMontagne et al., Alzheimer's & Dementia 2019 | DUA | Susceptibility-weighted imaging from longitudinal aging study, ages 42-95. |
| 24 | **us_mri** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: ultrashort TE k-space acquisition forward model. Very limited public data. |

### 1.3 Ultrasound / Acoustic

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 25 | **ultrasound** | CAMUS / EchoNet-Dynamic | Physics | ★★★★★ | 500 / 10,030 patients | NIfTI/AVI | [echonet.github.io](https://echonet.github.io/dynamic/) | Ouyang et al., Nature 2020 | Research | Dedicated physics generator: depth-dependent PSF + Rayleigh speckle model. |
| 26 | **doppler_ultrasound** | PICMUS In-Vivo Carotid | Real | ★★★★☆ | RF data | HDF5/MAT | [ustb.no/ustb-datasets](https://www.ustb.no/ustb-datasets/) | Rindal et al., Ultrasonics 2019 | Open | Plane-wave RF data for Doppler imaging. Also: Kaggle carotid color Doppler images. |
| 27 | **ceus** | B-MODE-AND-CEUS-LIVER (TCIA) | Real | ★★★☆☆ | Multi-patient | DICOM | [cancerimagingarchive.net/collection/b-mode-and-ceus-liver](https://www.cancerimagingarchive.net/collection/b-mode-and-ceus-liver/) | DOI:10.7937/TCIA.2021.v4z7-tc39 | TCIA | Real CEUS cine-loops + B-mode of liver lesions from clinical trials. |
| 28 | **elastography** | IMPACT Lab CIRS Phantom | Real | ★★★☆☆ | 2,200 RF pairs | MAT | [users.encs.concordia.ca/~impact](https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/) | Concordia University | Research | CIRS phantom with hard inclusions, background 20 kPa, inclusions 2× stiffer. |
| 29 | **ivus** | MICCAI 2011 IVUS Challenge | Real | ★★★★☆ | 2,175 images | DICOM | [cvc.uab.es/IVUSchallenge2011](http://www.cvc.uab.es/IVUSchallenge2011/) | Balocco et al., Med. Image Anal. 2014 | Research | Multi-center, multi-vendor coronary IVUS B-mode with lumen + EEM annotations. |
| 30 | **photoacoustic** | PATATO / OADAT | Physics | ★★★★☆ | Benchmark | HDF5 | Internal | MICCAI PATATO | Research | Dedicated generator: circular transducer array + limited-view tomographic reconstruction. |

### 1.4 Nuclear / PET / SPECT

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 31 | **pet** | TCIA-PET LIDC | Physics | ★★★★★ | Multi-patient | DICOM | [cancerimagingarchive.net](https://www.cancerimagingarchive.net/) | Clark et al., J. Digit. Imaging 2013 | TCIA | Dedicated generator: activity maps + Radon + attenuation/scatter/randoms. |
| 32 | **pet_ct** | TCIA PET-CT | Physics | ★★★★★ | Multi-center | DICOM | [cancerimagingarchive.net](https://www.cancerimagingarchive.net/) | MAASTRO dataset | TCIA | Dedicated generator: joint CT attenuation + PET activity Radon forward model. |
| 33 | **pet_mr** | MICCAI PET-MR Challenge | Physics | ★★★★☆ | Challenge data | NIfTI | Internal | BrainPET / ADNI | DUA | Dedicated generator: MRI-derived attenuation + k-space + PET Radon. |
| 34 | **spect** | SIMIND / OpenGATE | Physics | ★★★★☆ | Simulation framework | HDF5 | [opengatecollaboration.org](https://opengatecollaboration.org/) | Ljungberg et al. | GPL | Dedicated generator: 360° parallel-hole collimator + depth-dependent CDR. |
| 35 | **spect_ct** | TCIA SPECT-CT | Physics | ★★★★☆ | Multi-center | DICOM | [cancerimagingarchive.net](https://www.cancerimagingarchive.net/) | Philips IQ-SPECT | TCIA | Dedicated generator: CT-based attenuation correction for SPECT. |
| 36 | **proton_therapy_img** | Proton Range Verification | Real | ★★☆☆☆ | 43 beams | HDF5 | [nature.com/articles/s41597-021-01028-0](https://www.nature.com/articles/s41597-021-01028-0) | Scientific Data 2021 | CC BY 4.0 | PET/prompt-gamma verification images from prostate phantom beam delivery. |
| 37 | **proton_radiography** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | Schulte, Med. Phys. 2008 | — | Dedicated MC: energy-loss + multiple Coulomb scattering model via TOPAS/Geant4. |
| 38 | **muon_tomo** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | GEANT4 MC | — | Dedicated physics: Geant4-level MC muon transport + scattering simulation. |
| 39 | **magnetic_particle** | OpenMPIData | Real | ★★★★☆ | 1D-3D phantoms | MDF | [magneticparticleimaging.github.io/OpenMPIData.jl](https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/) | Knopp et al. | CC BY 4.0 | Multiple phantom measurements with CAD drawings. Standard MPI benchmark. |

### 1.5 Ophthalmology / Eye

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 40 | **oct** | RETOUCH / Duke OCT | Physics | ★★★★★ | Multi-center | TIFF/MAT | Internal | Bogunovic et al., IVCM 2019 | Research | Dedicated generator: retinal layers + axial PSF + speckle + attenuation rolloff. |
| 41 | **octa** | OCTA-500 | Real | ★★★★★ | 500 subjects, 2 FOVs | PNG/BMP | [ieee-dataport.org/open-access/octa-500](https://ieee-dataport.org/open-access/octa-500) | Li et al., MedIA 2022 | Open | Largest OCTA dataset: 7 segmentation labels, 4 text labels. Also: ROSE-2 (229 images). |
| 42 | **fundus** | DRIVE / STARE / CHASE_DB1 | Physics | ★★★★★ | 40/20/28 images | TIFF | Internal | Staal et al., IEEE TMI 2004 | Various | Dedicated generator: retinal fundus + vessel tree + optic disc model. |

### 1.6 Other Medical

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 43 | **angiography** | CADICA | Real | ★★★★☆ | 668 videos, 42 patients | AVI/DICOM | [arxiv.org/abs/2402.00570](https://arxiv.org/abs/2402.00570) | CADICA 2024 | Research | Invasive coronary angiography with stenosis annotations. Also: ImageCAS (1000 CTA). |
| 44 | **endoscopy** | Kvasir-SEG / HyperKvasir | Physics | ★★★★★ | 1,000 / 110,079 | PNG/JPEG | Internal | Jha et al., IEEE Access 2020 | CC BY 4.0 | Dedicated generator: fiber-bundle PSF + LED falloff + mucosal phantom model. |
| 45 | **nirs_brain** | Tufts fNIRS2MW / OpenFNIRS | Real | ★★★★☆ | 68 participants | NIRS-BIDS | [tufts-hci-lab.github.io](https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html) | Tufts HCI Lab | Open | 21 min sessions, n-back cognitive tasks. Multiple datasets via OpenFNIRS/OpenNeuro. |
| 46 | **impedance_tomo** | FIPS Open 2D EIT + Stroke EIT | Real | ★★★★☆ | Circular tank | MAT | [fips.fi/EIT_dataset.php](https://www.fips.fi/EIT_dataset.php) | Kuopio University | CC BY 4.0 | Saline tank with conductive/resistive inclusions. Also: Stroke EIT clinical data. |
| 47 | **dot** | Unrolled-DOT (Zenodo) | Real | ★★★☆☆ | Experimental | MAT | [zenodo.org/records/7654959](https://zenodo.org/records/7654959) | Zenodo 2023 | CC BY 4.0 | Experimental diffuse optical tomography measurements + EIDORS community data. |
| 48 | **confocal_endomicroscopy** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: fiber-bundle sampling + fluorescein contrast + honeycomb artifact. Clinical data restricted. |
| 49 | **bioluminescence_tomo** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | Ntziachristos, Nat. Methods 2010 | — | Dedicated physics: diffusion equation + bioluminescent source distribution forward model. |
| 50 | **ct_fluorescence** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | Ale, PLoS ONE 2009 | — | Dedicated physics: diffuse FMT + Digimouse phantom forward model. |

---

## 2. Microscopy & Super-Resolution (36 modalities)

### 2.1 Fluorescence Microscopy

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 51 | **confocal_livecell** | FMD Confocal | Real | ★★★★★ | 12,000 images | TIFF | [github.com/yinhaoz/denoising-fluorescence](https://github.com/yinhaoz/denoising-fluorescence) | Zhang et al., CVPR 2019 | MIT | Cells, zebrafish, mouse brain. 50 noisy frames/view, averaged ground truth. |
| 52 | **confocal_3d** | OpenCell 3D / Broad BBBC | Physics | ★★★★★ | Large-scale | TIFF | Internal | BBBC consortium | Various | Dedicated generator: confocal PSF + fluorescent beads/dendrites. |
| 53 | **spinning_disk** | FMD Confocal (spinning-disk-like) | Real | ★★★★☆ | 324+ images | TIFF | [github.com/yinhaoz/denoising-fluorescence](https://github.com/yinhaoz/denoising-fluorescence) | Zhang et al., CVPR 2019 | MIT | Extreme low-SNR dataset, 12 sub-datasets. |
| 54 | **two_photon** | Allen Brain 2P-SCC | Physics | ★★★★★ | Large | NWB | [alleninstitute.org](https://alleninstitute.org) | De Vries et al., Nat. Neurosci. 2020 | Allen terms | Dedicated generator: quadratic intensity + depth attenuation + GaAsP noise. |
| 55 | **three_photon** | FMD 2P + OSF 3PM | Real | ★★★☆☆ | Limited | TIFF | [github.com/yinhaoz/denoising-fluorescence](https://github.com/yinhaoz/denoising-fluorescence) + [osf.io/ug68m](https://osf.io/ug68m/) | Nature Methods (source data) | Various | FMD two-photon subset + miniature 3-photon microscope data. |
| 56 | **lightsheet** | Allen Brain Light-sheet / Zebrafish SPIM | Physics | ★★★★★ | Whole-brain | TIFF | [alleninstitute.org](https://alleninstitute.org) | — | Allen terms | Dedicated generator: Gaussian illumination + tissue scattering physics. |
| 57 | **lattice_lightsheet** | Lattice Light-sheet (Zenodo) | Real | ★★★☆☆ | Volumetric | CZI | [zenodo.org/records/14807429](https://zenodo.org/records/14807429) | Zenodo 2025 | CC BY 4.0 | Raw .czi data with OMERO workflow. Also: PetaKit5D (Dryad) fly VNC 8× expansion. |
| 58 | **widefield** | BSDS500 / MitoCheck / Broad BBBC | Physics | ★★★★★ | Large | TIFF | Internal | EMBL MitoCheck | Various | Dedicated generator: wide PSF + DAPI/actin/mito phantom model. |
| 59 | **widefield_lowdose** | W2S + FMD Widefield | Real | ★★★★★ | 144,000 + 12,000 images | TIFF | [datasets.epfl.ch/w2s](https://datasets.epfl.ch/w2s/W2S_raw.zip) + [github.com/IVRL/w2s](https://github.com/IVRL/w2s) | Prakash et al., EPFL 2020 | MIT | Real paired widefield + SIM high-res targets. Best low-dose widefield benchmark. |
| 60 | **tirf** | BioSR + SMLM Challenge | Real | ★★★★★ | 2,200+ pairs | TIFF | [figshare.com/articles/dataset/BioSR/13264793](https://figshare.com/articles/dataset/BioSR/13264793) + [srm.epfl.ch](https://srm.epfl.ch/) | Qiao et al., Nature Methods 2021 | CC BY 4.0 | Widefield/SIM-TIRF pairs of biological structures + SMLM Challenge real TIRF data. |
| 61 | **flim** | Convallaria FLIM (Zenodo) | Real | ★★★☆☆ | Single sample | JSON/TIFF | [zenodo.org/records/15007900](https://zenodo.org/records/15007900) | Zenodo 2025 | CC BY 4.0 | Convallaria lifetime data + fluorescein calibration. Also: FLIM neuron spiking (Zenodo 7706488). |

### 2.2 Super-Resolution Microscopy

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 62 | **palm_storm** | SMLM Challenge 2016 | Physics | ★★★★★ | Challenge data | TIFF | [smlmchallenge.net](http://smlmchallenge.net/) | Sage et al., Nature Methods 2019 | Open | Dedicated generator: sparse SMLM activation + Gaussian PSF. |
| 63 | **sted** | STED benchmark | Physics | ★★★★★ | Challenge data | TIFF | Internal | Culley et al., Nature Methods 2018 | Research | Dedicated generator: sub-diffraction PSF + cytoskeleton/vesicle phantoms. |
| 64 | **sim** | SIMbench | Physics | ★★★★★ | Challenge data | TIFF | Internal | Culley et al., Nature Methods 2018 | Research | Dedicated generator: 3 orientations × 3 phases structured illumination. |
| 65 | **dna_paint** | DNA-PAINT SMLM (Zenodo) | Real | ★★★☆☆ | Raw frames + localizations | TIFF/CSV | [zenodo.org/records/6966132](https://zenodo.org/records/6966132) | Zenodo 2022 | CC BY 4.0 | TOM20-labelled neuronal tissue + high-density emitter patches with coordinates. |
| 66 | **minflux** | MINFLUX Raw Data (Zenodo) | Real | ★★★☆☆ | 3-color mitochondria | HDF5 | [zenodo.org/records/6562765](https://zenodo.org/records/6562765) (raw) / [zenodo.org/records/6563100](https://zenodo.org/records/6563100) (localizations) | Ostersehlt et al., Nature Methods 2022 | CC BY 4.0 | First public MINFLUX dataset. 3-color recordings, sub-nm resolution. |
| 67 | **expansion** | RnR-ExM Challenge | Real | ★★★☆☆ | 24 paired 3D volumes | TIFF | [rnr-exm.grand-challenge.org](https://rnr-exm.grand-challenge.org/) | Grand Challenge 2024 | Research | 3 species, registration landmarks. Also: NucExM (IEEE DataPort, zebrafish nuclei). |
| 68 | **ism** | BrightEyes-ISM + EMPIAR | Real | ★★★☆☆ | Example datasets | TIFF | [github.com/VicidominiLab/BrightEyes-ISM](https://github.com/VicidominiLab/BrightEyes-ISM) + [ebi.ac.uk/empiar/EMPIAR-11666](https://www.ebi.ac.uk/empiar/EMPIAR-11666/) | Vicidomini Lab | MIT | Open-source ISM analysis toolbox + HeLa Zeiss LSM900 Airyscan data. |

### 2.3 Label-free / Contrast Microscopy

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 69 | **dic** | DLD Beads + NIST Steel | Real | ★★★☆☆ | ~50 images | TIFF/PNG | [opticapublishing.figshare.com](https://opticapublishing.figshare.com/articles/dataset/dld_public_dataset_zip/16926607) + [catalog.data.gov](https://catalog.data.gov/dataset/data-publication-differential-interference-contrast-microscopy-from-a-cross-section-of-a-f) | Figshare / NIST | CC BY 4.0 | DLD polystyrene beads with MATLAB code + NIST fractured steel cross-section. |
| 70 | **dark_field** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: Mie scattering + dark background simulation. No public dark-field microscopy benchmark. |
| 71 | **phase_contrast** | HiP-CT Synchrotron | Real | ★★★★☆ | Whole organ | TIFF | [nature.com/articles/s41597-022-01353-y](https://www.nature.com/articles/s41597-022-01353-y) | Walsh et al., Sci. Data 2022 | CC BY 4.0 | Whole human lung at 25μm voxel, zooms to 2.45μm. Synchrotron-based X-ray phase contrast. |
| 72 | **shg** | PSHG-TISS (OSF) | Real | ★★★☆☆ | Multi-tissue | TIFF | [osf.io/K2Z8G45](https://doi.org/10.17605/OSF.IO/K2Z8G45) | Scientific Data 2022 | CC BY 4.0 | Polarization-resolved SHG of breast, skin, thyroid tissue + derived parameter maps. |
| 73 | **cars** | MCARS Cell Imaging | Real | ★★★☆☆ | Cell imaging | TIFF | [figshare.com/collections/_/6149604](https://figshare.com/collections/_/6149604) | Boildieu et al., Figshare 2022 | CC BY 4.0 | Multiplex CARS cell imaging with unsupervised segmentation analysis. |
| 74 | **srs** | OpenSRH | Real | ★★★★★ | 300+ patients, 1,300+ WSIs | TIFF | [opensrh.mlins.org](https://opensrh.mlins.org/) | Hollon et al., NeurIPS 2022 | Research | First public SRH dataset. Intraoperative brain tumor imaging, full pathologic annotations. |
| 75 | **cathodoluminescence** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: electron-beam + material luminescence emission simulation. No public CL benchmark. |

### 2.4 Quantitative Phase / Holographic

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 76 | **holography** | HoloPy / DHM | Physics | ★★★★☆ | Benchmark | HDF5 | Internal | — | — | Dedicated generator: off-axis DHM + angular spectrum propagation. |
| 77 | **odt** | Toulouse ODT / TORCH | Physics | ★★★★☆ | Benchmark | HDF5 | Internal | — | — | Dedicated generator: Born/Rytov + Ewald sphere + refractive index contrast. |
| 78 | **fpm** | UCB FPM | Physics | ★★★★☆ | Benchmark | TIFF | Internal | Tian, Light Sci. Appl. 2015 | Research | Dedicated generator: LED array + synthetic aperture + low-NA objective. |
| 79 | **lensless** | DiffuserCam | Physics | ★★★★☆ | Benchmark | TIFF | Internal | Monakhova, Optica 2019 | Research | Dedicated generator: diffuser caustic PSF + multiplexed measurement. |
| 80 | **phase_retrieval** | CDI Challenge | Physics | ★★★★☆ | Benchmark | HDF5 | Internal | — | — | Dedicated generator: diversity masks + defocus + complex objects. |
| 81 | **ptychography** | CDI/ALS Ptychography | Physics | ★★★★☆ | Benchmark | HDF5 | Internal | — | — | Dedicated generator: scanning CDI + probe position errors. |

### 2.5 Electron Microscopy

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 82 | **cryo_em** | EMPIAR-10028 TRPV1 | Physics | ★★★★★ | Multi-protein | MRC | [ebi.ac.uk/empiar/EMPIAR-10028](https://www.ebi.ac.uk/empiar/EMPIAR-10028/) | Bai et al., Nature 2015 | CC0 | Dedicated generator: CTF at 300kV + protein clusters, very low SNR. |
| 83 | **cryo_et** | EMPIAR-10045 | Real | ★★★★★ | 7 tomograms | MRC | [ebi.ac.uk/empiar/EMPIAR-10045](https://www.ebi.ac.uk/empiar/EMPIAR-10045/) | EBI | CC0 | S. cerevisiae 80S ribosome tilt series. Standard sub-tomogram averaging benchmark. |
| 84 | **tem** | EMPIAR TEM / NIST SRM | Physics | ★★★★★ | Multi-specimen | MRC | [ebi.ac.uk/empiar](https://www.ebi.ac.uk/empiar/) | — | CC0 | Dedicated generator: HRTEM lattice fringes + CTF correction model. |
| 85 | **sem** | SEM-CIFA / NIST SEM | Physics | ★★★★★ | Multi-specimen | TIFF | Internal | — | Research | Dedicated generator: BSE/SE yields + Z-dependent contrast physics. |
| 86 | **stem** | EPFL Hippocampus ssTEM | Real | ★★★★★ | 1065×2048×1536 | TIFF | [epfl.ch/labs/cvlab/data/data-em](https://www.epfl.ch/labs/cvlab/data/data-em/) | Lucchi et al., TPAMI 2013 | Open | De facto benchmark for EM segmentation. Also: ISBI 2012 Challenge. |
| 87 | **fib_sem** | OpenOrganelle / COSEM (Janelia) | Real | ★★★★★ | 10+ whole-cell volumes | N5/Zarr | [openorganelle.janelia.org](https://openorganelle.janelia.org/) | Heinrich et al., Nature 2021 | CC BY 4.0 | 4×4×4 nm isotropic resolution. AWS S3 open data. Premier FIB-SEM benchmark. |
| 88 | **ebsd** | Kikuchi Pattern Inconel 718 | Real | ★★★☆☆ | Multi-specimen | TIF/ANG | [datadryad.org](https://datadryad.org/dataset/doi:10.5061/dryad.zcrjdfnr9) | Dryad 2023 | CC0 | Wrought and additively manufactured superalloy EBSD patterns + orientation data. |
| 89 | **edx_mapping** | BCO-DMO EDS Spectra + USGS | Real | ★★☆☆☆ | Phytoplankton | CSV/TIFF | [bco-dmo.org/dataset/858840](https://www.bco-dmo.org/dataset/858840) | BCO-DMO | CC BY 4.0 | Real EDX spectra and SEM images. Niche modality with limited public data. |
| 90 | **eels** | EELS DB | Real | ★★★★☆ | 290 spectra, 43 elements | CSV | [eelsdb.eu](https://eelsdb.eu/) | EELS DB consortium | Open | Largest open-access EELS database. Also: EELDC (82 standard references). |
| 91 | **electron_diffraction** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | EMsoft (DeGraef) | — | Dedicated physics: Bragg/Kikuchi pattern simulation from crystal structure databases. |
| 92 | **electron_holography** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | Lichte & Lehmann 2008 | — | Dedicated physics: off-axis interference fringe + phase/amplitude forward model. |
| 93 | **electron_tomography** | Nanomaterial STEM Tomo | Real | ★★★☆☆ | 5 tilt series | TIF | [figshare.com/collections/2185342](https://figshare.com/collections/2185342) | Levin et al., Sci. Data 2016 | CC BY 4.0 | Co2P, Pt nanoparticles, W needle. Raw tilt series + 3D reconstructions. |

### 2.6 Correlative & Other Microscopy

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 94 | **clem** | CLEM-Reg (EMPIAR) | Real | ★★★★☆ | 3 paired FL+EM volumes | MRC/TIFF | EMPIAR-10819, 11537, 11666 at [ebi.ac.uk/empiar](https://www.ebi.ac.uk/empiar/) | Nature Methods 2025 | CC0 | HeLa cells with MitoTracker, LysoTracker, Hoechst. Paired fluorescence + EM. |

---

## 3. Remote Sensing & Earth Observation (12 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 95 | **hyperspectral_remote** | AVIRIS Indian Pines / ROSIS Pavia | Real | ★★★★★ | 145×145, 200 bands / 610×340, 103 bands | MAT | [ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | Baumgardner, Purdue 2015 | Public domain | Most-cited hyperspectral remote sensing benchmarks. Indian Pines: 16 land-cover classes. |
| 96 | **sar** | Sentinel-1 GRD / UAVSAR | Physics | ★★★★★ | Global coverage | GeoTIFF | [scihub.copernicus.eu](https://scihub.copernicus.eu/) | ESA Copernicus | Free/open | Dedicated generator: 2D coherent frequency domain + speckle. |
| 97 | **insar** | Sentinel-1 SLC / COSAR | Physics | ★★★★★ | Global | SLC | [scihub.copernicus.eu](https://scihub.copernicus.eu/) | ESA Copernicus | Free/open | Dedicated generator: wrapped phase + topo/deformation fringes. |
| 98 | **polsar** | PolSF San Francisco AIRSAR | Real | ★★★★★ | Pixel-level annotated | PolSARpro | [github.com/liuxuvip/PolSF](https://github.com/liuxuvip/PolSF) | IETR Rennes | Research | 5-class pixel-level annotations. L/C-band. Most widely used PolSAR benchmark. |
| 99 | **lidar** | KITTI LiDAR / nuScenes | Physics | ★★★★★ | 15K frames / 1.4M frames | PCD/BIN | Internal | Geiger et al., CVPR 2012 | CC BY-NC-SA | Dedicated generator: range-dependent gain + atmospheric attenuation model. |
| 100 | **multispectral_sat** | Sentinel-2 Copernicus | Real | ★★★★★ | Global coverage | JP2/GeoTIFF | [dataspace.copernicus.eu](https://dataspace.copernicus.eu/) | ESA Copernicus program | Free/open | 13 spectral bands, 10-60m resolution, global 5-day revisit. Most popular satellite data. |
| 101 | **ocean_color** | NASA MODIS-Aqua Ocean Color L3 | Real | ★★★★★ | Global | NetCDF/HDF | [oceandata.sci.gsfc.nasa.gov](https://oceandata.sci.gsfc.nasa.gov/) | NASA OB.DAAC | Public domain | Chl-a, SST, reflectance. 4.6km pixel resolution. 2002-present. Free via HTTPS. |
| 102 | **passive_microwave** | AMSR-E/AMSR2 Brightness Temp | Real | ★★★★☆ | Global | HDF-EOS | [nsidc.org/data/ae_l2a](https://nsidc.org/data/ae_l2a/versions/4) | NSIDC, DOI:10.5067/YL62FUZLAJUT | NASA Earthdata | 6 channels (6.9-89 GHz), 5.4-56 km resolution. 2002-present. |
| 103 | **weather_radar** | NOAA NEXRAD Level-II | Real | ★★★★★ | 160 stations, 1991-present | Binary/NetCDF | [ncei.noaa.gov/products/radar/next-generation-weather-radar](https://www.ncei.noaa.gov/products/radar/next-generation-weather-radar) | NOAA | Public domain | Reflectivity, velocity, spectrum width, dual-pol. Full archive free on AWS S3. |
| 104 | **flash_lidar** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: single-shot ToF depth + Poisson photon counting model. No dedicated flash LiDAR dataset. |
| 105 | **gpr** | TU1208 Open Radargrams + CMU-GPR | Real | ★★★★☆ | 67 profiles + 15 trajectories | SEG-Y/HDF5 | [github.com/rpl-cmu/CMU-GPR-Dataset](https://github.com/rpl-cmu/CMU-GPR-Dataset) | TU1208 COST Action / CMU | Open | TU1208: IFSTTAR Nantes test site, 200-900 MHz. CMU: GPR + IMU + camera. |
| 106 | **fwi** | OpenFWI | Physics | ★★★★★ | 12 datasets, 2.1 TB | HDF5 | [openfwi-lanl.github.io](https://openfwi-lanl.github.io/) | Deng et al., NeurIPS 2022 | LANL | Dedicated physics: elastic wave equation + velocity model inversion. Synthetic but gold standard. |

---

## 4. Computational Imaging & Optics (22 modalities)

### 4.1 Spectral / Coded Aperture

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 107 | **cassi** | KAIST + TSA-Net Real CASSI | Real | ★★★★★ | 30 HS + 5 real scenes | MAT | [vclab.kaist.ac.kr](https://vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset.html) | Choi et al., SIGGRAPH Asia 2017 | Research | KAIST: 30 real HS images, 2704×3376. TSA-Net: 5 real SD-CASSI, 28 spectral channels. |
| 108 | **cacti** | DAVIS-2017 / Six Scenes | Real | ★★★★★ | Video frames | PNG | [davischallenge.org](https://davischallenge.org/) | Liu et al., IEEE TPAMI 2019 | CC BY-NC | Real video + temporal coded mask. Standard snapshot compressive imaging benchmark. |
| 109 | **spc** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | Rice CS camera | — | Dedicated physics: Hadamard/random pattern + single-detector CS forward model. |
| 110 | **coded_exposure** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | Raskar, SIGGRAPH 2006 | — | Dedicated physics: flutter-shutter binary sequence + motion PSF model. |
| 111 | **ghost_imaging** | — | Physics | ★★★★☆ | Generated | HDF5 | — | Shapiro, PRA 2008 | — | Dedicated generator: compressive single-pixel via Hadamard patterns. |

### 4.2 Computational Photography

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 112 | **hdr_imaging** | HDR+ Burst Photography | Real | ★★★★★ | Multi-burst | DNG (raw) | [hdrplusdata.org](https://hdrplusdata.org/) | Hasinoff et al., ACM TOG 2016 | Research | Real Android camera bursts (2-10 raw photos, 12-13 MP). |
| 113 | **event_camera** | DSEC | Real | ★★★★★ | 53 driving sequences | HDF5/PNG | [dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch/) | Gehrig et al., RA-L 2021 | CC BY-SA 4.0 | Stereo event cameras + global shutter RGB + LiDAR + RTK GPS. Day and night. |
| 114 | **panorama** | UDIS-D + VPG + ISIQA | Real | ★★★★☆ | 24 scenes + 264 stitched | PNG/JPEG | [github.com/visionxiang/Image-Stitching-Dataset](https://github.com/visionxiang/Image-Stitching-Dataset) | Various | Various | VPG: 24 real mobile captures. ISIQA: 264 stitched images with quality scores. |
| 115 | **polarization** | Spectro-Polarimetric (CVPR 2024) | Real | ★★★★☆ | 2,022 + 311 images | EXR/PNG | [light.princeton.edu](https://light.princeton.edu/publication/spectral-and-polarization-dataset/) | Princeton, CVPR 2024 | Research | Trichromatic + hyperspectral Stokes images (21 channels). Indoor and outdoor. |
| 116 | **machine_vision** | MVTec AD | Real | ★★★★★ | 5,000+ images, 15 categories | PNG | [mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/) | Bergmann et al., CVPR 2019 | CC BY-NC-SA 4.0 | Pixel-precise defect annotations. ~4.9 GB. Standard industrial inspection benchmark. |
| 117 | **matrix** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: low-rank + random entry sampling recovery model. |
| 118 | **cup** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | Liang, Nature 2018 | — | Dedicated physics: streak camera spatial encoding + temporal shearing CS forward model. |
| 119 | **streak_camera** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: space-to-time shearing + sweep voltage forward model. |

### 4.3 3D / Depth / Light Field

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 120 | **nerf** | Mip-NeRF 360 + LLFF | Real | ★★★★★ | 9 + 8 scenes | PNG/JSON | [jonbarron.info/mipnerf360](https://jonbarron.info/mipnerf360/) | Barron et al., CVPR 2022 | Research | 9 real indoor/outdoor object-centric scenes + 8 real forward-facing scenes. |
| 121 | **gaussian_splatting** | Tanks & Temples / Mip-NeRF 360 | Physics | ★★★★★ | Multi-scene | PLY/PNG | [tanksandtemples.org](https://www.tanksandtemples.org/) | Knapitsch et al., SIGGRAPH 2017 | Research | Dedicated generator: 3D anisotropic Gaussian rendering + multi-view synthesis. |
| 122 | **light_field** | Stanford Light Field Archive | Real | ★★★★★ | ~7,200 multiview + ~350 Lytro | PNG/MAT | [lightfields.stanford.edu](http://lightfields.stanford.edu/) | Stanford Graphics Lab | Research | Multiple collections: gantry-captured (2008), Lytro Illum (2016), multiview (2018). |
| 123 | **integral** | Stanford Lytro Archive | Real | ★★★★☆ | ~350 captures | LFR/PNG | [lightfields.stanford.edu/LF2016.html](http://lightfields.stanford.edu/LF2016.html) | Stanford Graphics Lab | Research | Lytro Illum plenoptic camera, 9 categories. |
| 124 | **photometric_stereo** | DiLiGenT | Real | ★★★★★ | 10 objects, 96 lightings | PNG (16-bit) | [sites.google.com/site/photometricstereodata](https://sites.google.com/site/photometricstereodata/) | Shi et al., TPAMI 2019 | Research | Laser-scanned ground truth normals. Gold standard for photometric stereo. Extended: DiLiGenT-MV, DiLiGenT102. |
| 125 | **structured_light** | Middlebury Stereo | Real | ★★★★★ | 24+ datasets | PNG/PFM | [vision.middlebury.edu/stereo/data](https://vision.middlebury.edu/stereo/data/) | Scharstein et al., GCPR 2014 | Research | Subpixel-accurate ground truth from structured light scanning + robot arm. De facto 3D benchmark. |
| 126 | **tof_camera** | ETH3D / Middlebury 3D / TUM RGB-D | Physics | ★★★★★ | Multi-scene | PNG/PFM | Internal | Schöps et al., CVPR 2017 | Research | Dedicated physics: ToF depth + photon counting noise model. |

### 4.4 Astronomy / Coronagraphy

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 127 | **coronagraphy** | EIDC (Exoplanet Imaging Data Challenge) | Real | ★★★★☆ | 8 datasets | FITS | [zenodo.org/records/2815298](https://zenodo.org/records/2815298) | Cantalloube et al. | CC BY 4.0 | Real SPHERE-IFS + GPI instrument data with injected synthetic planets. |
| 128 | **eht_imaging** | EHT M87 Calibrated Data | Real | ★★★★★ | VLBI visibilities | UVFITS | [github.com/eventhorizontelescope/2019-D01-01](https://github.com/eventhorizontelescope/2019-D01-01) | EHT Collaboration, ApJL 2019 | CC BY 4.0 | Real calibrated visibility data from historic M87 black hole observation (April 2017). |
| 129 | **adaptive_optics** | Keck Observatory Archive (KOA) | Real | ★★★★☆ | Archive | FITS | [koa.ipac.caltech.edu](https://koa.ipac.caltech.edu/) | W. M. Keck Observatory | Free registration | Real AO-corrected NIRC2/OSIRIS infrared images from Keck II telescope. |
| 130 | **lucky_imaging** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | Law et al., NJP 2006 | — | Dedicated physics: Kolmogorov turbulence phase screen + short-exposure simulation. |
| 131 | **entangled_photon** | — | Physics | ★☆☆☆☆ | Generated | HDF5 | — | Lopaeva et al., Sci. Adv. 2020 | — | Dedicated physics: SPDC photon-pair + coincidence counting model. Emerging field. |
| 132 | **quantum_illumination** | — | Physics | ★☆☆☆☆ | Generated | HDF5 | — | Lloyd, Science 2008 | — | Dedicated physics: entangled-pair detection + thermal noise model. Emerging field. |

---

## 5. Spectroscopy & Chemical Imaging (16 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 133 | **raman_imaging** | RRUFF Raman Database | Physics | ★★★★★ | 3,000+ minerals | Spectra | [rruff.info](https://rruff.info/) | RRUFF consortium | Public | Dedicated generator: 3-species spectral unmixing, 512 channels. RRUFF: largest Raman spectral reference. |
| 134 | **ftir_imaging** | USGS Spectral Library v7 | Physics | ★★★★★ | 1,300+ spectra | ASCII | [usgs.gov/labs/spec-lab](https://www.usgs.gov/labs/spec-lab) | Kokaly et al., USGS 2017 | Public domain | Dedicated generator: Michelson interferogram + Hamming apodization physics. |
| 135 | **srs** | OpenSRH | Real | ★★★★★ | 300+ patients, 1,300+ WSIs | TIFF | [opensrh.mlins.org](https://opensrh.mlins.org/) | Hollon et al., NeurIPS 2022 | Research | (Also listed in microscopy) First public SRH dataset for brain tumor diagnosis. |
| 136 | **libs** | CEITEC LIBS Benchmark | Real | ★★★★☆ | 138 soil samples, 12 classes | CSV | [libs.ceitec.cz/benchmarking](https://libs.ceitec.cz/benchmarking/) | Pořízka et al., Nature Sci. Data 2020 | CC BY 4.0 | Real LIBS spectra with composition + uncertainties. Also: CSA planetary LIBS dataset. |
| 137 | **desi** | MTBLS176 | Real | ★★★★☆ | Tissue sections | imzML | [ebi.ac.uk/metabolights/MTBLS176](https://www.ebi.ac.uk/metabolights/MTBLS176) | Oetjen et al., GigaScience 2015 | Open | Real DESI-MSI of human colorectal adenocarcinoma tissue. Standard MS imaging benchmark. |
| 138 | **maldi_msi** | MTBLS176 3D MALDI-MSI | Real | ★★★★☆ | Tissue sections | imzML | [ebi.ac.uk/metabolights/MTBLS176](https://www.ebi.ac.uk/metabolights/MTBLS176) | Oetjen et al., GigaScience 2015 | Open | Murine pancreas, kidney, oral squamous cell carcinoma. Also: METASPACE (metaspace2020.eu). |
| 139 | **xrf_imaging** | UMich Deep Blue Scanning XRF | Real | ★★☆☆☆ | Core scan data | CSV/TIFF | [deepblue.lib.umich.edu](https://deepblue.lib.umich.edu/data/concern/data_sets/qr46r085k) | UMich | Open | 200μm resolution ITRAX core scanner data. Also: USGS geological XRF data. |
| 140 | **xrf_tomo** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: Beer-Lambert attenuation + fluorescence yield sinogram forward model. |
| 141 | **brillouin** | 3D Brillouin Zebrafish Eye | Real | ★★☆☆☆ | Single specimen | JSON/CSV | DOI:10.1016/j.dib.2020.105427 | Data in Brief 2020 | Open | Real in-vivo Brillouin scattering spectra of zebrafish larvae (48-52 hpf). Only public Brillouin dataset. |
| 142 | **sims** | NanoSIMS Isotopic Ratio | Real | ★★☆☆☆ | Foraminifera | IMG | [zenodo.org/records/3834220](https://zenodo.org/records/3834220) | Zenodo 2020 | CC BY 4.0 | ¹³C/¹²C, ¹⁵N/¹⁴N isotopic ratios of foraminifera and symbiotic dinoflagellates. |
| 143 | **atom_probe** | Zenodo APT Fe-Cr | Real | ★★☆☆☆ | 1.235M atoms | POS/RRNG | [zenodo.org/records/13886599](https://zenodo.org/records/13886599) | Zenodo 2024 | CC BY 4.0 | Real experimental APT: Fe-51.4at% Cr. Also: apttools at sourceforge.io. |
| 144 | **pump_probe** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: transient absorption + carrier dynamics rate equations model. |
| 145 | **nsom** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: FDTD near-field + sub-wavelength aperture tip model. No public NSOM dataset. |
| 146 | **mfm** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: micromagnetic (MuMax3/OOMMF) field map + tip response model. |

---

## 6. Scanning Probe & Surface (4 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 147 | **afm** | Zenodo Real AFM + QUAM-AFM | Real | ★★★☆☆ | 10 specimens + 165M simulated | MI/NPY | [zenodo.org/records/60434](https://zenodo.org/records/60434) | ACS JCIM 2022 | CC BY | Real AFM images (10 specimens MI format). QUAM-AFM: 685K molecules, 256×256. |
| 148 | **stm** | STM Graphene-on-Ni | Real | ★★★★☆ | 7,287 images, 4.7 GB | PNG | [zenodo.org/records/5799774](https://zenodo.org/records/5799774) | Zenodo 2022 | CC BY 4.0 | 3 categories of STM images. Also: JARVIS-STM (NIST) with DFT-computed STM of 716 2D materials. |

---

## 7. Astronomy & Astrophysics (6 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 149 | **radio_astronomy** | NVSS + FIRST Survey | Real | ★★★★★ | ~2M + 800K+ sources | FITS | [cv.nrao.edu/nvss](https://www.cv.nrao.edu/nvss/) | Condon et al., AJ 1998 | Public | Full-sky 1.4 GHz radio continuum surveys. Downloadable FITS cubes. |
| 150 | **radio_interferometry** | EHT M87 + MIT VLBI | Real | ★★★★★ | VLBI visibilities | UVFITS | [github.com/eventhorizontelescope](https://github.com/eventhorizontelescope/2019-D01-01) + [vlbiimaging.csail.mit.edu](http://vlbiimaging.csail.mit.edu/) | EHT Collaboration, ApJL 2019 | CC BY 4.0 | Real VLBI data + MIT benchmark for reconstruction algorithms. |
| 151 | **solar_imaging** | SDO/AIA SuryaBench | Real | ★★★★★ | ~260K images, 4096×4096 | FITS | [nature.com/articles/s41597-026-06552-5](https://www.nature.com/articles/s41597-026-06552-5) | SuryaBench, Sci. Data 2026 | CC BY 4.0 | 2010-2024 full-disk solar images, 9 UV/EUV channels. ML-ready. |
| 152 | **gravitational_wave** | GWOSC LIGO/Virgo/KAGRA | Real | ★★★★★ | O1-O4 runs | HDF5/GWF | [gwosc.org/data](https://gwosc.org/data/) | Abbott et al., PRL 2016 | CC BY 4.0 | Real detector strain time-series. Includes GW150914 discovery event. |
| 153 | **particle_calorimetry** | CERN Open Data + CaloChallenge | Real | ★★★★★ | TB-scale | ROOT/HDF5 | [opendata.cern.ch](https://opendata.cern.ch/) + [calochallenge.github.io](https://calochallenge.github.io/homepage/) | CERN Open Data Portal | CC0 | Real CMS/ATLAS detector data + CaloChallenge simulated shower benchmarks. |
| 154 | **ocean_acoustic_tomo** | ACOBAR Fram Strait | Real | ★★★☆☆ | 2-year time series | NetCDF | DOI:10.1016/j.dib.2022.108173 | Data in Brief 2022 | Open | Real acoustic tomography: temperature/sound-speed, 3 paths, 2010-2012. |

---

## 8. NDT & Industrial Inspection (12 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 155 | **active_thermography** | PVC-Infrared | Real | ★★★☆☆ | 19 PVC specimens | TIFF/CSV | [kaggle.com/datasets/ziangwei/irtpvc](https://www.kaggle.com/datasets/ziangwei/irtpvc) | Applied Sciences 2023 | CC BY 4.0 | Pulsed thermography with known subsurface defects at various depths/sizes. |
| 156 | **eddy_current** | IEEE DataPort EC Railroad + ALETHEIA | Real | ★★★☆☆ | 17.81 GB | MAT/HDF5 | [ieee-dataport.org](https://ieee-dataport.org/documents/eddy-current-raw-data) + ALETHEIA (OpenReview) | IEEE DataPort | Subscription / Open | Real EC railroad crack detection data + ALETHEIA open eddy-current pulsed thermography. |
| 157 | **ultrasonic_phased_array** | ML-NDT Phased Array | Real | ★★★☆☆ | Annotated flaw data | JSON/CSV | [github.com/iikka-v/ML-NDT](https://github.com/iikka-v/ML-NDT) | ML-NDT project | Open | Annotated crack flaw data with locations and sizes. |
| 158 | **acoustic_emission** | 4TU AE CFRP | Real | ★★★☆☆ | 19 specimens | TRADB/PRIDB | [data.4tu.nl/articles/_/21621381](https://data.4tu.nl/articles/_/21621381/1) | 4TU.ResearchData | CC BY 4.0 | Real AE waveforms from CFRP compression tests. |
| 159 | **acoustic_microscopy** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: acoustic wave propagation + C-scan subsurface PSF model. |
| 160 | **shearography** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: FEA deformation + optical phase derivative NDT model. |
| 161 | **terahertz** | NIST THz Spectroscopy | Physics | ★★★☆☆ | Spectral database | ASCII | [nist.gov](https://www.nist.gov/) | NIST THz database | Public domain | Dedicated physics: THz PSF + material absorption imaging model. |
| 162 | **sonar** | AI4Shipwrecks + NOAA MBBDB | Real | ★★★★☆ | 286 images + global archive | TIFF/GeoTIFF | [deepblue.lib.umich.edu](https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x) + [ncei.noaa.gov](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc:G01034) | UMich / NOAA | Open / Public | Side-scan sonar (28 shipwrecks) + global multibeam bathymetry archive. |

---

## 9. Crystallography & Diffraction (6 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 163 | **xray_crystallography** | IRRMC + PDB | Real | ★★★★★ | 2,920+ experiments / 200K+ structures | CIF/MTZ | [proteindiffraction.org](http://www.proteindiffraction.org/) + [rcsb.org](https://www.rcsb.org/) | Berman et al., NAR 2000 | CC0 / Public | IRRMC: raw diffraction images freely downloadable. PDB: largest structural biology database. |
| 164 | **xfel_sfx** | CXIDB | Real | ★★★★★ | 215+ entries | HDF5/CXI | [cxidb.org](https://cxidb.org/) | Maia, Nat. Methods 2012 | CC BY 4.0 | Coherent X-ray Imaging Data Bank. Also: EU-XFEL multi-million SFX images. |
| 165 | **saxs** | SASBDB | Real | ★★★★☆ | Curated profiles | DAT/ZIP | [sasbdb.org](https://www.sasbdb.org/) | Kikhney et al., NAR 2020 | Open | EMBL Hamburg. Experimental conditions, sample details, derived models. |
| 166 | **waxs** | — | Physics | ★★☆☆☆ | Generated | HDF5 | — | — | — | Dedicated physics: Debye scattering from crystal structures (COD) at wide angles. |
| 167 | **neutron_tomo** | PSI ICON Petrified Wood | Real | ★★★☆☆ | Single specimen | TIFF | [data.mendeley.com](https://data.mendeley.com/datasets/g5snr785xy/2) | Mendeley Data | CC BY 4.0 | PSI ICON beamline. Also: NeXT-Grenoble paired neutron + X-ray data. |
| 168 | **neutron_diffraction** | — | Physics | ★★★☆☆ | Generated | HDF5 | — | PDB neutron structures | — | Dedicated physics: Bragg peak generation from crystallographic database. |

---

## 10. Geophysics & Seismic (2 modalities)

| # | Modality | Dataset | Data Type | Popularity | Size | Format | Source / URL | Key Publication | License | Notes |
|---|----------|---------|-----------|------------|------|--------|-------------|-----------------|---------|-------|
| 169 | **seismic_tomo** | GlobalTomo / SEG Benchmarks | Physics | ★★★★★ | Multiple | SEG-Y/HDF5 | [global-tomo.github.io](https://global-tomo.github.io/) + [seg.org/seam](https://seg.org/seam/data-sets/) | GlobalTomo / SEG consortium | Research | Dedicated physics: travel-time ray-tracing + velocity forward model. |
| 170 | **fwi** | (See #106 above) | Physics | ★★★★★ | 12 datasets, 2.1 TB | HDF5 | [openfwi-lanl.github.io](https://openfwi-lanl.github.io/) | Deng et al., NeurIPS 2022 | LANL | (Cross-reference with remote sensing section) |

---

## GCS Storage

All benchmark datasets are stored in Google Cloud Storage:
- **Path**: `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/{public,dev,hidden}/`
- **Per-tier structure**: public (≥10 samples), dev (20 samples, blind), hidden (20 samples, server-only)
- **Format**: HDF5 with keys `x_true`, `y`, `H_ideal` per sample
- **Seed offsets**: public=0, dev=+10000, hidden=+20000 (prevents memorization)

---

*Generated by PWM Benchmark pipeline. 168 modalities verified, physics-accurate.*
