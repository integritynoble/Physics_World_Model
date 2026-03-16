# PWM Benchmark -- Standard Dataset State

Last updated: 2026-03-16 -- 170 modalities

## Cloud Storage (GCS)

- **GCS bucket:** `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/`
- **Download:** `python scripts/download_standard_from_gcs.py`
- **Upload:** `python scripts/upload_standard_to_gcs.py`

## Status Summary

| Status | Count | Description |
|--------|-------|-------------|
| done | 85 | Uses canonical/real data from this modality |
| needs_canonical | 24 | Canonical dataset exists, need to download 30 samples |
| simulation | 61 | No public benchmark -- using simulated data |
| **Total** | **170** | |

## All Modalities

| # | Modality | N | Canonical Dataset | Link | Status |
|---|----------|---|-------------------|------|--------|
| 1 | acoustic_emission | 20 | Acoustic emission NDT | -- | simulation |
| 2 | acoustic_microscopy | 20 | Scanning acoustic micro | -- | simulation |
| 3 | active_thermography | 20 | Active thermography NDT | -- | simulation |
| 4 | adaptive_optics | 20 | ESO VLT AO observation | [link](https://archive.eso.org/) | done |
| 5 | afm | 20 | AFM specimens (Zenodo 60434) | [link](https://zenodo.org/records/60434) | done |
| 6 | angiography | 20 | ARCADE coronary XCA (Zenodo 10390295) | [link](https://zenodo.org/records/10390295) | done |
| 7 | asl_mri | 20 | OpenNeuro ds000114 brain MRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 8 | atom_probe | 20 | Atom probe tomography | -- | simulation |
| 9 | bioluminescence_tomo | 20 | Bioluminescence tomography | -- | simulation |
| 10 | brachytherapy_img | 20 | Brachytherapy imaging | -- | simulation |
| 11 | brillouin | 20 | Brillouin microscopy | -- | simulation |
| 12 | cacti | 20 | DeSCI CACTI benchmark (6 grayscale videos) | [link](https://github.com/liuyang12/DeSCI) | done |
| 13 | cars | 20 | CARS microscopy | -- | simulation |
| 14 | cassi | 10 | CAVE 31-band hyperspectral | [link](https://www.cs.columbia.edu/CAVE/databases/multispectral/) | done |
| 15 | cathodoluminescence | 20 | CL zircon (Zenodo 6801483) | [link](https://zenodo.org/records/6801483) | done |
| 16 | cbct | 20 | Walnut CBCT (42 walnuts) | [link](https://zenodo.org/records/2686726) | needs_canonical |
| 17 | cest_mri | 20 | CEST MRI | -- | simulation |
| 18 | ceus | 20 | BUS-BRA contrast US (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 19 | clem | 20 | Correlative LEM | -- | simulation |
| 20 | coded_exposure | 20 | BSD68 (standard CI test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 21 | confocal_3d | 20 | BioSR (2200+ SR pairs) | [link](https://figshare.com/articles/dataset/BioSR/13264793) | needs_canonical |
| 22 | confocal_endomicroscopy | 30 | Kvasir-SEG GI tract (30 images) | [link](https://datasets.simula.no/kvasir-seg/) | done |
| 23 | confocal_livecell | 20 | LIVECell (5239 images) | [link](https://sartorius-research.github.io/LIVECell/) | needs_canonical |
| 24 | coronagraphy | 20 | ESO VLT coronagraph | [link](https://archive.eso.org/) | done |
| 25 | cryo_em | 20 | EMPIAR (micrograph archive) | [link](https://www.ebi.ac.uk/empiar/) | needs_canonical |
| 26 | cryo_et | 20 | SHREC 2021 cryo-ET | [link](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA) | needs_canonical |
| 27 | ct | 20 | LoDoPaB-CT (42K CT pairs) | [link](https://zenodo.org/records/3384092) | needs_canonical |
| 28 | ct_fluorescence | 20 | CT fluorescence | -- | simulation |
| 29 | cup | 20 | BSD68 (CUP test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 30 | dark_field | 20 | Dark-field X-ray | -- | simulation |
| 31 | desi | 20 | DESI MSI | -- | simulation |
| 32 | dexa | 20 | NHANES DXA | [link](https://www.cdc.gov/nchs/nhanes/) | needs_canonical |
| 33 | dic | 20 | BBBC003 mouse embryo DIC | [link](https://bbbc.broadinstitute.org/BBBC003) | done |
| 34 | diffusion_mri | 20 | HCP diffusion MRI | [link](https://www.humanconnectome.org/) | needs_canonical |
| 35 | digital_breast_tomo | 20 | Digital breast tomosynthesis | -- | simulation |
| 36 | dna_paint | 20 | DNA-PAINT | -- | simulation |
| 37 | doppler_ultrasound | 20 | BUS-BRA breast US (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 38 | dot | 20 | Diffuse optical tomography | -- | simulation |
| 39 | ebsd | 20 | EBSD deformed iron (Zenodo 1214829) | [link](https://zenodo.org/records/1214829) | done |
| 40 | eddy_current | 20 | Eddy current NDT | -- | simulation |
| 41 | edx_mapping | 20 | EDX elemental maps (Zenodo 14960843) | [link](https://zenodo.org/records/14960843) | done |
| 42 | eels | 20 | EMDB EELS spectral maps | [link](https://www.ebi.ac.uk/emdb/) | done |
| 43 | eht_imaging | 20 | EHT M87 black hole (2019 SR1) | [link](https://eventhorizontelescope.org/for-astronomers/data) | done |
| 44 | elastography | 20 | BUS-BRA elastography (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 45 | electron_diffraction | 20 | EMDB electron diffraction | [link](https://www.ebi.ac.uk/emdb/) | done |
| 46 | electron_holography | 20 | Electron holography (Zenodo 18289938) | [link](https://zenodo.org/records/18289938) | done |
| 47 | electron_tomography | 20 | EMDB electron tomography | [link](https://www.ebi.ac.uk/emdb/) | done |
| 48 | endoscopy | 20 | Kvasir-SEG (1000 colonoscopy polyp images) | [link](https://datasets.simula.no/kvasir-seg/) | done |
| 49 | entangled_photon | 20 | BSD68 (quantum imaging test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 50 | event_camera | 20 | EDHT21 DVS events (Zenodo 4918320) | [link](https://zenodo.org/records/4918320) | done |
| 51 | expansion | 20 | Expansion microscopy | -- | simulation |
| 52 | fib_sem | 20 | FIB-SEM golgi (Zenodo 8114392) | [link](https://zenodo.org/records/8114392) | done |
| 53 | flash_lidar | 20 | Middlebury Stereo depth | [link](https://vision.middlebury.edu/stereo/) | done |
| 54 | flim | 20 | FLIM | -- | simulation |
| 55 | fluoroscopy | 20 | WEISS Catheter Fluoroscopy | [link](https://rdr.ucl.ac.uk/articles/dataset/24624243) | needs_canonical |
| 56 | fmri | 20 | OpenNeuro ds000114 BOLD fMRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 57 | fpm | 20 | Fourier ptychographic micro | -- | simulation |
| 58 | ftir_imaging | 20 | FTIR breast tissue (Zenodo 4986399) | [link](https://zenodo.org/records/4986399) | done |
| 59 | fundus | 30 | RFMiD retinal fundus (Zenodo 7505822) | [link](https://zenodo.org/records/7505822) | done |
| 60 | fwi | 20 | Marmousi2 / OpenFWI | [link](https://openfwi-lanl.github.io/) | done |
| 61 | gaussian_splatting | 11 | Tiny NeRF Lego + 3DGS | [link](https://github.com/graphdeco-inria/gaussian-splatting) | done |
| 62 | ghost_imaging | 20 | BSD68 (ghost imaging test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 63 | gpr | 20 | CMU-GPR / TU1208 radargrams | [link](https://github.com/rpl-cmu/CMU-GPR-Dataset) | needs_canonical |
| 64 | gravitational_wave | 20 | GWOSC GW150914 strain | [link](https://gwosc.org/) | done |
| 65 | hdr_imaging | 20 | BSD68 (HDR test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 66 | holography | 20 | Electron holography (Zenodo 18289938) | [link](https://zenodo.org/records/18289938) | done |
| 67 | hyperspectral_remote | 30 | Indian Pines AVIRIS (30 patches) | [link](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | done |
| 68 | impedance_tomo | 20 | KTC 2023 EIT phantom (Zenodo 10986692) | [link](https://zenodo.org/records/10986692) | done |
| 69 | industrial_ct | 20 | Walnut CBCT micro-CT | [link](https://zenodo.org/records/2686726) | needs_canonical |
| 70 | insar | 20 | InSAR | -- | simulation |
| 71 | integral | 20 | BSD68 (integral imaging test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 72 | ism | 20 | Image scanning microscopy | -- | simulation |
| 73 | ivus | 20 | BUS-BRA intravascular (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 74 | lattice_lightsheet | 20 | Lattice light-sheet | -- | simulation |
| 75 | lensless | 20 | DiffuserCam | [link](https://waller-lab.github.io/DiffuserCam/) | needs_canonical |
| 76 | libs | 20 | LIBS spectroscopy | -- | simulation |
| 77 | lidar | 20 | SemanticKITTI (43K scans) | [link](https://semantic-kitti.org/) | needs_canonical |
| 78 | light_field | 20 | Stanford Lytro light field | [link](http://lightfields.stanford.edu/LF2016.html) | done |
| 79 | lightsheet | 20 | CARE Tribolium | [link](https://publications.mpi-cbg.de/publications-sites/7207/) | needs_canonical |
| 80 | lucky_imaging | 20 | ESO lucky imaging | [link](https://archive.eso.org/) | done |
| 81 | machine_vision | 20 | FaultSeg wheel defects (Zenodo 13162335) | [link](https://zenodo.org/records/13162335) | done |
| 82 | magnetic_particle | 20 | Magnetic particle imaging | -- | simulation |
| 83 | maldi_msi | 20 | MALDI MSI | -- | simulation |
| 84 | mammography | 20 | Benign Breast Tumor (Zenodo 5084116) | [link](https://zenodo.org/records/5084116) | done |
| 85 | matrix | 20 | EMDB density maps | [link](https://www.ebi.ac.uk/emdb/) | done |
| 86 | mfm | 20 | Magnetic force microscopy | -- | simulation |
| 87 | minflux | 20 | MINFLUX | -- | simulation |
| 88 | mr_elastography | 20 | MR elastography | -- | simulation |
| 89 | mr_fingerprinting | 20 | MR fingerprinting | -- | simulation |
| 90 | mra | 20 | OpenNeuro ds000114 brain MRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 91 | mri | 20 | fastMRI (8K+ volumes, NYU/Meta) | [link](https://fastmri.med.nyu.edu/) | needs_canonical |
| 92 | mrs | 20 | MR spectroscopy | -- | simulation |
| 93 | multispectral_sat | 20 | EuroSAT Sentinel-2 (27K images) | [link](https://zenodo.org/records/7711810) | done |
| 94 | muon_tomo | 20 | Muon tomography | -- | simulation |
| 95 | nerf | 11 | Tiny NeRF Lego scene | [link](https://github.com/bmild/nerf) | done |
| 96 | neutron_diffraction | 20 | Neutron diffraction | -- | simulation |
| 97 | neutron_tomo | 20 | Neutron tomography | -- | simulation |
| 98 | nirs_brain | 20 | fNIRS brain imaging | -- | simulation |
| 99 | nsom | 20 | Near-field scanning optical | -- | simulation |
| 100 | ocean_acoustic_tomo | 20 | ACOBAR Fram Strait OAT | [link](https://doi.org/10.1016/j.dib.2022.108160) | needs_canonical |
| 101 | ocean_color | 20 | EuroSAT SeaLake Sentinel-2 | [link](https://zenodo.org/records/7711810) | done |
| 102 | oct | 20 | OCTDL retinal OCT (2064 images) | [link](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) | done |
| 103 | octa | 20 | OCTA-Mosaicking (Zenodo 14333858) | [link](https://zenodo.org/records/14333858) | done |
| 104 | odt | 20 | Optical diffraction tomography | -- | simulation |
| 105 | palm_storm | 20 | STORM tubulin (Zenodo 7620025) | [link](https://zenodo.org/records/7620025) | done |
| 106 | panorama | 20 | BSD68 (panorama test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 107 | particle_calorimetry | 20 | Particle calorimetry | -- | simulation |
| 108 | passive_microwave | 20 | Passive microwave | -- | simulation |
| 109 | pet | 20 | UDPET ultra-low dose PET | [link](https://zenodo.org/records/6361846) | needs_canonical |
| 110 | pet_ct | 20 | autoPET (TCIA, 1014 studies) | [link](https://autopet.grand-challenge.org/) | needs_canonical |
| 111 | pet_mr | 20 | PET-MR | -- | simulation |
| 112 | phase_contrast | 30 | ISBI Cell Tracking PhC-C2DH-U373 | [link](https://celltrackingchallenge.net/) | done |
| 113 | phase_retrieval | 20 | Holographic phase (Zenodo 13771363) | [link](https://zenodo.org/records/13771363) | done |
| 114 | photoacoustic | 20 | Duke PAM mouse brain (Zenodo 4042171) | [link](https://zenodo.org/records/4042171) | done |
| 115 | photometric_stereo | 20 | BSD68 (photometric stereo test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 116 | polarization | 20 | Polarimetric camera (Zenodo 4483248) | [link](https://zenodo.org/records/4483248) | done |
| 117 | polsar | 20 | Polarimetric SAR | -- | simulation |
| 118 | portal_imaging | 20 | Portal imaging | -- | simulation |
| 119 | proton_radiography | 20 | Proton radiography | -- | simulation |
| 120 | proton_therapy_img | 20 | Proton therapy imaging | -- | simulation |
| 121 | ptychography | 20 | Ptychography exp (Zenodo 16263064) | [link](https://zenodo.org/records/16263064) | done |
| 122 | pump_probe | 20 | Pump-probe microscopy | -- | simulation |
| 123 | quantum_illumination | 20 | BSD68 (QI test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 124 | radio_astronomy | 30 | FIRST VLA 1.4GHz (30 cutouts) | [link](https://www.cv.nrao.edu/first/) | done |
| 125 | radio_interferometry | 30 | NVSS 1.4GHz Survey (30 cutouts) | [link](https://www.cv.nrao.edu/nvss/) | done |
| 126 | raman_imaging | 20 | Raman photothermal (Zenodo 8141012) | [link](https://zenodo.org/records/8141012) | done |
| 127 | sar | 20 | MSTAR (SAR target recognition) | [link](https://www.sdms.afrl.af.mil/index.php?collection=mstar) | needs_canonical |
| 128 | saxs | 20 | Small-angle X-ray scattering | -- | simulation |
| 129 | sd_cassi | 10 | CAVE 24-band hyperspectral | [link](https://www.cs.columbia.edu/CAVE/databases/multispectral/) | done |
| 130 | seismic_tomo | 20 | Marmousi2 elastic velocity | [link](https://wiki.seg.org/wiki/Open_data) | done |
| 131 | sem | 20 | SEM nanoparticle (Zenodo 7986673) | [link](https://zenodo.org/records/7986673) | done |
| 132 | shearography | 20 | Shearography NDT | -- | simulation |
| 133 | shg | 20 | PSHG-TISS | [link](https://doi.org/10.17605/OSF.IO/K2Z8G) | needs_canonical |
| 134 | sim | 20 | UniFMIR SIM F-actin (Zenodo 8420100) | [link](https://zenodo.org/records/8420100) | done |
| 135 | sims | 20 | SIMS | -- | simulation |
| 136 | solar_imaging | 20 | NASA SDO AIA EUV composite | [link](https://sdo.gsfc.nasa.gov/) | done |
| 137 | sonar | 30 | UATD forward-looking sonar (30 images) | [link](https://figshare.com/articles/dataset/UATD_Dataset/21331143) | done |
| 138 | spc | 20 | BSD68 (SPC test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 139 | spc_kronecker | 10 | Indian Pines AVIRIS | [link](https://engineering.purdue.edu/~biehl/MultiSpec/) | done |
| 140 | spect | 20 | SPECT | -- | simulation |
| 141 | spect_ct | 20 | SPECT-CT | -- | simulation |
| 142 | spectral_ct | 20 | Spectral CT | -- | simulation |
| 143 | spinning_disk | 20 | Spinning disk confocal | -- | simulation |
| 144 | srs | 20 | SRS microscopy | -- | simulation |
| 145 | sted | 20 | UniFMIR STED (Zenodo 8420100) | [link](https://zenodo.org/records/8420100) | done |
| 146 | stem | 20 | Scanning TEM | -- | simulation |
| 147 | stm | 20 | Graphene/Ni STM (Zenodo 5799774) | [link](https://zenodo.org/records/5799774) | needs_canonical |
| 148 | streak_camera | 20 | BSD68 (streak camera test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 149 | structured_light | 20 | Middlebury Stereo depth | [link](https://vision.middlebury.edu/stereo/) | done |
| 150 | swi | 20 | OpenNeuro ds000114 brain MRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 151 | talbot_lau | 20 | Talbot-Lau | -- | simulation |
| 152 | tem | 20 | TEM cilia (Zenodo 11188503) | [link](https://zenodo.org/records/11188503) | done |
| 153 | terahertz | 20 | Active THz dataset | [link](https://github.com/LingLIx/THz_Dataset) | needs_canonical |
| 154 | three_photon | 20 | Three-photon microscopy | -- | simulation |
| 155 | tirf | 20 | TIRF microscopy | -- | simulation |
| 156 | tof_camera | 20 | ToF depth maps (Zenodo 10732158) | [link](https://zenodo.org/records/10732158) | done |
| 157 | two_photon | 20 | CaImAn calcium imaging | [link](https://github.com/flatironinstitute/CaImAn) | done |
| 158 | ultrasonic_phased_array | 20 | UT phased array NDT | -- | simulation |
| 159 | ultrasound | 20 | BUSI breast ultrasound (780 images) | [link](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) | done |
| 160 | us_mri | 20 | Ultrashort-TE MRI | -- | simulation |
| 161 | waxs | 20 | Wide-angle X-ray scattering | -- | simulation |
| 162 | weather_radar | 20 | NOAA NEXRAD composite | [link](https://www.ncei.noaa.gov/products/radar) | done |
| 163 | widefield | 20 | FMD widefield denoising | [link](https://github.com/yinhaoz/denoising-fluorescence) | needs_canonical |
| 164 | widefield_lowdose | 20 | FMD low-dose | [link](https://github.com/yinhaoz/denoising-fluorescence) | needs_canonical |
| 165 | xfel_sfx | 20 | EMDB XFEL SFX | [link](https://www.ebi.ac.uk/emdb/) | done |
| 166 | xray_crystallography | 20 | EMDB/PDB X-ray crystallography | [link](https://www.ebi.ac.uk/emdb/) | done |
| 167 | xray_ndt | 30 | X-ray radiography (Zenodo 7947924) | [link](https://zenodo.org/records/7947924) | done |
| 168 | xray_radiography | 20 | NIH ChestX-ray14 | [link](https://nihcc.app.box.com/v/ChestXray-NIHCC) | done |
| 169 | xrf_imaging | 20 | XRF fossil map (Zenodo 4005031) | [link](https://zenodo.org/records/4005031) | done |
| 170 | xrf_tomo | 20 | XRF tomography | -- | simulation |
