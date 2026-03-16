# PWM Benchmark -- Standard Dataset State

Last updated: 2026-03-16 -- 170 modalities, 20 samples each

## Cloud Storage (GCS)

- **GCS bucket:** `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/`
- **Download:** `python scripts/download_standard_from_gcs.py`
- **Upload:** `python scripts/upload_standard_to_gcs.py`

## Status Summary

| Status | Count | Description |
|--------|-------|-------------|
| done | 75 | Uses canonical/real data from this modality |
| needs_upgrade | 34 | Canonical dataset exists, currently using proxy |
| niche | 61 | No public benchmark dataset exists |
| **Total** | **170** | |

## All Modalities

| # | Modality | N | Canonical Dataset | Link | Status |
|---|----------|---|-------------------|------|--------|
| 1 | acoustic_emission | 20 | AE (niche) | -- | niche |
| 2 | acoustic_microscopy | 20 | SAM (niche) | -- | niche |
| 3 | active_thermography | 20 | Active thermo (niche) | -- | niche |
| 4 | adaptive_optics | 20 | ESO VLT AO observation | [link](https://archive.eso.org/) | done |
| 5 | afm | 20 | AFM specimens (Zenodo 60434) | [link](https://zenodo.org/records/60434) | done |
| 6 | angiography | 20 | ARCADE coronary XCA (Zenodo 10390295) | [link](https://zenodo.org/records/10390295) | done |
| 7 | asl_mri | 20 | OpenNeuro ds000114 brain MRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 8 | atom_probe | 20 | Atom probe (niche) | -- | niche |
| 9 | bioluminescence_tomo | 20 | BLT (niche) | -- | niche |
| 10 | brachytherapy_img | 20 | Brachytherapy (niche) | -- | niche |
| 11 | brillouin | 20 | Brillouin (niche) | -- | niche |
| 12 | cacti | 20 | DeSCI CACTI benchmark (6 grayscale videos) | [link](https://github.com/liuyang12/DeSCI) | done |
| 13 | cars | 20 | CARS (niche) | -- | niche |
| 14 | cassi | 10 | CAVE 31-band hyperspectral | [link](https://www.cs.columbia.edu/CAVE/databases/multispectral/) | done |
| 15 | cathodoluminescence | 20 | CL zircon (Zenodo 6801483) | [link](https://zenodo.org/records/6801483) | done |
| 16 | cbct | 20 | Walnut CBCT (42 walnuts) | [link](https://zenodo.org/records/2686726) | needs_upgrade |
| 17 | cest_mri | 20 | CEST MRI (niche) | -- | niche |
| 18 | ceus | 20 | BUS-BRA contrast US (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 19 | clem | 20 | CLEM (niche) | -- | niche |
| 20 | coded_exposure | 20 | BSD68 (standard CI test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 21 | confocal_3d | 20 | BioSR (2200+ SR pairs) | [link](https://figshare.com/articles/dataset/BioSR/13264793) | needs_upgrade |
| 22 | confocal_endomicroscopy | 20 | CVC-ClinicDB (612 frames) | [link](https://polyp.grand-challenge.org/CVCClinicDB/) | needs_upgrade |
| 23 | confocal_livecell | 20 | LIVECell (5239 images) | [link](https://sartorius-research.github.io/LIVECell/) | needs_upgrade |
| 24 | coronagraphy | 20 | ESO VLT coronagraph | [link](https://archive.eso.org/) | done |
| 25 | cryo_em | 20 | EMPIAR (micrograph archive) | [link](https://www.ebi.ac.uk/empiar/) | needs_upgrade |
| 26 | cryo_et | 20 | SHREC 2021 cryo-ET | [link](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA) | needs_upgrade |
| 27 | ct | 20 | LoDoPaB-CT (42K CT pairs) | [link](https://zenodo.org/records/3384092) | needs_upgrade |
| 28 | ct_fluorescence | 20 | CT fluorescence (niche) | -- | niche |
| 29 | cup | 20 | BSD68 (CUP test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 30 | dark_field | 20 | Dark-field X-ray (niche) | -- | niche |
| 31 | desi | 20 | DESI MSI (niche) | -- | niche |
| 32 | dexa | 20 | NHANES DXA | [link](https://www.cdc.gov/nchs/nhanes/) | needs_upgrade |
| 33 | dic | 20 | BBBC003 mouse embryo DIC | [link](https://bbbc.broadinstitute.org/BBBC003) | done |
| 34 | diffusion_mri | 20 | HCP diffusion MRI | [link](https://www.humanconnectome.org/) | needs_upgrade |
| 35 | digital_breast_tomo | 20 | DBT (niche) | -- | niche |
| 36 | dna_paint | 20 | DNA-PAINT (niche) | -- | niche |
| 37 | doppler_ultrasound | 20 | BUS-BRA breast US (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 38 | dot | 20 | DOT (niche) | -- | niche |
| 39 | ebsd | 20 | EBSD deformed iron (Zenodo 1214829) | [link](https://zenodo.org/records/1214829) | done |
| 40 | eddy_current | 20 | Eddy current (niche) | -- | niche |
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
| 51 | expansion | 20 | Expansion micro (niche) | -- | niche |
| 52 | fib_sem | 20 | FIB-SEM golgi (Zenodo 8114392) | [link](https://zenodo.org/records/8114392) | done |
| 53 | flash_lidar | 20 | Middlebury Stereo depth | [link](https://vision.middlebury.edu/stereo/) | done |
| 54 | flim | 20 | FLIM (niche) | -- | niche |
| 55 | fluoroscopy | 20 | WEISS Catheter Fluoroscopy | [link](https://rdr.ucl.ac.uk/articles/dataset/24624243) | needs_upgrade |
| 56 | fmri | 20 | OpenNeuro ds000114 BOLD fMRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 57 | fpm | 20 | FPM (niche) | -- | niche |
| 58 | ftir_imaging | 20 | FTIR breast tissue (Zenodo 4986399) | [link](https://zenodo.org/records/4986399) | done |
| 59 | fundus | 20 | DRIVE (40 retinal images) | [link](https://drive.grand-challenge.org/) | needs_upgrade |
| 60 | fwi | 20 | Marmousi2 / OpenFWI | [link](https://openfwi-lanl.github.io/) | done |
| 61 | gaussian_splatting | 11 | Tiny NeRF Lego + 3DGS | [link](https://github.com/graphdeco-inria/gaussian-splatting) | done |
| 62 | ghost_imaging | 20 | BSD68 (ghost imaging test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 63 | gpr | 20 | CMU-GPR / TU1208 radargrams | [link](https://github.com/rpl-cmu/CMU-GPR-Dataset) | needs_upgrade |
| 64 | gravitational_wave | 20 | GWOSC GW150914 strain | [link](https://gwosc.org/) | done |
| 65 | hdr_imaging | 20 | BSD68 (HDR test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 66 | holography | 20 | Electron holography (Zenodo 18289938) | [link](https://zenodo.org/records/18289938) | done |
| 67 | hyperspectral_remote | 20 | Indian Pines / Pavia Univ | [link](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | needs_upgrade |
| 68 | impedance_tomo | 20 | KTC 2023 EIT phantom (Zenodo 10986692) | [link](https://zenodo.org/records/10986692) | done |
| 69 | industrial_ct | 20 | Walnut CBCT micro-CT | [link](https://zenodo.org/records/2686726) | needs_upgrade |
| 70 | insar | 20 | InSAR (niche) | -- | niche |
| 71 | integral | 20 | BSD68 (integral imaging test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 72 | ism | 20 | ISM (niche) | -- | niche |
| 73 | ivus | 20 | BUS-BRA intravascular (Zenodo 7730709) | [link](https://zenodo.org/records/7730709) | done |
| 74 | lattice_lightsheet | 20 | Lattice LS (niche) | -- | niche |
| 75 | lensless | 20 | DiffuserCam | [link](https://waller-lab.github.io/DiffuserCam/) | needs_upgrade |
| 76 | libs | 20 | LIBS (niche) | -- | niche |
| 77 | lidar | 20 | SemanticKITTI (43K scans) | [link](https://semantic-kitti.org/) | needs_upgrade |
| 78 | light_field | 20 | Stanford Lytro light field | [link](http://lightfields.stanford.edu/LF2016.html) | done |
| 79 | lightsheet | 20 | CARE Tribolium | [link](https://publications.mpi-cbg.de/publications-sites/7207/) | needs_upgrade |
| 80 | lucky_imaging | 20 | ESO lucky imaging | [link](https://archive.eso.org/) | done |
| 81 | machine_vision | 20 | FaultSeg wheel defects (Zenodo 13162335) | [link](https://zenodo.org/records/13162335) | done |
| 82 | magnetic_particle | 20 | MPI (niche) | -- | niche |
| 83 | maldi_msi | 20 | MALDI MSI (niche) | -- | niche |
| 84 | mammography | 20 | CBIS-DDSM (10K mammograms) | [link](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) | needs_upgrade |
| 85 | matrix | 20 | EMDB density maps | [link](https://www.ebi.ac.uk/emdb/) | done |
| 86 | mfm | 20 | MFM (niche) | -- | niche |
| 87 | minflux | 20 | MINFLUX (niche) | -- | niche |
| 88 | mr_elastography | 20 | MRE (niche) | -- | niche |
| 89 | mr_fingerprinting | 20 | MRF (niche) | -- | niche |
| 90 | mra | 20 | OpenNeuro ds000114 brain MRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 91 | mri | 20 | fastMRI (8K+ volumes, NYU/Meta) | [link](https://fastmri.med.nyu.edu/) | needs_upgrade |
| 92 | mrs | 20 | MRS (niche) | -- | niche |
| 93 | multispectral_sat | 20 | EuroSAT Sentinel-2 (27K images) | [link](https://zenodo.org/records/7711810) | done |
| 94 | muon_tomo | 20 | Muon tomo (niche) | -- | niche |
| 95 | nerf | 11 | Tiny NeRF Lego scene | [link](https://github.com/bmild/nerf) | done |
| 96 | neutron_diffraction | 20 | Neutron diff (niche) | -- | niche |
| 97 | neutron_tomo | 20 | Neutron tomo (niche) | -- | niche |
| 98 | nirs_brain | 20 | fNIRS (niche) | -- | niche |
| 99 | nsom | 20 | NSOM (niche) | -- | niche |
| 100 | ocean_acoustic_tomo | 20 | ACOBAR Fram Strait OAT | [link](https://doi.org/10.1016/j.dib.2022.108160) | needs_upgrade |
| 101 | ocean_color | 20 | EuroSAT SeaLake Sentinel-2 | [link](https://zenodo.org/records/7711810) | done |
| 102 | oct | 20 | OCTDL retinal OCT (2064 images) | [link](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) | done |
| 103 | octa | 20 | OCTA-Mosaicking (Zenodo 14333858) | [link](https://zenodo.org/records/14333858) | done |
| 104 | odt | 20 | ODT (niche) | -- | niche |
| 105 | palm_storm | 20 | STORM tubulin (Zenodo 7620025) | [link](https://zenodo.org/records/7620025) | done |
| 106 | panorama | 20 | BSD68 (panorama test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 107 | particle_calorimetry | 20 | Calorimeter (niche) | -- | niche |
| 108 | passive_microwave | 20 | Passive MW (niche) | -- | niche |
| 109 | pet | 20 | UDPET ultra-low dose PET | [link](https://zenodo.org/records/6361846) | needs_upgrade |
| 110 | pet_ct | 20 | autoPET (TCIA, 1014 studies) | [link](https://autopet.grand-challenge.org/) | needs_upgrade |
| 111 | pet_mr | 20 | PET-MR (niche) | -- | niche |
| 112 | phase_contrast | 20 | ISBI Cell Tracking PhC | [link](https://celltrackingchallenge.net/) | needs_upgrade |
| 113 | phase_retrieval | 20 | Holographic phase (Zenodo 13771363) | [link](https://zenodo.org/records/13771363) | done |
| 114 | photoacoustic | 20 | Duke PAM mouse brain (Zenodo 4042171) | [link](https://zenodo.org/records/4042171) | done |
| 115 | photometric_stereo | 20 | BSD68 (photometric stereo test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 116 | polarization | 20 | Polarimetric camera (Zenodo 4483248) | [link](https://zenodo.org/records/4483248) | done |
| 117 | polsar | 20 | PolSAR (niche) | -- | niche |
| 118 | portal_imaging | 20 | Portal imaging (niche) | -- | niche |
| 119 | proton_radiography | 20 | Proton radiography (niche) | -- | niche |
| 120 | proton_therapy_img | 20 | Proton therapy (niche) | -- | niche |
| 121 | ptychography | 20 | Ptychography exp (Zenodo 16263064) | [link](https://zenodo.org/records/16263064) | done |
| 122 | pump_probe | 20 | Pump-probe (niche) | -- | niche |
| 123 | quantum_illumination | 20 | BSD68 (QI test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 124 | radio_astronomy | 20 | FIRST VLA survey | [link](https://www.cv.nrao.edu/first/) | needs_upgrade |
| 125 | radio_interferometry | 20 | ALMA/VLA archive | [link](https://almascience.eso.org/) | needs_upgrade |
| 126 | raman_imaging | 20 | Raman photothermal (Zenodo 8141012) | [link](https://zenodo.org/records/8141012) | done |
| 127 | sar | 20 | MSTAR (SAR target recognition) | [link](https://www.sdms.afrl.af.mil/index.php?collection=mstar) | needs_upgrade |
| 128 | saxs | 20 | SAXS (niche) | -- | niche |
| 129 | sd_cassi | 10 | CAVE 24-band hyperspectral | [link](https://www.cs.columbia.edu/CAVE/databases/multispectral/) | done |
| 130 | seismic_tomo | 20 | Marmousi2 elastic velocity | [link](https://wiki.seg.org/wiki/Open_data) | done |
| 131 | sem | 20 | NFFA-EUROPE SEM (21K images) | [link](https://b2share.eudat.eu/records/f1aa0f5ad38c456eaf7b04d47a65af53) | needs_upgrade |
| 132 | shearography | 20 | Shearography (niche) | -- | niche |
| 133 | shg | 20 | PSHG-TISS | [link](https://doi.org/10.17605/OSF.IO/K2Z8G) | needs_upgrade |
| 134 | sim | 20 | UniFMIR SIM F-actin (Zenodo 8420100) | [link](https://zenodo.org/records/8420100) | done |
| 135 | sims | 20 | SIMS (niche) | -- | niche |
| 136 | solar_imaging | 20 | NASA SDO AIA EUV composite | [link](https://sdo.gsfc.nasa.gov/) | done |
| 137 | sonar | 20 | UATD sonar (9200 FLS images) | [link](https://figshare.com/articles/dataset/UATD_Dataset/21331143) | needs_upgrade |
| 138 | spc | 20 | BSD68 (SPC test images) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 139 | spc_kronecker | 10 | Indian Pines AVIRIS | [link](https://engineering.purdue.edu/~biehl/MultiSpec/) | done |
| 140 | spect | 20 | SPECT (niche) | -- | niche |
| 141 | spect_ct | 20 | SPECT-CT (niche) | -- | niche |
| 142 | spectral_ct | 20 | Spectral CT (niche) | -- | niche |
| 143 | spinning_disk | 20 | Spinning disk (niche) | -- | niche |
| 144 | srs | 20 | SRS (niche) | -- | niche |
| 145 | sted | 20 | UniFMIR STED (Zenodo 8420100) | [link](https://zenodo.org/records/8420100) | done |
| 146 | stem | 20 | STEM (niche) | -- | niche |
| 147 | stm | 20 | Graphene/Ni STM (Zenodo 5799774) | [link](https://zenodo.org/records/5799774) | needs_upgrade |
| 148 | streak_camera | 20 | BSD68 (streak camera test) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | done |
| 149 | structured_light | 20 | Middlebury Stereo depth | [link](https://vision.middlebury.edu/stereo/) | done |
| 150 | swi | 20 | OpenNeuro ds000114 brain MRI | [link](https://openneuro.org/datasets/ds000114) | done |
| 151 | talbot_lau | 20 | Talbot-Lau (niche) | -- | niche |
| 152 | tem | 20 | TEM cilia (Zenodo 11188503) | [link](https://zenodo.org/records/11188503) | done |
| 153 | terahertz | 20 | Active THz dataset | [link](https://github.com/LingLIx/THz_Dataset) | needs_upgrade |
| 154 | three_photon | 20 | Three-photon (niche) | -- | niche |
| 155 | tirf | 20 | TIRF (niche) | -- | niche |
| 156 | tof_camera | 20 | ToF depth maps (Zenodo 10732158) | [link](https://zenodo.org/records/10732158) | done |
| 157 | two_photon | 20 | CaImAn calcium imaging | [link](https://github.com/flatironinstitute/CaImAn) | done |
| 158 | ultrasonic_phased_array | 20 | UT phased array (niche) | -- | niche |
| 159 | ultrasound | 20 | BUSI breast ultrasound (780 images) | [link](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) | done |
| 160 | us_mri | 20 | UTE MRI (niche) | -- | niche |
| 161 | waxs | 20 | WAXS (niche) | -- | niche |
| 162 | weather_radar | 20 | NOAA NEXRAD composite | [link](https://www.ncei.noaa.gov/products/radar) | done |
| 163 | widefield | 20 | FMD widefield denoising | [link](https://github.com/yinhaoz/denoising-fluorescence) | needs_upgrade |
| 164 | widefield_lowdose | 20 | FMD low-dose | [link](https://github.com/yinhaoz/denoising-fluorescence) | needs_upgrade |
| 165 | xfel_sfx | 20 | EMDB XFEL SFX | [link](https://www.ebi.ac.uk/emdb/) | done |
| 166 | xray_crystallography | 20 | EMDB/PDB X-ray crystallography | [link](https://www.ebi.ac.uk/emdb/) | done |
| 167 | xray_ndt | 20 | GDXray (19K NDT images) | [link](https://domingomery.ing.puc.cl/material/gdxray/) | needs_upgrade |
| 168 | xray_radiography | 20 | NIH ChestX-ray14 | [link](https://nihcc.app.box.com/v/ChestXray-NIHCC) | done |
| 169 | xrf_imaging | 20 | XRF fossil map (Zenodo 4005031) | [link](https://zenodo.org/records/4005031) | done |
| 170 | xrf_tomo | 20 | XRF tomography (niche) | -- | niche |
