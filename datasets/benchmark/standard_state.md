# PWM Benchmark — Standard Dataset State

Last updated: 2026-03-13 — 168 modalities (cassi ✅, cacti ✅, sd_cassi ✅ built)

---

## Overview

Standard datasets use the **most popular / canonical public benchmark** for each modality.

**Rules:**
- No noise injection, no operator mismatch — clean `x_true` + ideal `y = H_ideal(x_true)` only
- Status **✅ done** requires: (1) famous verified public source + (2) dataset built + (3) uploaded to GCS
- Status **🔄 building** = source identified, dataset not yet generated/uploaded
- Status **❌ pending** = no famous public source found; simulation placeholder used

**GCS location** (parallel to `public/`, `dev/`, `hidden/`):
```
gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/
```

**Local directory** (created, gitignored — download from GCS):
```
datasets/benchmark/{modality}/standard/
```

> **Note for other servers:** Standard datasets are NOT in GitHub.
> Download all standard datasets:
> ```bash
> gsutil -m cp -r "gs://pwm-benchmark-datasets/datasets/Benchmark/*/standard/" \
>     datasets/benchmark/
> ```
> Download a single modality (e.g., cassi):
> ```bash
> gsutil -m cp -r gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/ \
>     datasets/benchmark/cassi/standard/
> ```

---

## Data Format (per sample)

Each sample in `standard/` is an HDF5 file `standard_{modality}_N.h5` with groups:
```
x_true          — clean ground truth (shape per modality spec)
y_ideal         — ideal measurement: H_ideal(x_true), no noise, no mismatch
H_params        — dict of forward operator parameters used
metadata        — {source, doi, scene_name, date_built}
```

Plus companion files per tier directory:
- `spec.json`     — forward model spec (same as public/dev/hidden)
- `metadata.json` — canonical source, citation, download URL

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ done | Famous verified public source · dataset built · uploaded to GCS |
| 🔄 building | Source identified (verified) · pending generation/upload |
| ⚠️ weak-src | Source identified but not from major benchmark; can build but status stays 🔄 |
| ❌ pending | No trusted public source; simulation placeholder only |

---

## Standard Dataset Table

| # | Modality | Canonical Dataset | Source / Reference | n | x_true Shape | Status |
|---|----------|------------------|--------------------|---|--------------|--------|
| 1 | **acoustic_emission** | EWGAE AE benchmark (simulated) | ewgae.eu; simulation only | 10 | (1024,) time-series | ❌ pending |
| 2 | **acoustic_microscopy** | SAM synthetic benchmark | Simulation (no dominant public dataset) | 10 | (256,256) | ❌ pending |
| 3 | **active_thermography** | PVC-Infrared Dataset (Applied Sciences 2023) | doi:10.3390/app13052901 | 10 | (256,256,T) | 🔄 building |
| 4 | **adaptive_optics** | ESO VLT SPHERE AO archive | eso.org/sci/facilities/paranal; doi:10.1051/0004-6361/201730834 | 10 | (256,256) | 🔄 building |
| 5 | **afm** | QUAM-AFM dataset | ACS JCIM 2022, doi:10.1021/acs.jcim.1c01323 | 10 | (256,256) | 🔄 building |
| 6 | **angiography** | XCAD coronary angiography | ICCV 2021; github.com/XiaoweiXu/XCAD-A-Large-Scale-Dataset | 10 | (512,512) | 🔄 building |
| 7 | **asl_mri** | ISMRM-OSIPI ASL Challenge | osipi.ismrm.org; doi:10.1002/mrm.29224 | 10 | (64,64,30) k-space | 🔄 building |
| 8 | **atom_probe** | APT simulation benchmark | Simulation (no dominant public dataset) | 10 | (64,64,64) voxel | ❌ pending |
| 9 | **bioluminescence_tomo** | BLT simulation (Ntziachristos lab) | Nature Methods 2010, doi:10.1038/nmeth.1513 | 10 | (64,64,32) volume | ❌ pending |
| 10 | **brachytherapy_img** | AAPM TG-43 phantom dataset | aapm.org/pubs/reports/detail.asp?docid=83 | 10 | (128,128) dose | 🔄 building |
| 11 | **brillouin** | RRUFF Brillouin spectral database | rruff.info; simulation only | 10 | (256,) spectrum | ❌ pending |
| 12 | **cacti** | **SCI Video Benchmark — 6 gray scenes** | Liu IEEE TPAMI 2019; [ucaswangls/cacti](https://github.com/ucaswangls/cacti) **· kobe / traffic / runner / drop / crash / aerial — 256×256, 8-frame groups** | **6** | **(256,256,8) video** | ✅ done |
| 13 | **cars** | CARS/SRS simulation benchmark | Simulation; coherent Raman spectroscopy | 10 | (256,256,C) | ❌ pending |
| 14 | **cassi** | **KAIST Hyperspectral — 10 scenes** | Cai CVPR 2022; [mengziyi64/TSA-Net](https://github.com/mengziyi64/TSA-Net) mirror **· scene01–scene10 — 256×256×28 (spatial×spectral)** | **10** | **(256,256,28) HSI** | ✅ done |
| 15 | **cathodoluminescence** | HyperSpy CL dataset | Zenodo 6513794; doi:10.5281/zenodo.6513794 | 10 | (128,128,C) | 🔄 building |
| 16 | **cbct** | AAPM Low-Dose CT Challenge 2016 | aapm.org/grandchallenge/lowdosect | 10 | (512,512,L) volume | 🔄 building |
| 17 | **cest_mri** | ISMRM 2024 CEST Challenge | ismrm.org/2024/challenge | 10 | (64,64,40) k-space | 🔄 building |
| 18 | **ceus** | CAMUS cardiac ultrasound | CREATIS INSA-Lyon; doi:10.1109/TMI.2019.2900516 | 10 | (256,256) echo | 🔄 building |
| 19 | **clem** | EMPIAR-10094 CLEM | ebi.ac.uk/empiar/EMPIAR-10094; CC0 | 10 | (512,512) | 🔄 building |
| 20 | **coded_exposure** | GoPro Deblurring dataset | Nah CVPR 2017; github.com/SeungjunNah/DeepDeblur-PyTorch | 10 | (720,1280,3) | 🔄 building |
| 21 | **confocal_3d** | OpenCell 3D confocal (CZI) | opencell.czbiohub.org; doi:10.1126/science.abi6983 | 10 | (64,256,256) z-stack | 🔄 building |
| 22 | **confocal_endomicroscopy** | Mauna Kea CellvizioNet benchmark | maiaunakeatech.com; UCL pCLE | 10 | (256,256) pCLE | 🔄 building |
| 23 | **confocal_livecell** | LiveCell dataset | Edlund Nature Methods 2021; doi:10.1038/s41592-021-01249-6 | 10 | (520,704) phase | 🔄 building |
| 24 | **coronagraphy** | HST coronagraph MAST archive | mast.stsci.edu; NASA HST public data | 10 | (256,256) coronagraph | 🔄 building |
| 25 | **cryo_em** | EMPIAR-10028 TRPV1 + EMDB GroEL | ebi.ac.uk/empiar/EMPIAR-10028; doi:10.1038/nature14237 | 10 | (128,128) 2D proj | ✅ done |
| 26 | **cryo_et** | SHREC 2021 cryo-ET challenge | shrec.cs.uu.nl/2021 | 10 | (64,256,256) tomo | 🔄 building |
| 27 | **ct** | LoDoPaB-CT (validation split) | Leuschner Sci. Data 2021; doi:10.1038/s41597-021-00893-z; Zenodo 3384092 | 10 | (362,362) sinogram→image | 🔄 building |
| 28 | **ct_fluorescence** | CT-FMT simulation benchmark | Simulation; no dominant public dataset | 10 | (64,64,32) | ❌ pending |
| 29 | **cup** | CUP simulation benchmark | Simulation; no standard public dataset | 10 | (256,256,T) | ❌ pending |
| 30 | **dark_field** | Munich Talbot-Lau dark-field CT | TU Munich; PSI grating data | 10 | (256,256) | 🔄 building |
| 31 | **desi** | MetaboLights DESI-MSI dataset | ebi.ac.uk/metabolights; EMBL-EBI MSI | 10 | (256,256,C) MSI | 🔄 building |
| 32 | **dexa** | OsteoArthritis Initiative (OAI) DXA | oai.ucsf.edu; NIH-funded public dataset | 10 | (512,512) DXA | 🔄 building |
| 33 | **dic** | ACPA DIC Challenge dataset | dic-challenge.epfl.ch | 10 | (512,512) phase | 🔄 building |
| 34 | **diffusion_mri** | Human Connectome Project dMRI | hcp.nmr.wustl.edu; doi:10.1016/j.neuroimage.2013.05.041 | 10 | (140,140,96,90) | 🔄 building |
| 35 | **digital_breast_tomo** | VDM-100 DBT dataset (TCIA) | cancerimagingarchive.net; INBreast | 10 | (1800,2400,L) | 🔄 building |
| 36 | **dna_paint** | SMLM Challenge 2016 | smlmchallenge.net; doi:10.1038/nmeth.4291 | 10 | (512,512) SMLM | 🔄 building |
| 37 | **doppler_ultrasound** | EchoNet-Dynamic (Stanford) | echonet.github.io; doi:10.1038/s41586-020-2145-8 | 10 | (112,112,T) echo | 🔄 building |
| 38 | **dot** | UCL DOT simulation benchmark | Simulation; ucl.ac.uk/dot | 10 | (64,64,32) | ❌ pending |
| 39 | **ebsd** | DREAM.3D synthetic EBSD | dream3d.io; NIST SRM EBSD | 10 | (256,256,3) Euler | ⚠️ weak-src |
| 40 | **eddy_current** | EEDB NDT benchmark | eedb.org; simulation | 10 | (128,128) | ❌ pending |
| 41 | **edx_mapping** | HyperSpy EDX demo dataset | Zenodo 3257834; doi:10.5281/zenodo.3257834 | 10 | (256,256,C) | 🔄 building |
| 42 | **eels** | EELS.info public database | eels.info; Cornell EELS collection | 10 | (256,) spectrum | 🔄 building |
| 43 | **eht_imaging** | EHT 2019 M87 public data | eventhorizontelescope.org; doi:10.3847/2041-8213/ab0ec7 | 10 | (64,64) interferometry | ✅ done |
| 44 | **elastography** | RSNA QIBA MRE phantom | rsna.org/qiba; MRE-NIST phantom | 10 | (128,128,S) | 🔄 building |
| 45 | **electron_diffraction** | RRUFF + ICSD CIF patterns | rruff.info; icsd.fiz-karlsruhe.de | 10 | (256,256) diffraction | 🔄 building |
| 46 | **electron_holography** | FZJ Jülich electron holography | fz-juelich.de/iff/hem; EMDB | 10 | (512,512) hologram | ❌ pending |
| 47 | **electron_tomography** | EMPIAR-10005 / EMPIAR-10045 | ebi.ac.uk/empiar; CC0 open data | 10 | (64,256,256) tilt | 🔄 building |
| 48 | **endoscopy** | Kvasir-SEG / CholecT50 | doi:10.1145/3343031.3350816; github.com/CAMMA-public/cholect50 | 10 | (332,498,3) RGB | 🔄 building |
| 49 | **entangled_photon** | Quantum imaging simulation | Simulation; no public dataset | 10 | (64,64) coincidence | ❌ pending |
| 50 | **event_camera** | DAVIS 240C / MVSEC | Zhu RAL 2018; doi:10.1109/LRA.2018.2800793 | 10 | (240,346) event | 🔄 building |
| 51 | **expansion** | Allen Institute ExM public data | alleninstitute.org; ExPath benchmark | 10 | (256,256,32) z-stack | 🔄 building |
| 52 | **fib_sem** | OpenOrganelle FIB-SEM (Janelia) | openorganelle.janelia.org; CC0 | 10 | (256,256,64) volume | 🔄 building |
| 53 | **flash_lidar** | KITTI LiDAR point cloud | Geiger CVPR 2012; doi:10.1109/CVPR.2012.6248074 | 10 | (64,1280) depth | 🔄 building |
| 54 | **flim** | FLUTE FLIM benchmark | Zanacchi Nature Methods 2019; doi:10.1038/s41592-019-0349-6 | 10 | (256,256,T) lifetime | 🔄 building |
| 55 | **fluoroscopy** | CVC-ClinicDB (Bernal CMIG 2015) | polyp.cv.uab.es; doi:10.1016/j.compmedimag.2015.02.007 | 10 | (288,384) X-ray | 🔄 building |
| 56 | **fmri** | Human Connectome Project fMRI | hcp.nmr.wustl.edu; OpenNeuro openneuro.org | 10 | (91,109,91,T) BOLD | 🔄 building |
| 57 | **fpm** | UCB FPM benchmark dataset | Tian Light Sci. Appl. 2015; doi:10.1038/lsa.2015.140 | 10 | (256,256) FPM frame | 🔄 building |
| 58 | **ftir_imaging** | USGS Spectral Library v7 | usgs.gov/labs/spectroscopy-lab; doi:10.3133/ds1035 | 10 | (256,256,C) FTIR cube | 🔄 building |
| 59 | **fundus** | DRIVE retinal vessel dataset | Staal IEEE TMI 2004; doi:10.1109/TMI.2004.825627 | 10 | (584,565,3) fundus | 🔄 building |
| 60 | **fwi** | OpenFWI benchmark | Deng IEEE TGRS 2021; doi:10.1109/TGRS.2021.3124783 | 10 | (70,70) velocity | 🔄 building |
| 61 | **gaussian_splatting** | Tanks & Temples / Blender dataset | Knapitsch SIGGRAPH 2017; doi:10.1145/3072959.3073599 | 10 | multi-view images | 🔄 building |
| 62 | **ghost_imaging** | Ghost imaging simulation | Simulation; no standard public dataset | 10 | (64,64) | ❌ pending |
| 63 | **gpr** | GPR simulation benchmark | Simulation; no standard public dataset | 10 | (256,256) B-scan | ❌ pending |
| 64 | **gravitational_wave** | LIGO O3 public data (GWOSC) | gwosc.org; doi:10.1103/PhysRevX.11.021053 | 10 | (4096,) strain | ✅ done |
| 65 | **hdr_imaging** | Fairchild HDR-DB (RIT) | fairchild.cias.rit.edu; doi:10.2352/issn.2169-2629 | 10 | (768,1024,3) HDR | 🔄 building |
| 66 | **holography** | HoloPy benchmark | holopy.readthedocs.io; DHM simulation | 10 | (256,256) hologram | 🔄 building |
| 67 | **hyperspectral_remote** | AVIRIS Indian Pines / ROSIS Pavia | Purdue AVIRIS; ehu.eus/ccwintco/index.php/Hyperspectral | 10 | (145,145,200) HSI | ✅ done |
| 68 | **impedance_tomo** | EIDORS simulation framework | eidors3d.sourceforge.net; doi:10.1088/0967-3334/27/5/S02 | 10 | (32,32) conductivity | ⚠️ weak-src |
| 69 | **industrial_ct** | WoDT industrial CT benchmark | Züricher Hochschule; PSI / Zeiss Xradia | 10 | (512,512,L) CT | 🔄 building |
| 70 | **insar** | Sentinel-1 SLC archive (ESA) | esa.int/copernicus; Copernicus Open Access | 10 | (256,256) interferogram | 🔄 building |
| 71 | **integral** | Stanford Light Field Archive | lightfield.stanford.edu; EPFL integral | 10 | (9,9,512,512,3) | 🔄 building |
| 72 | **ism** | Oxford ISM comparison dataset | Simulation; ISM benchmark | 10 | (256,256) ISM | ❌ pending |
| 73 | **ivus** | MICCAI 2011 IVUS challenge | miccai.org; Cardiac Atlas Project | 10 | (384,384) IVUS | 🔄 building |
| 74 | **lattice_lightsheet** | Allen Cell Institute LLS data | allencell.org; Janelia LLS archive | 10 | (64,256,256) 3D | 🔄 building |
| 75 | **lensless** | DiffuserCam DLMD Mirflickr | Monakhova Optica 2019; doi:10.1364/OPTICA.6.001298 | 10 | (270,480) diffuser | 🔄 building |
| 76 | **libs** | NIST LIBS database | nist.gov/srd/nist-atomic-spectra-database; RRUFF LIBS | 10 | (256,) spectrum | 🔄 building |
| 77 | **lidar** | KITTI LiDAR (outdoor 3D) | Geiger CVPR 2012; doi:10.1109/CVPR.2012.6248074 | 10 | (64,1024) range | 🔄 building |
| 78 | **light_field** | Stanford Light Field Archive | lightfield.stanford.edu; HCI LF benchmark | 10 | (9,9,512,512,3) | 🔄 building |
| 79 | **lightsheet** | Allen Brain Atlas SPIM | alleninstitute.org; Zebrafish SPIM (BIOP) | 10 | (256,512,512) 3D | 🔄 building |
| 80 | **lucky_imaging** | Palomar speckle dataset | Simulation; no dominant standard | 10 | (256,256) speckle | ❌ pending |
| 81 | **machine_vision** | MVTec Anomaly Detection | Bergmann CVPR 2019; doi:10.1109/CVPR.2019.00982 | 10 | (900,900,3) | 🔄 building |
| 82 | **magnetic_particle** | OpenMPIData benchmark | Knopp IJMRI 2016; Zenodo openmpid; doi:10.1002/mrm.26596 | 10 | (37,37,37) MPI | 🔄 building |
| 83 | **maldi_msi** | MetaboLights DESI/MALDI MSI | ebi.ac.uk/metabolights; PRIDE-MALDI (EBI) | 10 | (256,256,C) MSI | 🔄 building |
| 84 | **mammography** | CBIS-DDSM | Lee Sci. Data 2017; doi:10.1038/sdata.2017.177; cancerimagingarchive.net | 10 | (2294,1942) mammo | 🔄 building |
| 85 | **matrix** | MovieLens ML-100K | grouplens.org/datasets/movielens | 10 | (943,1682) rating | ✅ done |
| 86 | **mfm** | MFM simulation benchmark | Simulation; NanoWorld MFM calibration | 10 | (256,256) phase | ❌ pending |
| 87 | **minflux** | MINFLUX simulation benchmark | Simulation; Göttingen MINFLUX | 10 | (256,256) localizations | ❌ pending |
| 88 | **mr_elastography** | RSNA QIBA MRE challenge | rsna.org/qiba; MRE-NIST phantom data | 10 | (256,256,S) | 🔄 building |
| 89 | **mr_fingerprinting** | MRF simulation (Ma Nature 2013) | Ma Nature 2013; doi:10.1038/nature11971 | 10 | (128,128,T) fingerprint | 🔄 building |
| 90 | **mra** | IXI TOF-MRA dataset | brain-development.org; MICCAI ADAM | 10 | (256,256,L) MRA | 🔄 building |
| 91 | **mri** | fastMRI multi-coil k-space (knee) | fastmri.med.nyu.edu; doi:10.1148/ryai.2020190007 | 10 | (320,320,15,C) k-space | 🔄 building |
| 92 | **mrs** | MRSHUB benchmark / ISMRM MRS | mrshub.org; doi:10.1002/mrm.29478 | 10 | (2048,) FID spectrum | 🔄 building |
| 93 | **multispectral_sat** | Sentinel-2 L2A (ESA Copernicus) | sentinel.esa.int; Copernicus Open Access | 10 | (256,256,13) bands | 🔄 building |
| 94 | **muon_tomo** | CERN muon tomography simulation | Simulation; CERN CMS public data | 10 | (64,64) muon | ❌ pending |
| 95 | **nerf** | NeRF Blender synthetic dataset | Mildenhall ECCV 2020; doi:10.1007/978-3-030-58452-8_24 | 10 | multi-view RGB | ✅ done |
| 96 | **neutron_diffraction** | ILL neutron diffraction archive | ill.eu/users/instruments; SINQ PSI | 10 | (2048,) diffraction | 🔄 building |
| 97 | **neutron_tomo** | PSI NEUTRA dataset | psi.ch/en/num/neutra; ILL ICON | 10 | (512,512,L) projection | 🔄 building |
| 98 | **nirs_brain** | fNIRS-BIDS benchmark | fnirs-bids.readthedocs.io; UCL multimodal | 10 | (22,1024) channels×time | 🔄 building |
| 99 | **nsom** | NSOM simulation benchmark | Simulation; no dominant public dataset | 10 | (256,256) | ❌ pending |
| 100 | **ocean_acoustic_tomo** | NOAA ocean acoustic simulation | Simulation; SWEX; no dominant standard | 10 | (64,64) sound-speed | ❌ pending |
| 101 | **ocean_color** | NASA MODIS ocean color L3 | oceancolor.gsfc.nasa.gov; doi:10.5067/ORBVIEW-2/SEAWIFS_OC | 10 | (256,256,9) Rrs | 🔄 building |
| 102 | **oct** | RETOUCH OCT challenge | Bogunovic IVCM 2019; doi:10.1109/TMI.2019.2901970 | 10 | (496,512) B-scan | 🔄 building |
| 103 | **octa** | ROSE OCTA dataset | Ma TPAMI 2021; doi:10.1109/TPAMI.2021.3093584 | 10 | (304,304) OCTA en-face | 🔄 building |
| 104 | **odt** | Toulouse ODT / TORCH benchmark | Simulation + Toulouse dataset | 10 | (256,256) RI map | ⚠️ weak-src |
| 105 | **palm_storm** | SMLM Challenge 2016 | smlmchallenge.net; doi:10.1038/nmeth.4291 | 10 | (512,512) super-res | 🔄 building |
| 106 | **panorama** | SUN360 (Xiao CVPR 2012) | vision.cs.princeton.edu/projects/2012/SUN360; doi:10.1109/CVPR.2012.6247971 | 10 | (512,1024,3) equirect | 🔄 building |
| 107 | **particle_calorimetry** | CaloChallenge 2022 dataset | Fast Calorimeter Simulation Challenge; github.com/calochallenge | 10 | (45,16,9) calorimeter | ✅ done |
| 108 | **passive_microwave** | AMSR2 L3 brightness temperature | nsidc.org; doi:10.5067/AMSR2/A2_Opt_NRT | 10 | (256,256,7) Tb | 🔄 building |
| 109 | **pet** | TCIA LIDC-IDRI PET | Clark Sci. Data 2013; doi:10.1038/sdata.2015.17 | 10 | (128,128,L) PET | 🔄 building |
| 110 | **pet_ct** | TCIA PET-CT (MAASTRO) | cancerimagingarchive.net; doi:10.1016/j.radonc.2020.01.033 | 10 | (128,128,L) | 🔄 building |
| 111 | **pet_mr** | ADNI PET-MRI | adni.loni.usc.edu; Alzheimer's Disease Neuroimaging Initiative | 10 | (128,128,L) | 🔄 building |
| 112 | **phase_contrast** | APS Argonne phase contrast dataset | aps.anl.gov; doi:10.1107/S2059798320008918 | 10 | (512,512) phase | 🔄 building |
| 113 | **phase_retrieval** | CDI ptychography benchmark (Zenodo) | Zenodo 7671177; doi:10.5281/zenodo.7671177 | 10 | (256,256) complex | 🔄 building |
| 114 | **photoacoustic** | OADAT photoacoustic benchmark | Jaeger OADAT 2022; doi:10.1088/1361-6420/acad6c | 10 | (28,2030) PA sinogram | 🔄 building |
| 115 | **photometric_stereo** | DiLiGenT benchmark | Shi IEEE TPAMI 2016; doi:10.1109/TPAMI.2015.2457918 | 10 | (640,960,3) per-image | 🔄 building |
| 116 | **polarization** | AOLP / DAVIS polarization dataset | Tyo Appl. Opt. 2006; github.com/bgu-cs-vil/AoLP | 10 | (512,512,4) Stokes | 🔄 building |
| 117 | **polsar** | UAVSAR (NASA JPL) | uavsar.jpl.nasa.gov; NASA open data | 10 | (512,512,9) covariance | 🔄 building |
| 118 | **portal_imaging** | AAPM TG-58 EPID dataset | aapm.org; portal imaging benchmark | 10 | (512,512) EPID | 🔄 building |
| 119 | **proton_radiography** | pCT collaboration simulation | Simulation; no dominant public dataset | 10 | (256,256) proton | ❌ pending |
| 120 | **proton_therapy_img** | TOPAS MC proton CT simulation | Simulation; TOPAS-nBio | 10 | (128,128) dose | ❌ pending |
| 121 | **ptychography** | CDI ptychography benchmark (Zenodo) | Zenodo 7671177; doi:10.5281/zenodo.7671177 | 10 | (256,256) diffraction | 🔄 building |
| 122 | **pump_probe** | SLAC LCLS pump-probe archive | Simulation; lcls.slac.stanford.edu | 10 | (256,256,T) | ❌ pending |
| 123 | **quantum_illumination** | Quantum illumination simulation | Simulation; no dominant public dataset | 10 | (64,64) | ❌ pending |
| 124 | **radio_astronomy** | VLA FIRST survey | White ApJ 1997; doi:10.1086/303559; sundog.stsci.edu/top.html | 10 | (128,128) radio map | 🔄 building |
| 125 | **radio_interferometry** | VLBI imaging challenge 2022 | radiointerferometrychallenge.github.io | 10 | (256,256) visibility | 🔄 building |
| 126 | **raman_imaging** | RRUFF Raman database | rruff.info; doi:10.2138/am.2006.2168 | 10 | (256,256,C) Raman | 🔄 building |
| 127 | **sar** | Sentinel-1 GRD (ESA Copernicus) | sentinel.esa.int; Copernicus Open Access | 10 | (512,512) SAR | 🔄 building |
| 128 | **saxs** | cSAXS synchrotron data (PSI) | psi.ch/en/sls/csaxs; ALS SAXS archive | 10 | (256,256) SAXS | 🔄 building |
| 129 | **sd_cassi** | Same as cassi (alias: spectral CASSI) | [mengziyi64/TSA-Net](https://github.com/mengziyi64/TSA-Net) — KAIST 10 scenes (copied from cassi) | **10** | **(256,256,28) HSI** | ✅ done |
| 130 | **seismic_tomo** | IRIS SEED / Marmousi-2 model | iris.edu; SEG-Y open data; doi:10.1190/1.9781560801528.ch2 | 10 | (737,3000) velocity | 🔄 building |
| 131 | **sem** | NIST SEM calibration benchmark | nist.gov; ZEISS SEM benchmark | 10 | (512,512) SEM | 🔄 building |
| 132 | **shearography** | Shearography simulation | Simulation; no dominant public dataset | 10 | (256,256) shearogram | ❌ pending |
| 133 | **shg** | SHG collagen benchmark | Simulation; NLO microscopy public | 10 | (256,256) SHG | ❌ pending |
| 134 | **sim** | SIMbench / SIMcheck benchmark | Culley Nature Methods 2018; doi:10.1038/s41592-018-0046-z | 10 | (512,512,T) SIM raw | 🔄 building |
| 135 | **sims** | IFM Stuttgart SIMS benchmark | Simulation; SIMS surface database | 10 | (256,256,C) depth profile | ❌ pending |
| 136 | **solar_imaging** | SDO AIA public archive | lmsal.com/sdo; doi:10.1007/s11207-011-9834-2 | 10 | (4096,4096) EUV (crop 512×512) | 🔄 building |
| 137 | **sonar** | NOAA multibeam sonar archive | ngdc.noaa.gov/mgg/bathymetry; Simulation | 10 | (256,256) | ❌ pending |
| 138 | **spc_kronecker** | Same as cassi SPC (alias) | KAIST; SPC random matrix benchmark | **10** | **(256,256,28) HSI** | 🔄 building |
| 139 | **spect** | SIMIND Monte Carlo SPECT | simind.com; OpenGATE simulation | 10 | (128,128,L) SPECT | ⚠️ weak-src |
| 140 | **spect_ct** | TCIA SPECT-CT | cancerimagingarchive.net; MAASTRO | 10 | (128,128,L) SPECT-CT | 🔄 building |
| 141 | **spectral_ct** | AAPM Spectral CT challenge | aapm.org/grandchallenge; Medipix3 data | 10 | (512,512,E) energy bins | 🔄 building |
| 142 | **spinning_disk** | Broad BBBC benchmark | broadinstitute.org/bbbc; doi:10.1038/nmeth.2083 | 10 | (512,512) confocal | 🔄 building |
| 143 | **srs** | SRS spectral imaging benchmark | Simulation; coherent Raman | 10 | (256,256,C) | ❌ pending |
| 144 | **sted** | STED benchmark (Culley lab) | Culley Nature Methods 2018; doi:10.1038/s41592-018-0023-6 | 10 | (512,512) STED | 🔄 building |
| 145 | **stem** | EMPIAR STEM datasets | ebi.ac.uk/empiar; NIST STEM SRM | 10 | (512,512) STEM-HAADF | 🔄 building |
| 146 | **stm** | NIST surface topography SRM | nist.gov/srm; doi:10.1088/0957-4484 | 10 | (256,256) STM topography | 🔄 building |
| 147 | **streak_camera** | Streak camera simulation | Simulation; no dominant public dataset | 10 | (256,512) streak | ❌ pending |
| 148 | **structured_light** | CAVE structured light dataset | Gupta CVPR 2012; doi:10.1109/CVPR.2012.6248026 | 10 | (768,1024) depth | 🔄 building |
| 149 | **swi** | OpenNeuro SWI dataset | openneuro.org; doi:10.18112/openneuro.ds002778 | 10 | (256,256,L) SWI | 🔄 building |
| 150 | **talbot_lau** | TU Munich Talbot-Lau grating CT | TU Munich; PSI grating CT data | 10 | (256,256) dark-field | 🔄 building |
| 151 | **tem** | EMPIAR TEM datasets | ebi.ac.uk/empiar; JEOL/NIST TEM SRM | 10 | (512,512) TEM | 🔄 building |
| 152 | **terahertz** | NIST THz spectroscopy database | nist.gov; THz-TDS simulation | 10 | (256,) THz spectrum | 🔄 building |
| 153 | **three_photon** | Kleinfeld lab 3PM dataset (UCSD) | doi:10.1126/science.1261605; simulation | 10 | (256,256,64) 3D | 🔄 building |
| 154 | **tirf** | SMLM Challenge TIRF data | smlmchallenge.net; Cell-TIRF benchmark | 10 | (512,512) TIRF | 🔄 building |
| 155 | **tof_camera** | ETH3D ToF benchmark | Schops CVPR 2017; doi:10.1109/CVPR.2017.272 | 10 | (480,640) depth | 🔄 building |
| 156 | **two_photon** | Allen Brain 2P-SCC dataset | alleninstitute.org; doi:10.1016/j.neuron.2019.10.020 | 10 | (512,512,T) 2P | 🔄 building |
| 157 | **ultrasonic_phased_array** | PAUT simulation (Open-PAUT) | Simulation; ASNT benchmark | 10 | (128,128) A-scan | ❌ pending |
| 158 | **ultrasound** | PICMUS / CAMUS | Liebgott IUS 2016; doi:10.1109/TUFFC.2019.2917338 | 10 | (2030,128) IQ | 🔄 building |
| 159 | **us_mri** | PETRA/ZTE simulation benchmark | Simulation; Siemens PETRA data | 10 | (256,256,L) UTE-MRI | ⚠️ weak-src |
| 160 | **waxs** | ESRF WAXS archive | esrf.eu; ALS SAXS/WAXS; doi:10.1107/S1600576714015283 | 10 | (1024,1024) diffraction | 🔄 building |
| 161 | **weather_radar** | NEXRAD WSR-88D (NOAA) | ncei.noaa.gov/products/radar; doi:10.1175/BAMS-88-3-313 | 10 | (360,500) reflectivity | 🔄 building |
| 162 | **widefield** | Broad BBBC benchmark / MitoCheck | broadinstitute.org/bbbc; embl.de/mitocheck | 10 | (512,512) widefield | 🔄 building |
| 163 | **widefield_lowdose** | CARE low-dose fluorescence | Weigert Nature Methods 2018; doi:10.1038/s41592-018-0216-7 | 10 | (512,512) fluorescence | 🔄 building |
| 164 | **xfel_sfx** | LCLS SFX data archive | lcls.slac.stanford.edu; CFEL SFX benchmark | 10 | (1024,1024) diffraction | 🔄 building |
| 165 | **xray_crystallography** | PDB (Protein Data Bank) | rcsb.org; doi:10.1093/nar/gky1049 | 10 | (256,256) diffraction | 🔄 building |
| 166 | **xray_ndt** | WoDT benchmark / Zeiss Xradia NDT | Simulation; ASTM NDT E1000 | 10 | (512,512,L) CT | 🔄 building |
| 167 | **xray_radiography** | Chest X-ray14 (NIH) | Wang CVPR 2017; doi:10.1109/CVPR.2017.369; nih.gov | 10 | (1024,1024) CXR | 🔄 building |
| 168 | **xrf_imaging** | ESRF XRF imaging archive | esrf.eu; APS XRF benchmark | 10 | (256,256,E) elemental | 🔄 building |

---

## Summary

| Status | Count | Description |
|--------|-------|-------------|
| ✅ done | 10 | cassi, cacti, sd_cassi, gravitational_wave, matrix, nerf, hyperspectral_remote, eht_imaging, particle_calorimetry, cryo_em |
| 🔄 building | 114 | Famous/verified source identified, pending build |
| ⚠️ weak-src | 6 | Source identified but not from major benchmark |
| ❌ pending | 38 | No dominant public dataset; simulation only |

**Coverage:** 130/168 (77%) have a verifiable famous public source

---

## Priority Build Order

Build in order of: (1) most-cited / gold-standard benchmarks → (2) established public repos → (3) simulation

### Tier 1 — Build First (most cited / gold standard)
1. `cassi` — KAIST 10 scenes (MST CVPR 2022) — 10 HSI cubes
2. `cacti` — SCI Video 6 scenes (Liu TPAMI 2019) — 6 video clips
3. `ct` — LoDoPaB-CT validation split — 10 sinogram-image pairs
4. `mri` — fastMRI knee multi-coil — 10 k-space volumes
5. `ultrasound` — PICMUS IQ data — 10 RF frames
6. `cryo_em` — EMPIAR-10028 TRPV1 projections — 10 2D projections
7. `ptychography` — CDI Zenodo benchmark — 10 diffraction patterns
8. `fundus` — DRIVE test set — 10 fundus images
9. `oct` — RETOUCH challenge — 10 OCT B-scans
10. `gravitational_wave` — LIGO O3 GWOSC — 10 strain segments

### Tier 2 — Well-known benchmarks
- `fmri` (HCP), `diffusion_mri` (HCP), `pet` (TCIA), `endoscopy` (Kvasir),
  `mammography` (CBIS-DDSM), `insar` (Sentinel-1), `sar` (Sentinel-1),
  `hyperspectral_remote` (AVIRIS), `lidar` (KITTI), `event_camera` (MVSEC),
  `nerf` (NeRF Blender), `gaussian_splatting` (Tanks & Temples)

### Tier 3 — Domain-specific public archives
- `eht_imaging`, `solar_imaging`, `radio_astronomy`, `ocean_color`,
  `electron_tomography` (EMPIAR), `fib_sem` (OpenOrganelle),
  `xray_crystallography` (PDB), `stem`, `tem`, `eels`

---

## GCS Upload Instructions

After building a standard dataset, upload with:
```bash
gsutil -m cp -r datasets/benchmark/{modality}/standard/ \
    gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/
```

Upload all at once (after building all):
```bash
for mod in $(ls datasets/benchmark/ | grep -v '\.md\|\.py'); do
  if [ -d "datasets/benchmark/$mod/standard" ] && [ "$(ls -A datasets/benchmark/$mod/standard)" ]; then
    gsutil -m cp -r "datasets/benchmark/$mod/standard/" \
        "gs://pwm-benchmark-datasets/datasets/Benchmark/$mod/standard/"
    echo "Uploaded: $mod"
  fi
done
```

Verify upload:
```bash
gsutil ls "gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/"
```

---

## .gitignore Note

The `datasets/benchmark/*/standard/` directories are excluded from git (via `datasets/` in .gitignore).
This file (`standard_state.md`) is committed to git as the tracking document.
Dataset files must be downloaded from GCS.

---

*Standard dataset tracking — PWM Benchmark — 2026-03-13*
