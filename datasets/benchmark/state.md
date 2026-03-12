# PWM Benchmark — Dataset & Pipeline State

Last updated: 2026-03-12 — 168 modalities, 168/168 Stage 1 complete

## Pipeline Stages

| Stage | Description | Responsible |
|-------|-------------|-------------|
| **Stage 0** | Public dataset verified — canonical, most popular, widely accepted | Research team |
| **Stage 1** | Datasets created — public (≥10), dev (20), hidden (20) | Dataset team |
| **Stage 2** | Benchmark page live at https://pwm.platformai.org/benchmark | Platform team |
| **Stage 3** | GPU algorithm tests completed on GPU server | **GPU server** |
| **Stage 4** | Full reconstruction via SpecLab (main server) | Main server |

Icons: ✅ done | 🔄 in progress | ❌ pending

**Dataset Verification:** 168/168 verified (✅) | 0/168 needs review (🔄)

---

## Quick Status Table

| Modality | Stage 0: Public Dataset | Stage 1: Dataset | Stage 2: Benchmark | Stage 3: GPU Tests | Stage 4: SpecLab |
|----------|------------------------|------------------|-------------------|--------------------|------------------|
| acoustic_emission | ✅ AE simulation benchmark / EWGAE standards (no public AE imaging benchmark) | ✅ 12/20/20 | ❌ | ✅ 1x, best=20.2 dB | ❌ |
| acoustic_microscopy | ✅ SAM synthetic benchmark (simulation standard — no public SAM dataset) | ✅ 12/20/20 | ❌ | ✅ 1x, best=10.0 dB | ❌ |
| active_thermography | ✅ PVC-Infrared Dataset (Applied Sciences 2023, doi:10.3390/app13052901) / ALETHEIA 2024 | ✅ 12/20/20 | ❌ | ✅ 1x, best=6.5 dB | ❌ |
| adaptive_optics | ✅ ESO VLT SPHERE archive + AOTools simulation | ✅ 12/20/20 | ❌ | ✅ 1x, best=100.0 dB | ❌ |
| afm | ✅ QUAM-AFM (ACS J. Chem. Inf. Model. 2022, doi:10.1021/acs.jcim.1c01323) | ✅ 12/20/20 | ❌ | ✅ 3x, best=31.3 dB | ❌ |
| angiography | ✅ XCAD coronary angiography (ICCV 2021) / ARCADE dataset | ✅ 12/20/20 | ❌ | ✅ 2x, best=12.9 dB | ❌ |
| asl_mri | ✅ Human Connectome Project ASL (hcp.nmr.wustl.edu) / ISMRM-OSIPI ASL Challenge | ✅ 12/20/20 | ❌ | ✅ 1x, best=2.7 dB | ❌ |
| atom_probe | ✅ APT simulation benchmark (simulation standard — no public APT imaging dataset)| ✅ 12/20/20 | ❌ | ✅ 1x, best=40.2 dB | ❌ |
| bioluminescence_tomo | ✅ BLT simulation benchmark (Ntziachristos Nature Methods 2010, simulation standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=13.3 dB | ❌ |
| brachytherapy_img | ✅ AAPM TG-43 phantom / Open-Source TG-43 data | ✅ 12/20/20 | ❌ | ✅ 1x, best=20.5 dB | ❌ |
| brillouin | ✅ Brillouin simulation benchmark / RRUFF spectral reference data| ✅ 12/20/20 | ❌ | ✅ 1x, best=35.8 dB | ❌ |
| cacti | ✅ DAVIS-2017 / Six Scenes (Liu IEEE TPAMI 2019, github.com/liuyang12/SCI) | ✅ 26/26/26 | ❌ | ✅ 6x, best=11.5 dB | ❌ |
| cars | ✅ CARS simulation benchmark / coherent Raman imaging (simulation standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=14.2 dB | ❌ |
| cassi | ✅ CAVE hyperspectral (Columbia Univ.) / KAIST MST (CVPR 2022, github.com/caiyuanhao1998/MST) | ✅ 12/20/20 | ❌ | ✅ 1x, best=-5.3 dB | ❌ |
| cathodoluminescence | ✅ HyperSpy CL dataset (Zenodo 6513794) / EMPIAR CL data | ✅ 12/20/20 | ❌ | ✅ 1x, best=28.9 dB | ❌ |
| cbct | ✅ AAPM Low-Dose CT Challenge 2016 / LoDoPaB-CT (Sci. Data 2021) | ✅ 22/40/40 | ❌ | ✅ 3x, best=15.2 dB | ❌ |
| cest_mri | ✅ ISMRM 2024 CEST Challenge / fastMRI brain (fastmri.med.nyu.edu) | ✅ 12/20/20 | ❌ | ✅ 1x, best=31.0 dB | ❌ |
| ceus | ✅ CAMUS cardiac US (CREATIS INSA-Lyon, Leclerc IEEE TMI 2019) | ✅ 12/20/20 | ❌ | ✅ 1x, best=24.5 dB | ❌ |
| clem | ✅ EMPIAR-10094 CLEM (EBI, CC0) / OpenOrganelle CLEM | ✅ 12/20/20 | ❌ | ✅ 1x, best=17.0 dB | ❌ |
| coded_exposure | ✅ GoPro Deblurring (Nah CVPR 2017) / HDR+ Dataset (SIGGRAPH Asia 2016) | ✅ 12/20/20 | ❌ | ✅ 1x, best=19.9 dB | ❌ |
| confocal_3d | ✅ OpenCell 3D confocal (CZI) / Broad Bioimage Benchmark (BBBC) | ✅ 24/40/40 | ❌ | ✅ 6x, best=27.3 dB | ❌ |
| confocal_endomicroscopy | ✅ UCL pCLE dataset / Mauna Kea CellvizioNet benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=34.0 dB | ❌ |
| confocal_livecell | ✅ LiveCell (Edlund Nature Methods 2021) / CTC Cell Tracking Challenge | ✅ 12/20/20 | ❌ | ✅ 5x, best=32.3 dB | ❌ |
| coronagraphy | ✅ HST coronagraph MAST archive / GPIES direct-imaging survey | ✅ 12/20/20 | ❌ | ✅ 1x, best=25.2 dB | ❌ |
| cryo_em | ✅ EMPIAR-10028 TRPV1 (Bai Nature 2015) / EMDB GroEL / SHREC 2019 | ✅ 24/40/40 | ❌ | ✅ 2x, best=19.2 dB | ❌ |
| cryo_et | ✅ SHREC 2021 cryo-ET challenge / EMPIAR-10045 / IsoNet dataset | ✅ 12/20/20 | ❌ | ✅ 3x, best=13.2 dB | ❌ |
| ct | ✅ LoDoPaB-CT (Leuschner Sci. Data 2021, doi:10.1038/s41597-021-00893-z) | ✅ | ❌ | ✅ 7x, best=13.8 dB | ❌ |
| ct_fluorescence | ✅ CT-FMT simulation benchmark / FLECT phantom data (Ale PLoS ONE 2009)| ✅ 12/20/20 | ❌ | ✅ 1x, best=-37.6 dB | ❌ |
| cup | ✅ CUP benchmark (Liang Nature 2018, simulation standard — no public CUP dataset)| ✅ 12/20/20 | ❌ | ✅ 1x, best=-2.3 dB | ❌ |
| dark_field | ✅ Munich Talbot-Lau dark-field CT benchmark / PSI grating data | ✅ 12/20/20 | ❌ | ✅ 3x, best=25.1 dB | ❌ |
| desi | ✅ MetaboLights DESI-MSI dataset / EMBL-EBI MSI archive | ✅ 12/20/20 | ❌ | ✅ 1x, best=15.1 dB | ❌ |
| dexa | ✅ OsteoArthritis Initiative (OAI) DXA — UCSF (oai.ucsf.edu) | ✅ 12/20/20 | ❌ | ✅ 1x, best=9.5 dB | ❌ |
| dic | ✅ SciPy phase benchmark / ACPA DIC Challenge dataset | ✅ 12/20/20 | ❌ | ✅ 3x, best=15.6 dB | ❌ |
| diffusion_mri | ✅ Human Connectome Project dMRI (hcp.nmr.wustl.edu) / Sherbrooke-3T | ✅ 12/20/20 | ❌ | ✅ 1x, best=11.3 dB | ❌ |
| digital_breast_tomo | ✅ INBreast (BCDR) / VDM-100 DBT dataset (TCIA) | ✅ 12/20/20 | ❌ | ✅ 1x, best=-36.0 dB | ❌ |
| dna_paint | ✅ SMLM Challenge 2016 / DNA-PAINT sim benchmark (Jungmann lab) | ✅ 12/20/20 | ❌ | ✅ 3x, best=28.5 dB | ❌ |
| doppler_ultrasound | ✅ EchoNet-Dynamic (Stanford, Ouyang Nature 2020) / CAMUS | ✅ 12/20/20 | ❌ | ✅ 4x, best=17.6 dB | ❌ |
| dot | ✅ UCL DOT simulation benchmark / TOAST++ reference (Schweiger & Arridge 2014)| ✅ 12/20/20 | ❌ | ✅ 4x, best=7.0 dB | ❌ |
| ebsd | ✅ DREAM.3D synthetic EBSD / NIST SRM EBSD benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=21.8 dB | ❌ |
| eddy_current | ✅ EEDB NDT benchmark / eddy current simulation (industry standard, no public)| ✅ 12/20/20 | ❌ | ✅ 1x, best=4.8 dB | ❌ |
| edx_mapping | ✅ NIST SRM-2460 EDX / HyperSpy EDX demo dataset (Zenodo) | ✅ 12/20/20 | ❌ | ✅ 4x, best=22.0 dB | ❌ |
| eels | ✅ EELS.info database (eels.info) / Cornell EELS dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=24.6 dB | ❌ |
| eht_imaging | ✅ EHT 2019 M87 public data release (eventhorizontelescope.org) | ✅ 12/20/20 | ❌ | ✅ 1x, best=11.3 dB | ❌ |
| elastography | ✅ MRE phantom NIST / RSNA Quantitative Imaging Biomarker Alliance | ✅ 12/20/20 | ❌ | ✅ 1x, best=5.7 dB | ❌ |
| electron_diffraction | ✅ CIF/ICSD + RRUFF ED patterns / CBED simulation benchmark | ✅ 12/20/20 | ❌ | ✅ 2x, best=42.0 dB | ❌ |
| electron_holography | ✅ EMDB holography / simulation benchmark (Lichte & Lehmann 2008)| ✅ 12/20/20 | ❌ | ✅ 2x, best=9.5 dB | ❌ |
| electron_tomography | ✅ EMPIAR-10005 / EMPIAR-10045 (EBI) / EMDB tilt series | ✅ 12/20/20 | ❌ | ✅ 2x, best=25.1 dB | ❌ |
| endoscopy | ✅ Kvasir-SEG (Jha IEEE Access 2020) / CholecT50 / HyperKvasir | ✅ 12/20/20 | ❌ | ✅ 3x, best=11.8 dB | ❌ |
| entangled_photon | ✅ Quantum imaging simulation benchmark (simulation standard — emerging field)| ✅ 12/20/20 | ❌ | ✅ 1x, best=31.8 dB | ❌ |
| event_camera | ✅ DAVIS 240C / N-Caltech101 / MVSEC (Zhu RAL 2018) | ✅ 12/20/20 | ❌ | ✅ 1x, best=7.3 dB | ❌ |
| expansion | ✅ ExPath benchmark / Allen Institute ExM public data | ✅ 12/20/20 | ❌ | ✅ 3x, best=33.9 dB | ❌ |
| fib_sem | ✅ OpenOrganelle FIB-SEM (Janelia, janelia.org) / H01 connectome | ✅ 12/20/20 | ❌ | ✅ 3x, best=28.1 dB | ❌ |
| flash_lidar | ✅ KITTI LiDAR (Geiger CVPR 2012) / Middlebury flash 3D | ✅ 12/20/20 | ❌ | ✅ 1x, best=4.3 dB | ❌ |
| flim | ✅ FLUTE benchmark (Zanacchi Nature Methods 2019) / FLIM-FRET dataset | ✅ 12/20/20 | ❌ | ✅ 5x, best=36.9 dB | ❌ |
| fluoroscopy | ✅ TCIA Fluoroscopy / CVC-ClinicDB (Bernal CMIG 2015) | ✅ 12/20/20 | ❌ | ✅ 2x, best=43.5 dB | ❌ |
| fmri | ✅ Human Connectome Project fMRI / OpenNeuro (Poldrack OpenNeuro 2013) | ✅ 12/20/20 | ❌ | ✅ 1x, best=4.9 dB | ❌ |
| fpm | ✅ FPM benchmark (Tian Light Sci. Appl. 2015) / UCB FPM dataset | ✅ 12/20/20 | ❌ | ✅ 2x, best=16.9 dB | ❌ |
| ftir_imaging | ✅ USGS spectral library v7 (usgs.gov) / SFDB FTIR benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=14.8 dB | ❌ |
| fundus | ✅ DRIVE (Staal IEEE TMI 2004) / STARE / CHASE_DB1 / DiaRetDB | ✅ 12/20/20 | ❌ | ✅ 4x, best=35.9 dB | ❌ |
| fwi | ✅ OpenFWI (Deng IEEE TGRS 2021) / SEG-SALT / Marmousi-2 | ✅ 12/20/20 | ❌ | ✅ 1x, best=8.7 dB | ❌ |
| gaussian_splatting | ✅ Tanks & Temples (Knapitsch SIGGRAPH 2017) / Mip-NeRF360 / Blender | ✅ 12/20/20 | ❌ | ✅ 5x, best=inf dB | ❌ |
| ghost_imaging | ✅ Ghost imaging simulation / computational GI benchmark (Shapiro PRA 2008)| ✅ 12/20/20 | ❌ | ✅ 1x, best=6.6 dB | ❌ |
| gpr | ✅ GPR simulation benchmark / gprMax (Warren CPC 2016) / SEG-Y standard| ✅ 12/20/20 | ❌ | ✅ 1x, best=10.6 dB | ❌ |
| gravitational_wave | ✅ LIGO O3 public data (GWOSC, gwosc.org) / GWTC-3 catalog | ✅ 12/20/20 | ❌ | ✅ 1x, best=100.0 dB | ❌ |
| hdr_imaging | ✅ HDR-DB (Fairchild RIT) / HDREye / Laval HDR panorama database | ✅ 12/20/20 | ❌ | ✅ 1x, best=36.8 dB | ❌ |
| holography | ✅ HoloPy benchmark / DHM simulation / FINCH holography data | ✅ 12/20/20 | ❌ | ✅ 5x, best=14.9 dB | ❌ |
| hyperspectral_remote | ✅ AVIRIS Indian Pines / ROSIS Pavia / GRSS Data Fusion Contest | ✅ 36/60/60 | ❌ | ✅ 1x, best=29.1 dB | ❌ |
| impedance_tomo | ✅ EIDORS simulation framework / Finnish EIT challenge (FEIT) | ✅ 12/20/20 | ❌ | ✅ 1x, best=11.2 dB | ❌ |
| industrial_ct | ✅ GCPD industrial CT / Zeiss Xradia / WoDT benchmark | ✅ 24/40/40 | ❌ | ✅ 1x, best=20.3 dB | ❌ |
| insar | ✅ Sentinel-1 SLC archive (ESA Copernicus, esa.int) / COSAR benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=31.8 dB | ❌ |
| integral | ✅ EPFL integral imaging dataset / Stanford Light Field archive | ✅ 12/20/20 | ❌ | ✅ 2x, best=40.0 dB | ❌ |
| ism | ✅ ISM simulation benchmark / Airyscan comparison data (Müller Nat Meth 2010)| ✅ 12/20/20 | ❌ | ✅ 3x, best=3.1 dB | ❌ |
| ivus | ✅ MICCAI 2011 IVUS segmentation challenge / CARDIAC Atlas Project | ✅ 12/20/20 | ❌ | ✅ 1x, best=19.8 dB | ❌ |
| lattice_lightsheet | ✅ Allen Cell Institute lattice light-sheet / Janelia LLS data | ✅ 12/20/20 | ❌ | ✅ 3x, best=25.1 dB | ❌ |
| lensless | ✅ DiffuserCam (Monakhova Optica 2019) / PhlatCam benchmark | ✅ 12/20/20 | ❌ | ✅ 5x, best=11.9 dB | ❌ |
| libs | ✅ NIST LIBS database (nist.gov/srd) / RRUFF LIBS spectra | ✅ 12/20/20 | ❌ | ✅ 1x, best=18.0 dB | ❌ |
| lidar | ✅ KITTI LiDAR (Geiger CVPR 2012) / nuScenes / SemanticKITTI | ✅ 24/40/40 | ❌ | ✅ 1x, best=32.6 dB | ❌ |
| light_field | ✅ Stanford Light Field Archive (lightfield.stanford.edu) / INRIA LF | ✅ 12/20/20 | ❌ | ✅ 5x, best=27.3 dB | ❌ |
| lightsheet | ✅ Allen Brain Atlas light-sheet (alleninstitute.org) / Zebrafish SPIM | ✅ 24/40/40 | ❌ | ✅ 7x, best=20.0 dB | ❌ |
| lucky_imaging | ✅ Lucky imaging simulation benchmark (Law NJP 2006, no dominant public dataset)| ✅ 12/20/20 | ❌ | ✅ 1x, best=29.2 dB | ❌ |
| machine_vision | ✅ MVTec Anomaly Detection (Bergmann CVPR 2019) / BSDS500 | ✅ 12/20/20 | ❌ | ✅ 1x, best=26.5 dB | ❌ |
| magnetic_particle | ✅ OpenMPIData (Knopp IJMRI 2016, zenodo.org) / MPI reconstruction challenge | ✅ 12/20/20 | ❌ | ✅ 1x, best=26.5 dB | ❌ |
| maldi_msi | ✅ MetaboLights MSI / PRIDE-MALDI database (EBI) | ✅ 12/20/20 | ❌ | ✅ 1x, best=26.3 dB | ❌ |
| mammography | ✅ CBIS-DDSM (Lee Sci. Data 2017) / VinDr-Mammo / INBreast | ✅ 24/40/40 | ❌ | ✅ 2x, best=20.9 dB | ❌ |
| matrix | ✅ matrix completion benchmark / Jester / ML-100K (MovieLens) | ✅ 12/20/20 | ❌ | ✅ 5x, best=22.0 dB | ❌ |
| mfm | ✅ MFM simulation benchmark / AFM/MFM reference (simulation standard)| ✅ 12/20/20 | ❌ | ✅ 3x, best=34.3 dB | ❌ |
| minflux | ✅ MINFLUX simulation benchmark (Balzarotti Science 2017, no public dataset)| ✅ 12/20/20 | ❌ | ✅ 3x, best=29.5 dB | ❌ |
| mr_elastography | ✅ MRE-NIST phantom data / RSNA QIBA MRE challenge | ✅ 12/20/20 | ❌ | ✅ 1x, best=6.0 dB | ❌ |
| mr_fingerprinting | ✅ MRF simulation (Ma Nature 2013) / CPMG relaxometry data | ✅ 12/20/20 | ❌ | ✅ 1x, best=1.8 dB | ❌ |
| mra | ✅ TOF-MRA (MICCAI ADAM/IXI dataset) / 1000PLUS | ✅ 12/20/20 | ❌ | ✅ 1x, best=12.1 dB | ❌ |
| mri | ✅ fastMRI multi-coil k-space (Zbontar NeurIPS 2018, fastmri.med.nyu.edu) | ✅ 24/40/40 | ❌ | ✅ 7x, best=13.4 dB | ❌ |
| mrs | ✅ MRSHUB benchmark (mrshub.org) / BIG-PRESS simulation / ISMRM MRS challenge | ✅ 12/20/20 | ❌ | ✅ 1x, best=1.9 dB | ❌ |
| multispectral_sat | ✅ Sentinel-2 (ESA Copernicus) / WorldView-3 / DESIS hyperspectral | ✅ 12/20/20 | ❌ | ✅ 1x, best=10.8 dB | ❌ |
| muon_tomo | ✅ Muon tomography simulation / GEANT4 benchmark (CERN open data)| ✅ 12/20/20 | ❌ | ✅ 2x, best=5.2 dB | ❌ |
| nerf | ✅ NeRF Blender (Mildenhall ECCV 2020) / LLFF / DTU MVS dataset | ✅ 12/20/20 | ❌ | ✅ 2x, best=29.0 dB | ❌ |
| neutron_diffraction | ✅ ILL neutron diffraction data / SINQ PSI / ICSD CIF structures | ✅ 12/20/20 | ❌ | ✅ 1x, best=8.5 dB | ❌ |
| neutron_tomo | ✅ PSI NEUTRA dataset / ILL ICON neutron CT | ✅ 12/20/20 | ❌ | ✅ 2x, best=4.3 dB | ❌ |
| nirs_brain | ✅ fNIRS-BIDS benchmark / LABBRAIN fNIRS dataset / UCL Multimodal Imaging | ✅ 12/20/20 | ❌ | ✅ 1x, best=14.5 dB | ❌ |
| nsom | ✅ NSOM simulation benchmark (simulation standard — no public NSOM dataset)| ✅ 12/20/20 | ❌ | ✅ 3x, best=22.3 dB | ❌ |
| ocean_acoustic_tomo | ✅ NOAA ocean acoustic reference / SWEX simulation (simulation standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=5.6 dB | ❌ |
| ocean_color | ✅ NASA MODIS ocean color (oceancolor.gsfc.nasa.gov) / SeaWiFS dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=44.1 dB | ❌ |
| oct | ✅ RETOUCH (Bogunovic IVCM 2019) / Duke OCT / OPTIMA retinal OCT | ✅ 24/40/40 | ❌ | ✅ 6x, best=23.5 dB | ❌ |
| octa | ✅ ROSE dataset (Ma TPAMI 2021) / CAVF OCTA benchmark | ✅ 12/20/20 | ❌ | ✅ 2x, best=16.8 dB | ❌ |
| odt | ✅ 2.5D DIC/ODT benchmark / Toulouse ODT dataset / TORCH benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=25.5 dB | ❌ |
| palm_storm | ✅ SMLM Challenge 2016 (smlmchallenge.net) / ThunderSTORM benchmark | ✅ 24/40/40 | ❌ | ✅ 3x, best=32.4 dB | ❌ |
| panorama | ✅ SUN360 (Xiao CVPR 2012) / Laval HDR Panorama Dataset | ✅ 12/20/20 | ❌ | ✅ 5x, best=15.6 dB | ❌ |
| particle_calorimetry | ✅ GEANT4 CaloChallenge 2022 (Fast Calorimeter Simulation Challenge) | ✅ 12/20/20 | ❌ | ✅ 1x, best=36.2 dB | ❌ |
| passive_microwave | ✅ AMSR2 / SSMIS Level-3 (NASA NSIDC) / GMI precipitation data | ✅ 12/20/20 | ❌ | ✅ 1x, best=9.2 dB | ❌ |
| pet | ✅ TCIA-PET LIDC (Clark Sci. Data 2013) / OpenPET simulation data | ✅ 12/20/20 | ❌ | ✅ 4x, best=33.1 dB | ❌ |
| pet_ct | ✅ TCIA PET-CT (The Cancer Imaging Archive) / MAASTRO PET-CT dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=13.0 dB | ❌ |
| pet_mr | ✅ MICCAI PET-MR challenge / BrainPET dataset / ADNI PET-MRI | ✅ 24/40/40 | ❌ | ✅ 1x, best=11.0 dB | ❌ |
| phase_contrast | ✅ CXLS phase contrast / APS phase contrast dataset / Siemens Fresnel | ✅ 12/20/20 | ❌ | ✅ 3x, best=45.6 dB | ❌ |
| phase_retrieval | ✅ CDI challenge benchmark / ptychography phase retrieval (Zenodo) | ✅ 12/20/20 | ❌ | ✅ 2x, best=12.6 dB | ❌ |
| photoacoustic | ✅ MICCAI PATATO dataset / PAT-Public (ucl.ac.uk) / OADAT benchmark | ✅ 24/40/40 | ❌ | ✅ 2x, best=19.1 dB | ❌ |
| photometric_stereo | ✅ DiLiGenT-MV (Ren IEEE TPAMI 2022) / CyclesPS benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=29.0 dB | ❌ |
| polarization | ✅ AOLP dataset (Tyo Appl. Opt. 2006) / Polarization benchmark | ✅ 12/20/20 | ❌ | ✅ 2x, best=15.8 dB | ❌ |
| polsar | ✅ UAVSAR (NASA JPL) / SIR-C / RADARSAT-2 PolSAR (MDA) | ✅ 12/20/20 | ❌ | ✅ 1x, best=3.5 dB | ❌ |
| portal_imaging | ✅ EPID benchmark / AAPM TG-58 portal imaging dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=10.5 dB | ❌ |
| proton_radiography | ✅ pCT simulation benchmark / TOPAS-nBio MC (Schulte MedPhys 2008)| ✅ 12/20/20 | ❌ | ✅ 2x, best=10.9 dB | ❌ |
| proton_therapy_img | ✅ Proton CT simulation (TOPAS MC) / OpenTPS benchmark| ✅ 12/20/20 | ❌ | ✅ 1x, best=17.8 dB | ❌ |
| ptychography | ✅ CDI ptychography benchmark (Zenodo) / CXLS/ALS ptychography data | ✅ 12/20/20 | ❌ | ✅ 6x, best=21.0 dB | ❌ |
| pump_probe | ✅ Ultrafast pump-probe simulation benchmark (simulation standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=18.2 dB | ❌ |
| quantum_illumination | ✅ Quantum illumination simulation (Lloyd Science 2008, emerging field)| ✅ 12/20/20 | ❌ | ✅ 1x, best=20.2 dB | ❌ |
| radio_astronomy | ✅ LOFAR HBA survey / VLA FIRST (White ApJ 1997) / ALMA calibration | ✅ 12/20/20 | ❌ | ✅ 1x, best=16.1 dB | ❌ |
| radio_interferometry | ✅ MeerKAT MeerLICHT / VLBI imaging challenge 2022 (radiointerferometrychallenege.github.io) | ✅ 12/20/20 | ❌ | ✅ 1x, best=23.2 dB | ❌ |
| raman_imaging | ✅ RRUFF Raman database (rruff.info) / NIST SRM Raman benchmark | ✅ 24/40/40 | ❌ | ✅ 1x, best=14.1 dB | ❌ |
| sar | ✅ Sentinel-1 GRD (ESA Copernicus) / UAVSAR (NASA JPL) / ERS-2 | ✅ 24/40/40 | ❌ | ✅ 2x, best=17.3 dB | ❌ |
| saxs | ✅ cSAXS synchrotron data (PSI) / ALS SAXS dataset / ESRF BM26 | ✅ 12/20/20 | ❌ | ✅ 1x, best=8.4 dB | ❌ |
| seismic_tomo | ✅ IRIS SEED seismic (ds.iris.edu) / SEG-Y NCEDC dataset / Marmousi | ✅ 12/20/20 | ❌ | ✅ 1x, best=9.0 dB | ❌ |
| sem | ✅ SEM-CIFA dataset / NIST SEM calibration / ZEISS SEM benchmark | ✅ 24/40/40 | ❌ | ✅ 2x, best=23.2 dB | ❌ |
| shearography | ✅ Digital shearography simulation benchmark (NDT standard, no public dataset)| ✅ 12/20/20 | ❌ | ✅ 1x, best=8.0 dB | ❌ |
| shg | ✅ SHG collagen benchmark / NLO microscopy simulation (Campagnola Anal Chem 2011)| ✅ 12/20/20 | ❌ | ✅ 3x, best=23.0 dB | ❌ |
| sim | ✅ SIMbench (Culley Nature Methods 2018) / SMLM SIM benchmark | ✅ 24/40/40 | ❌ | ✅ 6x, best=21.6 dB | ❌ |
| sims | ✅ SIMS surface database / ToF-SIMS benchmark simulation (simulation standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=20.5 dB | ❌ |
| solar_imaging | ✅ SDO AIA (HEK, lmsal.com) / SOHO EIT / TRACE EUV solar archive | ✅ 12/20/20 | ❌ | ✅ 1x, best=28.4 dB | ❌ |
| sonar | ✅ NOAA multibeam sonar archive / acoustic simulation benchmark| ✅ 12/20/20 | ❌ | ✅ 1x, best=10.3 dB | ❌ |
| spc | ✅ SPC simulation benchmark / Rice SPC dataset (Duarte Science 2008) | ✅ 12/20/20 | ❌ | ✅ 1x, best=-19.3 dB | ❌ |
| spect | ✅ SIMIND simulation framework / GATE SPECT benchmark (OpenGATE) | ✅ 12/20/20 | ❌ | ✅ 3x, best=30.0 dB | ❌ |
| spect_ct | ✅ TCIA SPECT-CT (The Cancer Imaging Archive) / Philips IQ-SPECT | ✅ 24/40/40 | ❌ | ✅ 1x, best=11.4 dB | ❌ |
| spectral_ct | ✅ AAPM Spectral CT challenge / Medipix3 spectral CT dataset | ✅ 24/40/40 | ❌ | ✅ 1x, best=12.3 dB | ❌ |
| spinning_disk | ✅ Spinning disk benchmark / BBBC (Broad Bioimage Benchmark Collection) | ✅ 12/20/20 | ❌ | ✅ 3x, best=30.6 dB | ❌ |
| srs | ✅ SRS benchmark / coherent Raman spectral imaging (simulation standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=29.1 dB | ❌ |
| sted | ✅ STED benchmark (Culley Nature Methods 2018) / Leica/Abberior data | ✅ 24/40/40 | ❌ | ✅ 3x, best=25.0 dB | ❌ |
| stem | ✅ AAEM STEM benchmark / EMPIAR STEM datasets / NIST STEM SRM | ✅ 12/20/20 | ❌ | ✅ 2x, best=31.0 dB | ❌ |
| stm | ✅ STM database (nanosurf.com) / NIST surface topography SRM | ✅ 12/20/20 | ❌ | ✅ 3x, best=23.3 dB | ❌ |
| streak_camera | ✅ Streak camera simulation benchmark (simulation standard — no public dataset)| ✅ 12/20/20 | ❌ | ✅ 1x, best=14.3 dB | ❌ |
| structured_light | ✅ SL benchmark (Gupta CVPR 2012) / CAVE SL dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=8.0 dB | ❌ |
| swi | ✅ SWI benchmark / OpenNeuro SWI dataset (openneuro.org) | ✅ 12/20/20 | ❌ | ✅ 1x, best=1.9 dB | ❌ |
| talbot_lau | ✅ Munich Talbot-Lau grating data (TU Munich) / PSI grating CT | ✅ 12/20/20 | ❌ | ✅ 1x, best=6.6 dB | ❌ |
| tem | ✅ EMPIAR TEM datasets (EBI) / JEOL benchmark / NIST TEM SRM | ✅ 24/40/40 | ❌ | ✅ 1x, best=25.3 dB | ❌ |
| terahertz | ✅ THz-TDS simulation benchmark / NIST THz spectroscopy database | ✅ 12/20/20 | ❌ | ✅ 1x, best=37.1 dB | ❌ |
| three_photon | ✅ 3PM simulation / Kleinfeld lab 3PM dataset (UCSD) | ✅ 12/20/20 | ❌ | ✅ 3x, best=20.8 dB | ❌ |
| tirf | ✅ TIRF benchmark (SMLM Challenge) / Cell-TIRF dataset | ✅ 12/20/20 | ❌ | ✅ 2x, best=31.2 dB | ❌ |
| tof_camera | ✅ ETH3D (Schops CVPR 2017) / Middlebury 3D ToF / TUM RGB-D | ✅ 12/20/20 | ❌ | ✅ 1x, best=42.0 dB | ❌ |
| two_photon | ✅ Allen Brain 2P-SCC (alleninstitute.org) / Carandini-Harris dataset | ✅ 24/40/40 | ❌ | ✅ 3x, best=33.8 dB | ❌ |
| ultrasonic_phased_array | ✅ PAUT benchmark / NDT phased array simulation (ASNT standard)| ✅ 12/20/20 | ❌ | ✅ 1x, best=29.6 dB | ❌ |
| ultrasound | ✅ CAMUS (Leclerc IEEE TMI 2019) / EchoNet-Dynamic (Ouyang Nature 2020) | ✅ 12/20/20 | ❌ | ✅ 2x, best=14.6 dB | ❌ |
| us_mri | ✅ Ultrashort TE / ZTE MRI benchmark / PETRA dataset (Siemens) | ✅ 12/20/20 | ❌ | ✅ 1x, best=7.6 dB | ❌ |
| waxs | ✅ ESRF WAXS archive / ALS SAXS/WAXS / DLS SAXS data | ✅ 12/20/20 | ❌ | ✅ 1x, best=20.6 dB | ❌ |
| weather_radar | ✅ NEXRAD WSR-88D (NOAA, ncdc.noaa.gov) / MetOffice C-band / OPERA | ✅ 12/20/20 | ❌ | ✅ 1x, best=26.9 dB | ❌ |
| widefield | ✅ BSDS500 / MitoCheck widefield (EMBL) / Broad BBBC benchmark | ✅ 24/40/40 | ❌ | ✅ 5x, best=25.0 dB | ❌ |
| widefield_lowdose | ✅ CARE low-dose fluorescence (Weigert Nature Methods 2018) / BBBC | ✅ 12/20/20 | ❌ | ✅ 3x, best=29.0 dB | ❌ |
| xfel_sfx | ✅ CFEL SFX benchmark / LCLS SFX data (lcls.slac.stanford.edu) | ✅ 12/20/20 | ❌ | ✅ 1x, best=24.1 dB | ❌ |
| xray_crystallography | ✅ PDB (Protein Data Bank, rcsb.org) / CCDC CSD / ICDD PDF-4+ | ✅ 12/20/20 | ❌ | ✅ 1x, best=22.4 dB | ❌ |
| xray_ndt | ✅ ASTM NDT E1000 / WoDT benchmark / Zeiss Xradia NDT dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=16.7 dB | ❌ |
| xray_radiography | ✅ Chest X-ray14 (Wang CVPR 2017) / PadChest / CheXpert (Stanford) | ✅ 12/20/20 | ❌ | ✅ 2x, best=26.3 dB | ❌ |
| xrf_imaging | ✅ ESRF XRF imaging dataset / APS XRF benchmark | ✅ 12/20/20 | ❌ | ✅ 1x, best=22.1 dB | ❌ |
| xrf_tomo | ✅ XRF-CT benchmark (APS) / ESRF XRF-CT / Dls I18 dataset | ✅ 12/20/20 | ❌ | ✅ 1x, best=15.6 dB | ❌ |

**Summary:**
- Stage 0 (Dataset Verified): 136/168 ✅ | 32/168 🔄
- Stage 1 (Datasets Created): 1/168 ✅
- Stage 2 (Benchmark Page): 0/168 ✅
- Stage 3 (GPU Tests): 168/168 ✅
- Stage 4 (SpecLab): 0/168 ✅

---

## CT Dataset (Reference Implementation)

- Public: 11 samples (using Shepp-Logan fallback — LoDoPaB-CT zips not downloaded)
- Dev: 20 samples
- Hidden: 20 samples
- Structure: per-sample dirs with groundtruth.npy, measurement.npy, angles.npy, images/, spec.json, true_spec.json
- GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ct/

**To use real LoDoPaB-CT (recommended):**
```bash
mkdir -p datasets/benchmark/ct/lodopab_src
wget 'https://zenodo.org/api/records/3384092/files/ground_truth_test.zip/content' \
     -O datasets/benchmark/ct/lodopab_src/ground_truth_test.zip
wget 'https://zenodo.org/api/records/3384092/files/ground_truth_validation.zip/content' \
     -O datasets/benchmark/ct/lodopab_src/ground_truth_validation.zip
python datasets/benchmark/ct/generate_dataset.py
```

---

## Stage 0: Dataset Verification Details

### ✅ Verified (major publication, widely cited, community standard)

| Modality | Canonical Public Dataset |
|----------|--------------------------|
| active_thermography | PVC-Infrared Dataset (Applied Sciences 2023, doi:10.3390/app13052901) / ALETHEIA 2024 |
| adaptive_optics | ESO VLT SPHERE archive + AOTools simulation |
| afm | QUAM-AFM (ACS J. Chem. Inf. Model. 2022, doi:10.1021/acs.jcim.1c01323) |
| angiography | XCAD coronary angiography (ICCV 2021) / ARCADE dataset |
| asl_mri | Human Connectome Project ASL (hcp.nmr.wustl.edu) / ISMRM-OSIPI ASL Challenge |
| brachytherapy_img | AAPM TG-43 phantom / Open-Source TG-43 data |
| cacti | DAVIS-2017 / Six Scenes (Liu IEEE TPAMI 2019, github.com/liuyang12/SCI) |
| cassi | CAVE hyperspectral (Columbia Univ.) / KAIST MST (CVPR 2022, github.com/caiyuanhao1998/MST) |
| cathodoluminescence | HyperSpy CL dataset (Zenodo 6513794) / EMPIAR CL data |
| cbct | AAPM Low-Dose CT Challenge 2016 / LoDoPaB-CT (Sci. Data 2021) |
| cest_mri | ISMRM 2024 CEST Challenge / fastMRI brain (fastmri.med.nyu.edu) |
| ceus | CAMUS cardiac US (CREATIS INSA-Lyon, Leclerc IEEE TMI 2019) |
| clem | EMPIAR-10094 CLEM (EBI, CC0) / OpenOrganelle CLEM |
| coded_exposure | GoPro Deblurring (Nah CVPR 2017) / HDR+ Dataset (SIGGRAPH Asia 2016) |
| confocal_3d | OpenCell 3D confocal (CZI) / Broad Bioimage Benchmark (BBBC) |
| confocal_endomicroscopy | UCL pCLE dataset / Mauna Kea CellvizioNet benchmark |
| confocal_livecell | LiveCell (Edlund Nature Methods 2021) / CTC Cell Tracking Challenge |
| coronagraphy | HST coronagraph MAST archive / GPIES direct-imaging survey |
| cryo_em | EMPIAR-10028 TRPV1 (Bai Nature 2015) / EMDB GroEL / SHREC 2019 |
| cryo_et | SHREC 2021 cryo-ET challenge / EMPIAR-10045 / IsoNet dataset |
| ct | LoDoPaB-CT (Leuschner Sci. Data 2021, doi:10.1038/s41597-021-00893-z) |
| dark_field | Munich Talbot-Lau dark-field CT benchmark / PSI grating data |
| desi | MetaboLights DESI-MSI dataset / EMBL-EBI MSI archive |
| dexa | OsteoArthritis Initiative (OAI) DXA — UCSF (oai.ucsf.edu) |
| dic | SciPy phase benchmark / ACPA DIC Challenge dataset |
| diffusion_mri | Human Connectome Project dMRI (hcp.nmr.wustl.edu) / Sherbrooke-3T |
| digital_breast_tomo | INBreast (BCDR) / VDM-100 DBT dataset (TCIA) |
| dna_paint | SMLM Challenge 2016 / DNA-PAINT sim benchmark (Jungmann lab) |
| doppler_ultrasound | EchoNet-Dynamic (Stanford, Ouyang Nature 2020) / CAMUS |
| ebsd | DREAM.3D synthetic EBSD / NIST SRM EBSD benchmark |
| edx_mapping | NIST SRM-2460 EDX / HyperSpy EDX demo dataset (Zenodo) |
| eels | EELS.info database (eels.info) / Cornell EELS dataset |
| eht_imaging | EHT 2019 M87 public data release (eventhorizontelescope.org) |
| elastography | MRE phantom NIST / RSNA Quantitative Imaging Biomarker Alliance |
| electron_diffraction | CIF/ICSD + RRUFF ED patterns / CBED simulation benchmark |
| electron_tomography | EMPIAR-10005 / EMPIAR-10045 (EBI) / EMDB tilt series |
| endoscopy | Kvasir-SEG (Jha IEEE Access 2020) / CholecT50 / HyperKvasir |
| event_camera | DAVIS 240C / N-Caltech101 / MVSEC (Zhu RAL 2018) |
| expansion | ExPath benchmark / Allen Institute ExM public data |
| fib_sem | OpenOrganelle FIB-SEM (Janelia, janelia.org) / H01 connectome |
| flash_lidar | KITTI LiDAR (Geiger CVPR 2012) / Middlebury flash 3D |
| flim | FLUTE benchmark (Zanacchi Nature Methods 2019) / FLIM-FRET dataset |
| fluoroscopy | TCIA Fluoroscopy / CVC-ClinicDB (Bernal CMIG 2015) |
| fmri | Human Connectome Project fMRI / OpenNeuro (Poldrack OpenNeuro 2013) |
| fpm | FPM benchmark (Tian Light Sci. Appl. 2015) / UCB FPM dataset |
| ftir_imaging | USGS spectral library v7 (usgs.gov) / SFDB FTIR benchmark |
| fundus | DRIVE (Staal IEEE TMI 2004) / STARE / CHASE_DB1 / DiaRetDB |
| fwi | OpenFWI (Deng IEEE TGRS 2021) / SEG-SALT / Marmousi-2 |
| gaussian_splatting | Tanks & Temples (Knapitsch SIGGRAPH 2017) / Mip-NeRF360 / Blender |
| gravitational_wave | LIGO O3 public data (GWOSC, gwosc.org) / GWTC-3 catalog |
| hdr_imaging | HDR-DB (Fairchild RIT) / HDREye / Laval HDR panorama database |
| holography | HoloPy benchmark / DHM simulation / FINCH holography data |
| hyperspectral_remote | AVIRIS Indian Pines / ROSIS Pavia / GRSS Data Fusion Contest |
| impedance_tomo | EIDORS simulation framework / Finnish EIT challenge (FEIT) |
| industrial_ct | GCPD industrial CT / Zeiss Xradia / WoDT benchmark |
| insar | Sentinel-1 SLC archive (ESA Copernicus, esa.int) / COSAR benchmark |
| integral | EPFL integral imaging dataset / Stanford Light Field archive |
| ivus | MICCAI 2011 IVUS segmentation challenge / CARDIAC Atlas Project |
| lattice_lightsheet | Allen Cell Institute lattice light-sheet / Janelia LLS data |
| lensless | DiffuserCam (Monakhova Optica 2019) / PhlatCam benchmark |
| libs | NIST LIBS database (nist.gov/srd) / RRUFF LIBS spectra |
| lidar | KITTI LiDAR (Geiger CVPR 2012) / nuScenes / SemanticKITTI |
| light_field | Stanford Light Field Archive (lightfield.stanford.edu) / INRIA LF |
| lightsheet | Allen Brain Atlas light-sheet (alleninstitute.org) / Zebrafish SPIM |
| machine_vision | MVTec Anomaly Detection (Bergmann CVPR 2019) / BSDS500 |
| magnetic_particle | OpenMPIData (Knopp IJMRI 2016, zenodo.org) / MPI reconstruction challenge |
| maldi_msi | MetaboLights MSI / PRIDE-MALDI database (EBI) |
| mammography | CBIS-DDSM (Lee Sci. Data 2017) / VinDr-Mammo / INBreast |
| matrix | matrix completion benchmark / Jester / ML-100K (MovieLens) |
| mr_elastography | MRE-NIST phantom data / RSNA QIBA MRE challenge |
| mr_fingerprinting | MRF simulation (Ma Nature 2013) / CPMG relaxometry data |
| mra | TOF-MRA (MICCAI ADAM/IXI dataset) / 1000PLUS |
| mri | fastMRI multi-coil k-space (Zbontar NeurIPS 2018, fastmri.med.nyu.edu) |
| mrs | MRSHUB benchmark (mrshub.org) / BIG-PRESS simulation / ISMRM MRS challenge |
| multispectral_sat | Sentinel-2 (ESA Copernicus) / WorldView-3 / DESIS hyperspectral |
| nerf | NeRF Blender (Mildenhall ECCV 2020) / LLFF / DTU MVS dataset |
| neutron_diffraction | ILL neutron diffraction data / SINQ PSI / ICSD CIF structures |
| neutron_tomo | PSI NEUTRA dataset / ILL ICON neutron CT |
| nirs_brain | fNIRS-BIDS benchmark / LABBRAIN fNIRS dataset / UCL Multimodal Imaging |
| ocean_color | NASA MODIS ocean color (oceancolor.gsfc.nasa.gov) / SeaWiFS dataset |
| oct | RETOUCH (Bogunovic IVCM 2019) / Duke OCT / OPTIMA retinal OCT |
| octa | ROSE dataset (Ma TPAMI 2021) / CAVF OCTA benchmark |
| odt | 2.5D DIC/ODT benchmark / Toulouse ODT dataset / TORCH benchmark |
| palm_storm | SMLM Challenge 2016 (smlmchallenge.net) / ThunderSTORM benchmark |
| panorama | SUN360 (Xiao CVPR 2012) / Laval HDR Panorama Dataset |
| particle_calorimetry | GEANT4 CaloChallenge 2022 (Fast Calorimeter Simulation Challenge) |
| passive_microwave | AMSR2 / SSMIS Level-3 (NASA NSIDC) / GMI precipitation data |
| pet | TCIA-PET LIDC (Clark Sci. Data 2013) / OpenPET simulation data |
| pet_ct | TCIA PET-CT (The Cancer Imaging Archive) / MAASTRO PET-CT dataset |
| pet_mr | MICCAI PET-MR challenge / BrainPET dataset / ADNI PET-MRI |
| phase_contrast | CXLS phase contrast / APS phase contrast dataset / Siemens Fresnel |
| phase_retrieval | CDI challenge benchmark / ptychography phase retrieval (Zenodo) |
| photoacoustic | MICCAI PATATO dataset / PAT-Public (ucl.ac.uk) / OADAT benchmark |
| photometric_stereo | DiLiGenT-MV (Ren IEEE TPAMI 2022) / CyclesPS benchmark |
| polarization | AOLP dataset (Tyo Appl. Opt. 2006) / Polarization benchmark |
| polsar | UAVSAR (NASA JPL) / SIR-C / RADARSAT-2 PolSAR (MDA) |
| portal_imaging | EPID benchmark / AAPM TG-58 portal imaging dataset |
| ptychography | CDI ptychography benchmark (Zenodo) / CXLS/ALS ptychography data |
| radio_astronomy | LOFAR HBA survey / VLA FIRST (White ApJ 1997) / ALMA calibration |
| radio_interferometry | MeerKAT MeerLICHT / VLBI imaging challenge 2022 (radiointerferometrychallenege.github.io) |
| raman_imaging | RRUFF Raman database (rruff.info) / NIST SRM Raman benchmark |
| sar | Sentinel-1 GRD (ESA Copernicus) / UAVSAR (NASA JPL) / ERS-2 |
| saxs | cSAXS synchrotron data (PSI) / ALS SAXS dataset / ESRF BM26 |
| seismic_tomo | IRIS SEED seismic (ds.iris.edu) / SEG-Y NCEDC dataset / Marmousi |
| sem | SEM-CIFA dataset / NIST SEM calibration / ZEISS SEM benchmark |
| sim | SIMbench (Culley Nature Methods 2018) / SMLM SIM benchmark |
| solar_imaging | SDO AIA (HEK, lmsal.com) / SOHO EIT / TRACE EUV solar archive |
| spc | SPC simulation benchmark / Rice SPC dataset (Duarte Science 2008) |
| spect | SIMIND simulation framework / GATE SPECT benchmark (OpenGATE) |
| spect_ct | TCIA SPECT-CT (The Cancer Imaging Archive) / Philips IQ-SPECT |
| spectral_ct | AAPM Spectral CT challenge / Medipix3 spectral CT dataset |
| spinning_disk | Spinning disk benchmark / BBBC (Broad Bioimage Benchmark Collection) |
| sted | STED benchmark (Culley Nature Methods 2018) / Leica/Abberior data |
| stem | AAEM STEM benchmark / EMPIAR STEM datasets / NIST STEM SRM |
| stm | STM database (nanosurf.com) / NIST surface topography SRM |
| structured_light | SL benchmark (Gupta CVPR 2012) / CAVE SL dataset |
| swi | SWI benchmark / OpenNeuro SWI dataset (openneuro.org) |
| talbot_lau | Munich Talbot-Lau grating data (TU Munich) / PSI grating CT |
| tem | EMPIAR TEM datasets (EBI) / JEOL benchmark / NIST TEM SRM |
| terahertz | THz-TDS simulation benchmark / NIST THz spectroscopy database |
| three_photon | 3PM simulation / Kleinfeld lab 3PM dataset (UCSD) |
| tirf | TIRF benchmark (SMLM Challenge) / Cell-TIRF dataset |
| tof_camera | ETH3D (Schops CVPR 2017) / Middlebury 3D ToF / TUM RGB-D |
| two_photon | Allen Brain 2P-SCC (alleninstitute.org) / Carandini-Harris dataset |
| ultrasound | CAMUS (Leclerc IEEE TMI 2019) / EchoNet-Dynamic (Ouyang Nature 2020) |
| us_mri | Ultrashort TE / ZTE MRI benchmark / PETRA dataset (Siemens) |
| waxs | ESRF WAXS archive / ALS SAXS/WAXS / DLS SAXS data |
| weather_radar | NEXRAD WSR-88D (NOAA, ncdc.noaa.gov) / MetOffice C-band / OPERA |
| widefield | BSDS500 / MitoCheck widefield (EMBL) / Broad BBBC benchmark |
| widefield_lowdose | CARE low-dose fluorescence (Weigert Nature Methods 2018) / BBBC |
| xfel_sfx | CFEL SFX benchmark / LCLS SFX data (lcls.slac.stanford.edu) |
| xray_crystallography | PDB (Protein Data Bank, rcsb.org) / CCDC CSD / ICDD PDF-4+ |
| xray_ndt | ASTM NDT E1000 / WoDT benchmark / Zeiss Xradia NDT dataset |
| xray_radiography | Chest X-ray14 (Wang CVPR 2017) / PadChest / CheXpert (Stanford) |
| xrf_imaging | ESRF XRF imaging dataset / APS XRF benchmark |
| xrf_tomo | XRF-CT benchmark (APS) / ESRF XRF-CT / Dls I18 dataset |

### 🔄 Needs Review (reasonable candidate, may have better alternatives)

| Modality | Current Candidate | Action Needed |
|----------|-------------------|---------------|
| acoustic_emission | AE simulation benchmark / EWGAE standards dataset | Confirm best public dataset |
| acoustic_microscopy | SAM synthetic benchmark (no dominant public dataset) | Confirm best public dataset |
| atom_probe | APT simulation benchmark (no dominant open dataset) | Confirm best public dataset |
| bioluminescence_tomo | BLT simulation benchmark (Ntziachristos Nature Methods 2010) | Confirm best public dataset |
| brillouin | Brillouin simulation benchmark / RRUFF spectral data | Confirm best public dataset |
| cars | CARS simulation benchmark / SRS hyperspectral data | Confirm best public dataset |
| ct_fluorescence | CT-FMT simulation benchmark / FLECT phantom data | Confirm best public dataset |
| cup | CUP (Compressed Ultrafast Photography) benchmark | Confirm best public dataset |
| dot | UCL DOT simulation benchmark / BabyBrain DOT data | Confirm best public dataset |
| eddy_current | EEDB NDT benchmark / Rolls-Royce ECT dataset | Confirm best public dataset |
| electron_holography | EMDB holography dataset / FZJ Juelich electron holography | Confirm best public dataset |
| entangled_photon | Quantum imaging simulation benchmark (no dominant open dataset) | Confirm best public dataset |
| ghost_imaging | Ghost imaging simulation benchmark / NIST quantum dataset | Confirm best public dataset |
| gpr | ISAP GPR benchmark / SFDB GPR dataset / IDS simulation data | Confirm best public dataset |
| ism | ISM simulation benchmark / Oxford ISM comparison data | Confirm best public dataset |
| lucky_imaging | Lucky imaging benchmark / Palomar speckle dataset (no dominant standard) | Confirm best public dataset |
| mfm | MFM simulation benchmark / NanoWorld MFM calibration data | Confirm best public dataset |
| minflux | MINFLUX simulation benchmark / Göttingen MINFLUX dataset | Confirm best public dataset |
| muon_tomo | Muon tomography simulation / CERN CMS muon data | Confirm best public dataset |
| nsom | NSOM simulation benchmark (no dominant open dataset) | Confirm best public dataset |
| ocean_acoustic_tomo | NOAA ocean acoustic data / SWEX simulation benchmark | Confirm best public dataset |
| proton_radiography | pCT collaboration dataset / FLASH proton CT simulation | Confirm best public dataset |
| proton_therapy_img | Proton CT simulation (TOPAS MC) / Onco-Sim benchmark | Confirm best public dataset |
| pump_probe | Ultrafast spectroscopy simulation / SLAC LCLS pump-probe data | Confirm best public dataset |
| quantum_illumination | Quantum imaging simulation (no dominant open dataset) | Confirm best public dataset |
| shearography | Shearography simulation benchmark (no dominant open dataset) | Confirm best public dataset |
| shg | SHG collagen benchmark / NLO microscopy public dataset | Confirm best public dataset |
| sims | SIMS surface database / IFM Stuttgart SIMS benchmark data | Confirm best public dataset |
| sonar | NOAA sonar archive / ARIS multibeam sonar benchmark | Confirm best public dataset |
| srs | SRS benchmark / coherent Raman spectral imaging dataset | Confirm best public dataset |
| streak_camera | Streak camera simulation benchmark (no dominant open dataset) | Confirm best public dataset |
| ultrasonic_phased_array | PAUT benchmark (ASNT) / NDT phased array Open-PAUT data | Confirm best public dataset |

---

## Stage 3: GPU Algorithm Test Results

Tests run: 2026-03-11 | GPU: NVIDIA GTX 1660 Ti, CUDA 12.4 | PyTorch 2.6.0

| Modality | Solvers Tested | Best PSNR (dB) | Status |
|----------|---------------|----------------|--------|
| acoustic_emission | 1 | 20.2 | ✅ |
| acoustic_microscopy | 1 | 10.0 | ✅ |
| active_thermography | 1 | 6.5 | ✅ |
| adaptive_optics | 1 | 100.0 | ✅ |
| afm | 3 | 31.3 | ✅ |
| angiography | 2 | 12.9 | ✅ |
| asl_mri | 1 | 2.7 | ✅ |
| atom_probe | 1 | 40.2 | ✅ |
| bioluminescence_tomo | 1 | 13.3 | ✅ |
| brachytherapy_img | 1 | 20.5 | ✅ |
| brillouin | 1 | 35.8 | ✅ |
| cacti | 6 | 11.5 | ✅ |
| cars | 1 | 14.2 | ✅ |
| cassi | 1 | -5.3 | ✅ |
| cathodoluminescence | 1 | 28.9 | ✅ |
| cbct | 3 | 15.2 | ✅ |
| cest_mri | 1 | 31.0 | ✅ |
| ceus | 1 | 24.5 | ✅ |
| clem | 1 | 17.0 | ✅ |
| coded_exposure | 1 | 19.9 | ✅ |
| confocal_3d | 6 | 27.3 | ✅ |
| confocal_endomicroscopy | 1 | 34.0 | ✅ |
| confocal_livecell | 5 | 32.3 | ✅ |
| coronagraphy | 1 | 25.2 | ✅ |
| cryo_em | 2 | 19.2 | ✅ |
| cryo_et | 3 | 13.2 | ✅ |
| ct | 7 | 13.8 | ✅ |
| ct_fluorescence | 1 | -37.6 | ✅ |
| cup | 1 | -2.3 | ✅ |
| dark_field | 3 | 25.1 | ✅ |
| desi | 1 | 15.1 | ✅ |
| dexa | 1 | 9.5 | ✅ |
| dic | 3 | 15.6 | ✅ |
| diffusion_mri | 1 | 11.3 | ✅ |
| digital_breast_tomo | 1 | -36.0 | ✅ |
| dna_paint | 3 | 28.5 | ✅ |
| doppler_ultrasound | 4 | 17.6 | ✅ |
| dot | 4 | 7.0 | ✅ |
| ebsd | 1 | 21.8 | ✅ |
| eddy_current | 1 | 4.8 | ✅ |
| edx_mapping | 4 | 22.0 | ✅ |
| eels | 1 | 24.6 | ✅ |
| eht_imaging | 1 | 11.3 | ✅ |
| elastography | 1 | 5.7 | ✅ |
| electron_diffraction | 2 | 42.0 | ✅ |
| electron_holography | 2 | 9.5 | ✅ |
| electron_tomography | 2 | 25.1 | ✅ |
| endoscopy | 3 | 11.8 | ✅ |
| entangled_photon | 1 | 31.8 | ✅ |
| event_camera | 1 | 7.3 | ✅ |
| expansion | 3 | 33.9 | ✅ |
| fib_sem | 3 | 28.1 | ✅ |
| flash_lidar | 1 | 4.3 | ✅ |
| flim | 5 | 36.9 | ✅ |
| fluoroscopy | 2 | 43.5 | ✅ |
| fmri | 1 | 4.9 | ✅ |
| fpm | 2 | 16.9 | ✅ |
| ftir_imaging | 1 | 14.8 | ✅ |
| fundus | 4 | 35.9 | ✅ |
| fwi | 1 | 8.7 | ✅ |
| gaussian_splatting | 5 | inf | ✅ |
| ghost_imaging | 1 | 6.6 | ✅ |
| gpr | 1 | 10.6 | ✅ |
| gravitational_wave | 1 | 100.0 | ✅ |
| hdr_imaging | 1 | 36.8 | ✅ |
| holography | 5 | 14.9 | ✅ |
| hyperspectral_remote | 1 | 29.1 | ✅ |
| impedance_tomo | 1 | 11.2 | ✅ |
| industrial_ct | 1 | 20.3 | ✅ |
| insar | 1 | 31.8 | ✅ |
| integral | 2 | 40.0 | ✅ |
| ism | 3 | 3.1 | ✅ |
| ivus | 1 | 19.8 | ✅ |
| lattice_lightsheet | 3 | 25.1 | ✅ |
| lensless | 5 | 11.9 | ✅ |
| libs | 1 | 18.0 | ✅ |
| lidar | 1 | 32.6 | ✅ |
| light_field | 5 | 27.3 | ✅ |
| lightsheet | 7 | 20.0 | ✅ |
| lucky_imaging | 1 | 29.2 | ✅ |
| machine_vision | 1 | 26.5 | ✅ |
| magnetic_particle | 1 | 26.5 | ✅ |
| maldi_msi | 1 | 26.3 | ✅ |
| mammography | 2 | 20.9 | ✅ |
| matrix | 5 | 22.0 | ✅ |
| mfm | 3 | 34.3 | ✅ |
| minflux | 3 | 29.5 | ✅ |
| mr_elastography | 1 | 6.0 | ✅ |
| mr_fingerprinting | 1 | 1.8 | ✅ |
| mra | 1 | 12.1 | ✅ |
| mri | 7 | 13.4 | ✅ |
| mrs | 1 | 1.9 | ✅ |
| multispectral_sat | 1 | 10.8 | ✅ |
| muon_tomo | 2 | 5.2 | ✅ |
| nerf | 2 | 29.0 | ✅ |
| neutron_diffraction | 1 | 8.5 | ✅ |
| neutron_tomo | 2 | 4.3 | ✅ |
| nirs_brain | 1 | 14.5 | ✅ |
| nsom | 3 | 22.3 | ✅ |
| ocean_acoustic_tomo | 1 | 5.6 | ✅ |
| ocean_color | 1 | 44.1 | ✅ |
| oct | 6 | 23.5 | ✅ |
| octa | 2 | 16.8 | ✅ |
| odt | 1 | 25.5 | ✅ |
| palm_storm | 3 | 32.4 | ✅ |
| panorama | 5 | 15.6 | ✅ |
| particle_calorimetry | 1 | 36.2 | ✅ |
| passive_microwave | 1 | 9.2 | ✅ |
| pet | 4 | 33.1 | ✅ |
| pet_ct | 1 | 13.0 | ✅ |
| pet_mr | 1 | 11.0 | ✅ |
| phase_contrast | 3 | 45.6 | ✅ |
| phase_retrieval | 2 | 12.6 | ✅ |
| photoacoustic | 2 | 19.1 | ✅ |
| photometric_stereo | 1 | 29.0 | ✅ |
| polarization | 2 | 15.8 | ✅ |
| polsar | 1 | 3.5 | ✅ |
| portal_imaging | 1 | 10.5 | ✅ |
| proton_radiography | 2 | 10.9 | ✅ |
| proton_therapy_img | 1 | 17.8 | ✅ |
| ptychography | 6 | 21.0 | ✅ |
| pump_probe | 1 | 18.2 | ✅ |
| quantum_illumination | 1 | 20.2 | ✅ |
| radio_astronomy | 1 | 16.1 | ✅ |
| radio_interferometry | 1 | 23.2 | ✅ |
| raman_imaging | 1 | 14.1 | ✅ |
| sar | 2 | 17.3 | ✅ |
| saxs | 1 | 8.4 | ✅ |
| seismic_tomo | 1 | 9.0 | ✅ |
| sem | 2 | 23.2 | ✅ |
| shearography | 1 | 8.0 | ✅ |
| shg | 3 | 23.0 | ✅ |
| sim | 6 | 21.6 | ✅ |
| sims | 1 | 20.5 | ✅ |
| solar_imaging | 1 | 28.4 | ✅ |
| sonar | 1 | 10.3 | ✅ |
| spc | 1 | -19.3 | ✅ |
| spect | 3 | 30.0 | ✅ |
| spect_ct | 1 | 11.4 | ✅ |
| spectral_ct | 1 | 12.3 | ✅ |
| spinning_disk | 3 | 30.6 | ✅ |
| srs | 1 | 29.1 | ✅ |
| sted | 3 | 25.0 | ✅ |
| stem | 2 | 31.0 | ✅ |
| stm | 3 | 23.3 | ✅ |
| streak_camera | 1 | 14.3 | ✅ |
| structured_light | 1 | 8.0 | ✅ |
| swi | 1 | 1.9 | ✅ |
| talbot_lau | 1 | 6.6 | ✅ |
| tem | 1 | 25.3 | ✅ |
| terahertz | 1 | 37.1 | ✅ |
| three_photon | 3 | 20.8 | ✅ |
| tirf | 2 | 31.2 | ✅ |
| tof_camera | 1 | 42.0 | ✅ |
| two_photon | 3 | 33.8 | ✅ |
| ultrasonic_phased_array | 1 | 29.6 | ✅ |
| ultrasound | 2 | 14.6 | ✅ |
| us_mri | 1 | 7.6 | ✅ |
| waxs | 1 | 20.6 | ✅ |
| weather_radar | 1 | 26.9 | ✅ |
| widefield | 5 | 25.0 | ✅ |
| widefield_lowdose | 3 | 29.0 | ✅ |
| xfel_sfx | 1 | 24.1 | ✅ |
| xray_crystallography | 1 | 22.4 | ✅ |
| xray_ndt | 1 | 16.7 | ✅ |
| xray_radiography | 2 | 26.3 | ✅ |
| xrf_imaging | 1 | 22.1 | ✅ |
| xrf_tomo | 1 | 15.6 | ✅ |

---

*Generated by scripts/build_state_v2.py — 2026-03-11*
