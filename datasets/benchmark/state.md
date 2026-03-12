# PWM Benchmark — Dataset & Pipeline State

Last updated: 2026-03-12 — 168 modalities, 168/168 Stage 1 complete, source quality checked

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

**Source Quality Check:** 168/168 ✅ — GOLD: 65 (community gold standard) | BEST: 72 (best available) | SIM: 31 (simulation standard)

---

## Quick Status Table

| Modality | Stage 0: Public Dataset | Stage 1: Dataset | Stage 2: Benchmark | Stage 3: GPU Tests | Stage 4: SpecLab |
|----------|------------------------|------------------|-------------------|--------------------|------------------|
| acoustic_emission | ✅ [SIM] AE simulation benchmark / EWGAE standards (no public AE imaging benchmark) | ✅ 12/20/20 | ❌ | ✅ 4x, best=20.2 dB | ❌ |
| acoustic_microscopy | ✅ [SIM] SAM synthetic benchmark (simulation standard — no public SAM dataset) | ✅ 12/20/20 | ❌ | ✅ 4x, best=22.0 dB | ❌ |
| active_thermography | ✅ [BEST] PVC-Infrared Dataset (Applied Sciences 2023, doi:10.3390/app13052901) / ALETHEIA 2024 | ✅ 12/20/20 | ❌ | ✅ 4x, best=7.2 dB | ❌ |
| adaptive_optics | ✅ [BEST] ESO VLT SPHERE archive + AOTools simulation | ✅ 12/20/20 | ❌ | ✅ 4x, best=100.0 dB | ❌ |
| afm | ✅ [BEST] QUAM-AFM (ACS J. Chem. Inf. Model. 2022, doi:10.1021/acs.jcim.1c01323) | ✅ 12/20/20 | ❌ | ✅ 3x, best=31.3 dB | ❌ |
| angiography | ✅ [BEST] XCAD coronary angiography (ICCV 2021) / ARCADE dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=12.9 dB | ❌ |
| asl_mri | ✅ [GOLD] Human Connectome Project ASL (hcp.nmr.wustl.edu) / ISMRM-OSIPI ASL Challenge | ✅ 12/20/20 | ❌ | ✅ 4x, best=4.1 dB | ❌ |
| atom_probe | ✅ [SIM] APT simulation benchmark (simulation standard — no public APT imaging dataset) | ✅ 12/20/20 | ❌ | ✅ 4x, best=40.2 dB | ❌ |
| bioluminescence_tomo | ✅ [SIM] BLT simulation benchmark (Ntziachristos Nature Methods 2010, simulation standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=13.3 dB | ❌ |
| brachytherapy_img | ✅ [BEST] AAPM TG-43 phantom / Open-Source TG-43 data | ✅ 12/20/20 | ❌ | ✅ 4x, best=25.2 dB | ❌ |
| brillouin | ✅ [SIM] Brillouin simulation benchmark / RRUFF spectral reference data | ✅ 12/20/20 | ❌ | ✅ 4x, best=35.8 dB | ❌ |
| cacti | ✅ [GOLD] DAVIS-2017 / Six Scenes (Liu IEEE TPAMI 2019, github.com/liuyang12/SCI) | ✅ 26/26/26 | ❌ | ✅ 6x, best=11.5 dB | ❌ |
| cars | ✅ [SIM] CARS simulation benchmark / coherent Raman imaging (simulation standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=16.7 dB | ❌ |
| cassi | ✅ [GOLD] CAVE hyperspectral (Columbia Univ.) / KAIST MST (CVPR 2022, github.com/caiyuanhao1998/MST) | ✅ 12/20/20 | ❌ | ✅ 4x, best=13.6 dB | ❌ |
| cathodoluminescence | ✅ [BEST] HyperSpy CL dataset (Zenodo 6513794) / EMPIAR CL data | ✅ 12/20/20 | ❌ | ✅ 4x, best=28.9 dB | ❌ |
| cbct | ✅ [GOLD] AAPM Low-Dose CT Challenge 2016 / LoDoPaB-CT (Sci. Data 2021) | ✅ 22/40/40 | ❌ | ✅ 3x, best=15.2 dB | ❌ |
| cest_mri | ✅ [BEST] ISMRM 2024 CEST Challenge / fastMRI brain (fastmri.med.nyu.edu) | ✅ 12/20/20 | ❌ | ✅ 4x, best=32.1 dB | ❌ |
| ceus | ✅ [GOLD] CAMUS cardiac US (CREATIS INSA-Lyon, Leclerc IEEE TMI 2019) | ✅ 12/20/20 | ❌ | ✅ 4x, best=24.5 dB | ❌ |
| clem | ✅ [BEST] EMPIAR-10094 CLEM (EBI, CC0) / OpenOrganelle CLEM | ✅ 12/20/20 | ❌ | ✅ 4x, best=28.1 dB | ❌ |
| coded_exposure | ✅ [GOLD] GoPro Deblurring (Nah CVPR 2017) / HDR+ Dataset (SIGGRAPH Asia 2016) | ✅ 12/20/20 | ❌ | ✅ 4x, best=32.1 dB | ❌ |
| confocal_3d | ✅ [GOLD] OpenCell 3D confocal (CZI) / Broad Bioimage Benchmark (BBBC) | ✅ 24/40/40 | ❌ | ✅ 6x, best=27.3 dB | ❌ |
| confocal_endomicroscopy | ✅ [BEST] UCL pCLE dataset / Mauna Kea CellvizioNet benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=34.0 dB | ❌ |
| confocal_livecell | ✅ [GOLD] LiveCell (Edlund Nature Methods 2021) / CTC Cell Tracking Challenge | ✅ 12/20/20 | ❌ | ✅ 5x, best=32.3 dB | ❌ |
| coronagraphy | ✅ [BEST] HST coronagraph MAST archive / GPIES direct-imaging survey | ✅ 12/20/20 | ❌ | ✅ 4x, best=25.2 dB | ❌ |
| cryo_em | ✅ [GOLD] EMPIAR-10028 TRPV1 (Bai Nature 2015) / EMDB GroEL / SHREC 2019 | ✅ 24/40/40 | ❌ | ✅ 5x, best=19.2 dB | ❌ |
| cryo_et | ✅ [GOLD] SHREC 2021 cryo-ET challenge / EMPIAR-10045 / IsoNet dataset | ✅ 12/20/20 | ❌ | ✅ 3x, best=13.2 dB | ❌ |
| ct | ✅ [GOLD] LoDoPaB-CT (Leuschner Sci. Data 2021, doi:10.1038/s41597-021-00893-z) | ✅ | ❌ | ✅ 7x, best=13.8 dB | ❌ |
| ct_fluorescence | ✅ [SIM] CT-FMT simulation benchmark / FLECT phantom data (Ale PLoS ONE 2009) | ✅ 12/20/20 | ❌ | ✅ 4x, best=3.3 dB | ❌ |
| cup | ✅ [SIM] CUP benchmark (Liang Nature 2018, simulation standard — no public CUP dataset) | ✅ 12/20/20 | ❌ | ✅ 4x, best=5.5 dB | ❌ |
| dark_field | ✅ [BEST] Munich Talbot-Lau dark-field CT benchmark / PSI grating data | ✅ 12/20/20 | ❌ | ✅ 3x, best=25.1 dB | ❌ |
| desi | ✅ [BEST] MetaboLights DESI-MSI dataset / EMBL-EBI MSI archive | ✅ 12/20/20 | ❌ | ✅ 4x, best=15.1 dB | ❌ |
| dexa | ✅ [BEST] OsteoArthritis Initiative (OAI) DXA — UCSF (oai.ucsf.edu) | ✅ 12/20/20 | ❌ | ✅ 4x, best=inf dB | ❌ |
| dic | ✅ [BEST] SciPy phase benchmark / ACPA DIC Challenge dataset | ✅ 12/20/20 | ❌ | ✅ 3x, best=15.6 dB | ❌ |
| diffusion_mri | ✅ [GOLD] Human Connectome Project dMRI (hcp.nmr.wustl.edu) / Sherbrooke-3T | ✅ 12/20/20 | ❌ | ✅ 4x, best=11.3 dB | ❌ |
| digital_breast_tomo | ✅ [BEST] INBreast (BCDR) / VDM-100 DBT dataset (TCIA) | ✅ 12/20/20 | ❌ | ✅ 4x, best=2.5 dB | ❌ |
| dna_paint | ✅ [GOLD] SMLM Challenge 2016 / DNA-PAINT sim benchmark (Jungmann lab) | ✅ 12/20/20 | ❌ | ✅ 3x, best=28.5 dB | ❌ |
| doppler_ultrasound | ✅ [GOLD] EchoNet-Dynamic (Stanford, Ouyang Nature 2020) / CAMUS | ✅ 12/20/20 | ❌ | ✅ 6x, best=17.6 dB | ❌ |
| dot | ✅ [SIM] UCL DOT simulation benchmark / TOAST++ reference (Schweiger & Arridge 2014) | ✅ 12/20/20 | ❌ | ✅ 6x, best=7.0 dB | ❌ |
| ebsd | ✅ [BEST] DREAM.3D synthetic EBSD / NIST SRM EBSD benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=21.9 dB | ❌ |
| eddy_current | ✅ [SIM] EEDB NDT benchmark / eddy current simulation (industry standard, no public) | ✅ 12/20/20 | ❌ | ✅ 4x, best=22.9 dB | ❌ |
| edx_mapping | ✅ [BEST] NIST SRM-2460 EDX / HyperSpy EDX demo dataset (Zenodo) | ✅ 12/20/20 | ❌ | ✅ 4x, best=22.0 dB | ❌ |
| eels | ✅ [BEST] EELS.info database (eels.info) / Cornell EELS dataset | ✅ 12/20/20 | ❌ | ✅ 5x, best=25.2 dB | ❌ |
| eht_imaging | ✅ [GOLD] EHT 2019 M87 public data release (eventhorizontelescope.org) | ✅ 12/20/20 | ❌ | ✅ 4x, best=11.4 dB | ❌ |
| elastography | ✅ [BEST] MRE phantom NIST / RSNA Quantitative Imaging Biomarker Alliance | ✅ 12/20/20 | ❌ | ✅ 4x, best=11.0 dB | ❌ |
| electron_diffraction | ✅ [BEST] CIF/ICSD + RRUFF ED patterns / CBED simulation benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=42.0 dB | ❌ |
| electron_holography | ✅ [BEST] EMDB holography / simulation benchmark (Lichte & Lehmann 2008) | ✅ 12/20/20 | ❌ | ✅ 4x, best=9.5 dB | ❌ |
| electron_tomography | ✅ [GOLD] EMPIAR-10005 / EMPIAR-10045 (EBI) / EMDB tilt series | ✅ 12/20/20 | ❌ | ✅ 4x, best=25.1 dB | ❌ |
| endoscopy | ✅ [GOLD] Kvasir-SEG (Jha IEEE Access 2020) / CholecT50 / HyperKvasir | ✅ 12/20/20 | ❌ | ✅ 3x, best=11.8 dB | ❌ |
| entangled_photon | ✅ [SIM] Quantum imaging simulation benchmark (simulation standard — emerging field) | ✅ 12/20/20 | ❌ | ✅ 4x, best=31.8 dB | ❌ |
| event_camera | ✅ [GOLD] DAVIS 240C / N-Caltech101 / MVSEC (Zhu RAL 2018) | ✅ 12/20/20 | ❌ | ✅ 4x, best=7.3 dB | ❌ |
| expansion | ✅ [BEST] ExPath benchmark / Allen Institute ExM public data | ✅ 12/20/20 | ❌ | ✅ 3x, best=33.9 dB | ❌ |
| fib_sem | ✅ [GOLD] OpenOrganelle FIB-SEM (Janelia, janelia.org) / H01 connectome | ✅ 12/20/20 | ❌ | ✅ 3x, best=28.1 dB | ❌ |
| flash_lidar | ✅ [GOLD] KITTI LiDAR (Geiger CVPR 2012) / Middlebury flash 3D | ✅ 12/20/20 | ❌ | ✅ 4x, best=4.3 dB | ❌ |
| flim | ✅ [BEST] FLUTE benchmark (Zanacchi Nature Methods 2019) / FLIM-FRET dataset | ✅ 12/20/20 | ❌ | ✅ 5x, best=36.9 dB | ❌ |
| fluoroscopy | ✅ [BEST] TCIA Fluoroscopy / CVC-ClinicDB (Bernal CMIG 2015) | ✅ 12/20/20 | ❌ | ✅ 4x, best=43.6 dB | ❌ |
| fmri | ✅ [GOLD] Human Connectome Project fMRI / OpenNeuro (Poldrack OpenNeuro 2013) | ✅ 12/20/20 | ❌ | ✅ 4x, best=4.9 dB | ❌ |
| fpm | ✅ [BEST] FPM benchmark (Tian Light Sci. Appl. 2015) / UCB FPM dataset | ✅ 12/20/20 | ❌ | ✅ 5x, best=16.9 dB | ❌ |
| ftir_imaging | ✅ [BEST] USGS spectral library v7 (usgs.gov) / SFDB FTIR benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=34.6 dB | ❌ |
| fundus | ✅ [GOLD] DRIVE (Staal IEEE TMI 2004) / STARE / CHASE_DB1 / DiaRetDB | ✅ 12/20/20 | ❌ | ✅ 4x, best=35.9 dB | ❌ |
| fwi | ✅ [GOLD] OpenFWI (Deng IEEE TGRS 2021) / SEG-SALT / Marmousi-2 | ✅ 12/20/20 | ❌ | ✅ 4x, best=8.7 dB | ❌ |
| gaussian_splatting | ✅ [GOLD] Tanks & Temples (Knapitsch SIGGRAPH 2017) / Mip-NeRF360 / Blender | ✅ 12/20/20 | ❌ | ✅ 5x, best=inf dB | ❌ |
| ghost_imaging | ✅ [SIM] Ghost imaging simulation / computational GI benchmark (Shapiro PRA 2008) | ✅ 12/20/20 | ❌ | ✅ 4x, best=inf dB | ❌ |
| gpr | ✅ [SIM] GPR simulation benchmark / gprMax (Warren CPC 2016) / SEG-Y standard | ✅ 12/20/20 | ❌ | ✅ 4x, best=10.6 dB | ❌ |
| gravitational_wave | ✅ [GOLD] LIGO O3 public data (GWOSC, gwosc.org) / GWTC-3 catalog | ✅ 12/20/20 | ❌ | ✅ 4x, best=100.0 dB | ❌ |
| hdr_imaging | ✅ [GOLD] HDR-DB (Fairchild RIT) / HDREye / Laval HDR panorama database | ✅ 12/20/20 | ❌ | ✅ 4x, best=36.8 dB | ❌ |
| holography | ✅ [BEST] HoloPy benchmark / DHM simulation / FINCH holography data | ✅ 12/20/20 | ❌ | ✅ 5x, best=14.9 dB | ❌ |
| hyperspectral_remote | ✅ [GOLD] AVIRIS Indian Pines / ROSIS Pavia / GRSS Data Fusion Contest | ✅ 36/60/60 | ❌ | ✅ 4x, best=29.1 dB | ❌ |
| impedance_tomo | ✅ [BEST] EIDORS simulation framework / Finnish EIT challenge (FEIT) | ✅ 12/20/20 | ❌ | ✅ 4x, best=inf dB | ❌ |
| industrial_ct | ✅ [BEST] GCPD industrial CT / Zeiss Xradia / WoDT benchmark | ✅ 24/40/40 | ❌ | ✅ 4x, best=20.3 dB | ❌ |
| insar | ✅ [GOLD] Sentinel-1 SLC archive (ESA Copernicus, esa.int) / COSAR benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=31.8 dB | ❌ |
| integral | ✅ [BEST] EPFL integral imaging dataset / Stanford Light Field archive | ✅ 12/20/20 | ❌ | ✅ 5x, best=40.0 dB | ❌ |
| ism | ✅ [SIM] ISM simulation benchmark / Airyscan comparison data (Muller Nat Meth 2010) | ✅ 12/20/20 | ❌ | ✅ 3x, best=3.1 dB | ❌ |
| ivus | ✅ [BEST] MICCAI 2011 IVUS segmentation challenge / CARDIAC Atlas Project | ✅ 12/20/20 | ❌ | ✅ 4x, best=19.8 dB | ❌ |
| lattice_lightsheet | ✅ [BEST] Allen Cell Institute lattice light-sheet / Janelia LLS data | ✅ 12/20/20 | ❌ | ✅ 3x, best=25.1 dB | ❌ |
| lensless | ✅ [GOLD] DiffuserCam (Monakhova Optica 2019) / PhlatCam benchmark | ✅ 12/20/20 | ❌ | ✅ 5x, best=11.9 dB | ❌ |
| libs | ✅ [BEST] NIST LIBS database (nist.gov/srd) / RRUFF LIBS spectra | ✅ 12/20/20 | ❌ | ✅ 4x, best=26.5 dB | ❌ |
| lidar | ✅ [GOLD] KITTI LiDAR (Geiger CVPR 2012) / nuScenes / SemanticKITTI | ✅ 24/40/40 | ❌ | ✅ 4x, best=32.7 dB | ❌ |
| light_field | ✅ [GOLD] Stanford Light Field Archive (lightfield.stanford.edu) / INRIA LF | ✅ 12/20/20 | ❌ | ✅ 5x, best=27.3 dB | ❌ |
| lightsheet | ✅ [GOLD] Allen Brain Atlas light-sheet (alleninstitute.org) / Zebrafish SPIM | ✅ 24/40/40 | ❌ | ✅ 7x, best=20.0 dB | ❌ |
| lucky_imaging | ✅ [SIM] Lucky imaging simulation benchmark (Law NJP 2006, no dominant public dataset) | ✅ 12/20/20 | ❌ | ✅ 4x, best=29.6 dB | ❌ |
| machine_vision | ✅ [GOLD] MVTec Anomaly Detection (Bergmann CVPR 2019) / BSDS500 | ✅ 12/20/20 | ❌ | ✅ 4x, best=28.3 dB | ❌ |
| magnetic_particle | ✅ [GOLD] OpenMPIData (Knopp IJMRI 2016, zenodo.org) / MPI reconstruction challenge | ✅ 12/20/20 | ❌ | ✅ 4x, best=26.5 dB | ❌ |
| maldi_msi | ✅ [BEST] MetaboLights MSI / PRIDE-MALDI database (EBI) | ✅ 12/20/20 | ❌ | ✅ 4x, best=27.1 dB | ❌ |
| mammography | ✅ [GOLD] CBIS-DDSM (Lee Sci. Data 2017) / VinDr-Mammo / INBreast | ✅ 24/40/40 | ❌ | ✅ 4x, best=20.9 dB | ❌ |
| matrix | ✅ [BEST] matrix completion benchmark / Jester / ML-100K (MovieLens) | ✅ 12/20/20 | ❌ | ✅ 5x, best=22.0 dB | ❌ |
| mfm | ✅ [SIM] MFM simulation benchmark / AFM/MFM reference (simulation standard) | ✅ 12/20/20 | ❌ | ✅ 3x, best=34.3 dB | ❌ |
| minflux | ✅ [SIM] MINFLUX simulation benchmark (Balzarotti Science 2017, no public dataset) | ✅ 12/20/20 | ❌ | ✅ 3x, best=29.5 dB | ❌ |
| mr_elastography | ✅ [BEST] MRE-NIST phantom data / RSNA QIBA MRE challenge | ✅ 12/20/20 | ❌ | ✅ 4x, best=6.0 dB | ❌ |
| mr_fingerprinting | ✅ [BEST] MRF simulation (Ma Nature 2013) / CPMG relaxometry data | ✅ 12/20/20 | ❌ | ✅ 4x, best=4.2 dB | ❌ |
| mra | ✅ [BEST] TOF-MRA (MICCAI ADAM/IXI dataset) / 1000PLUS | ✅ 12/20/20 | ❌ | ✅ 4x, best=12.1 dB | ❌ |
| mri | ✅ [GOLD] fastMRI multi-coil k-space (Zbontar NeurIPS 2018, fastmri.med.nyu.edu) | ✅ 24/40/40 | ❌ | ✅ 7x, best=13.4 dB | ❌ |
| mrs | ✅ [BEST] MRSHUB benchmark (mrshub.org) / BIG-PRESS simulation / ISMRM MRS challenge | ✅ 12/20/20 | ❌ | ✅ 4x, best=4.3 dB | ❌ |
| multispectral_sat | ✅ [GOLD] Sentinel-2 (ESA Copernicus) / WorldView-3 / DESIS hyperspectral | ✅ 12/20/20 | ❌ | ✅ 4x, best=11.3 dB | ❌ |
| muon_tomo | ✅ [SIM] Muon tomography simulation / GEANT4 benchmark (CERN open data) | ✅ 12/20/20 | ❌ | ✅ 4x, best=5.2 dB | ❌ |
| nerf | ✅ [GOLD] NeRF Blender (Mildenhall ECCV 2020) / LLFF / DTU MVS dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=29.0 dB | ❌ |
| neutron_diffraction | ✅ [BEST] ILL neutron diffraction data / SINQ PSI / ICSD CIF structures | ✅ 12/20/20 | ❌ | ✅ 4x, best=8.5 dB | ❌ |
| neutron_tomo | ✅ [BEST] PSI NEUTRA dataset / ILL ICON neutron CT | ✅ 12/20/20 | ❌ | ✅ 4x, best=6.6 dB | ❌ |
| nirs_brain | ✅ [BEST] fNIRS-BIDS benchmark / LABBRAIN fNIRS dataset / UCL Multimodal Imaging | ✅ 12/20/20 | ❌ | ✅ 4x, best=20.2 dB | ❌ |
| nsom | ✅ [SIM] NSOM simulation benchmark (simulation standard — no public NSOM dataset) | ✅ 12/20/20 | ❌ | ✅ 3x, best=22.3 dB | ❌ |
| ocean_acoustic_tomo | ✅ [SIM] NOAA ocean acoustic reference / SWEX simulation (simulation standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=26.6 dB | ❌ |
| ocean_color | ✅ [GOLD] NASA MODIS ocean color (oceancolor.gsfc.nasa.gov) / SeaWiFS dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=44.1 dB | ❌ |
| oct | ✅ [GOLD] RETOUCH (Bogunovic IVCM 2019) / Duke OCT / OPTIMA retinal OCT | ✅ 24/40/40 | ❌ | ✅ 6x, best=23.5 dB | ❌ |
| octa | ✅ [BEST] ROSE dataset (Ma TPAMI 2021) / CAVF OCTA benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=18.8 dB | ❌ |
| odt | ✅ [BEST] 2.5D DIC/ODT benchmark / Toulouse ODT dataset / TORCH benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=25.5 dB | ❌ |
| palm_storm | ✅ [GOLD] SMLM Challenge 2016 (smlmchallenge.net) / ThunderSTORM benchmark | ✅ 24/40/40 | ❌ | ✅ 3x, best=32.4 dB | ❌ |
| panorama | ✅ [GOLD] SUN360 (Xiao CVPR 2012) / Laval HDR Panorama Dataset | ✅ 12/20/20 | ❌ | ✅ 5x, best=15.6 dB | ❌ |
| particle_calorimetry | ✅ [GOLD] GEANT4 CaloChallenge 2022 (Fast Calorimeter Simulation Challenge) | ✅ 12/20/20 | ❌ | ✅ 4x, best=36.7 dB | ❌ |
| passive_microwave | ✅ [BEST] AMSR2 / SSMIS Level-3 (NASA NSIDC) / GMI precipitation data | ✅ 12/20/20 | ❌ | ✅ 4x, best=16.9 dB | ❌ |
| pet | ✅ [GOLD] TCIA-PET LIDC (Clark Sci. Data 2013) / OpenPET simulation data | ✅ 12/20/20 | ❌ | ✅ 4x, best=33.1 dB | ❌ |
| pet_ct | ✅ [GOLD] TCIA PET-CT (The Cancer Imaging Archive) / MAASTRO PET-CT dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=13.0 dB | ❌ |
| pet_mr | ✅ [BEST] MICCAI PET-MR challenge / BrainPET dataset / ADNI PET-MRI | ✅ 24/40/40 | ❌ | ✅ 4x, best=11.0 dB | ❌ |
| phase_contrast | ✅ [BEST] CXLS phase contrast / APS phase contrast dataset / Siemens Fresnel | ✅ 12/20/20 | ❌ | ✅ 3x, best=45.6 dB | ❌ |
| phase_retrieval | ✅ [BEST] CDI challenge benchmark / ptychography phase retrieval (Zenodo) | ✅ 12/20/20 | ❌ | ✅ 5x, best=12.6 dB | ❌ |
| photoacoustic | ✅ [GOLD] MICCAI PATATO dataset / PAT-Public (ucl.ac.uk) / OADAT benchmark | ✅ 24/40/40 | ❌ | ✅ 5x, best=19.1 dB | ❌ |
| photometric_stereo | ✅ [GOLD] DiLiGenT-MV (Ren IEEE TPAMI 2022) / CyclesPS benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=29.0 dB | ❌ |
| polarization | ✅ [BEST] AOLP dataset (Tyo Appl. Opt. 2006) / Polarization benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=15.8 dB | ❌ |
| polsar | ✅ [GOLD] UAVSAR (NASA JPL) / SIR-C / RADARSAT-2 PolSAR (MDA) | ✅ 12/20/20 | ❌ | ✅ 4x, best=7.2 dB | ❌ |
| portal_imaging | ✅ [BEST] EPID benchmark / AAPM TG-58 portal imaging dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=17.3 dB | ❌ |
| proton_radiography | ✅ [SIM] pCT simulation benchmark / TOPAS-nBio MC (Schulte MedPhys 2008) | ✅ 12/20/20 | ❌ | ✅ 4x, best=12.0 dB | ❌ |
| proton_therapy_img | ✅ [SIM] Proton CT simulation (TOPAS MC) / OpenTPS benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=26.6 dB | ❌ |
| ptychography | ✅ [BEST] CDI ptychography benchmark (Zenodo) / CXLS/ALS ptychography data | ✅ 12/20/20 | ❌ | ✅ 6x, best=21.0 dB | ❌ |
| pump_probe | ✅ [SIM] Ultrafast pump-probe simulation benchmark (simulation standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=18.2 dB | ❌ |
| quantum_illumination | ✅ [SIM] Quantum illumination simulation (Lloyd Science 2008, emerging field) | ✅ 12/20/20 | ❌ | ✅ 4x, best=20.2 dB | ❌ |
| radio_astronomy | ✅ [GOLD] LOFAR HBA survey / VLA FIRST (White ApJ 1997) / ALMA calibration | ✅ 12/20/20 | ❌ | ✅ 4x, best=37.3 dB | ❌ |
| radio_interferometry | ✅ [BEST] MeerKAT MeerLICHT / VLBI imaging challenge 2022 (radiointerferometrychallenege.github.io) | ✅ 12/20/20 | ❌ | ✅ 4x, best=23.2 dB | ❌ |
| raman_imaging | ✅ [GOLD] RRUFF Raman database (rruff.info) / NIST SRM Raman benchmark | ✅ 24/40/40 | ❌ | ✅ 4x, best=19.7 dB | ❌ |
| sar | ✅ [GOLD] Sentinel-1 GRD (ESA Copernicus) / UAVSAR (NASA JPL) / ERS-2 | ✅ 24/40/40 | ❌ | ✅ 4x, best=17.8 dB | ❌ |
| saxs | ✅ [BEST] cSAXS synchrotron data (PSI) / ALS SAXS dataset / ESRF BM26 | ✅ 12/20/20 | ❌ | ✅ 4x, best=8.4 dB | ❌ |
| seismic_tomo | ✅ [BEST] IRIS SEED seismic (ds.iris.edu) / SEG-Y NCEDC dataset / Marmousi | ✅ 12/20/20 | ❌ | ✅ 4x, best=9.0 dB | ❌ |
| sem | ✅ [GOLD] SEM-CIFA dataset / NIST SEM calibration / ZEISS SEM benchmark | ✅ 24/40/40 | ❌ | ✅ 4x, best=23.2 dB | ❌ |
| shearography | ✅ [SIM] Digital shearography simulation benchmark (NDT standard, no public dataset) | ✅ 12/20/20 | ❌ | ✅ 4x, best=13.2 dB | ❌ |
| shg | ✅ [SIM] SHG collagen benchmark / NLO microscopy simulation (Campagnola Anal Chem 2011) | ✅ 12/20/20 | ❌ | ✅ 3x, best=23.0 dB | ❌ |
| sim | ✅ [GOLD] SIMbench (Culley Nature Methods 2018) / SMLM SIM benchmark | ✅ 24/40/40 | ❌ | ✅ 6x, best=21.6 dB | ❌ |
| sims | ✅ [SIM] SIMS surface database / ToF-SIMS benchmark simulation (simulation standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=20.5 dB | ❌ |
| solar_imaging | ✅ [GOLD] SDO AIA (HEK, lmsal.com) / SOHO EIT / TRACE EUV solar archive | ✅ 12/20/20 | ❌ | ✅ 4x, best=28.4 dB | ❌ |
| sonar | ✅ [SIM] NOAA multibeam sonar archive / acoustic simulation benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=15.0 dB | ❌ |
| spc | ✅ [BEST] SPC simulation benchmark / Rice SPC dataset (Duarte Science 2008) | ✅ 12/20/20 | ❌ | ✅ 4x, best=6.8 dB | ❌ |
| spect | ✅ [BEST] SIMIND simulation framework / GATE SPECT benchmark (OpenGATE) | ✅ 12/20/20 | ❌ | ✅ 3x, best=30.0 dB | ❌ |
| spect_ct | ✅ [GOLD] TCIA SPECT-CT (The Cancer Imaging Archive) / Philips IQ-SPECT | ✅ 24/40/40 | ❌ | ✅ 4x, best=11.4 dB | ❌ |
| spectral_ct | ✅ [GOLD] AAPM Spectral CT challenge / Medipix3 spectral CT dataset | ✅ 24/40/40 | ❌ | ✅ 4x, best=12.3 dB | ❌ |
| spinning_disk | ✅ [BEST] Spinning disk benchmark / BBBC (Broad Bioimage Benchmark Collection) | ✅ 12/20/20 | ❌ | ✅ 3x, best=30.6 dB | ❌ |
| srs | ✅ [SIM] SRS benchmark / coherent Raman spectral imaging (simulation standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=29.1 dB | ❌ |
| sted | ✅ [GOLD] STED benchmark (Culley Nature Methods 2018) / Leica/Abberior data | ✅ 24/40/40 | ❌ | ✅ 3x, best=25.0 dB | ❌ |
| stem | ✅ [BEST] AAEM STEM benchmark / EMPIAR STEM datasets / NIST STEM SRM | ✅ 12/20/20 | ❌ | ✅ 4x, best=31.0 dB | ❌ |
| stm | ✅ [BEST] STM database (nanosurf.com) / NIST surface topography SRM | ✅ 12/20/20 | ❌ | ✅ 3x, best=23.3 dB | ❌ |
| streak_camera | ✅ [SIM] Streak camera simulation benchmark (simulation standard — no public dataset) | ✅ 12/20/20 | ❌ | ✅ 4x, best=30.8 dB | ❌ |
| structured_light | ✅ [BEST] SL benchmark (Gupta CVPR 2012) / CAVE SL dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=8.0 dB | ❌ |
| swi | ✅ [BEST] SWI benchmark / OpenNeuro SWI dataset (openneuro.org) | ✅ 12/20/20 | ❌ | ✅ 4x, best=4.6 dB | ❌ |
| talbot_lau | ✅ [BEST] Munich Talbot-Lau grating data (TU Munich) / PSI grating CT | ✅ 12/20/20 | ❌ | ✅ 4x, best=28.9 dB | ❌ |
| tem | ✅ [GOLD] EMPIAR TEM datasets (EBI) / JEOL benchmark / NIST TEM SRM | ✅ 24/40/40 | ❌ | ✅ 4x, best=25.3 dB | ❌ |
| terahertz | ✅ [BEST] THz-TDS simulation benchmark / NIST THz spectroscopy database | ✅ 12/20/20 | ❌ | ✅ 4x, best=37.1 dB | ❌ |
| three_photon | ✅ [BEST] 3PM simulation / Kleinfeld lab 3PM dataset (UCSD) | ✅ 12/20/20 | ❌ | ✅ 3x, best=20.8 dB | ❌ |
| tirf | ✅ [BEST] TIRF benchmark (SMLM Challenge) / Cell-TIRF dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=31.2 dB | ❌ |
| tof_camera | ✅ [GOLD] ETH3D (Schops CVPR 2017) / Middlebury 3D ToF / TUM RGB-D | ✅ 12/20/20 | ❌ | ✅ 4x, best=42.0 dB | ❌ |
| two_photon | ✅ [GOLD] Allen Brain 2P-SCC (alleninstitute.org) / Carandini-Harris dataset | ✅ 24/40/40 | ❌ | ✅ 3x, best=33.8 dB | ❌ |
| ultrasonic_phased_array | ✅ [SIM] PAUT benchmark / NDT phased array simulation (ASNT standard) | ✅ 12/20/20 | ❌ | ✅ 4x, best=30.8 dB | ❌ |
| ultrasound | ✅ [GOLD] CAMUS (Leclerc IEEE TMI 2019) / EchoNet-Dynamic (Ouyang Nature 2020) | ✅ 12/20/20 | ❌ | ✅ 5x, best=14.8 dB | ❌ |
| us_mri | ✅ [BEST] Ultrashort TE / ZTE MRI benchmark / PETRA dataset (Siemens) | ✅ 12/20/20 | ❌ | ✅ 4x, best=25.5 dB | ❌ |
| waxs | ✅ [BEST] ESRF WAXS archive / ALS SAXS/WAXS / DLS SAXS data | ✅ 12/20/20 | ❌ | ✅ 4x, best=20.6 dB | ❌ |
| weather_radar | ✅ [GOLD] NEXRAD WSR-88D (NOAA, ncdc.noaa.gov) / MetOffice C-band / OPERA | ✅ 12/20/20 | ❌ | ✅ 4x, best=26.9 dB | ❌ |
| widefield | ✅ [BEST] BSDS500 / MitoCheck widefield (EMBL) / Broad BBBC benchmark | ✅ 24/40/40 | ❌ | ✅ 5x, best=25.0 dB | ❌ |
| widefield_lowdose | ✅ [GOLD] CARE low-dose fluorescence (Weigert Nature Methods 2018) / BBBC | ✅ 12/20/20 | ❌ | ✅ 3x, best=29.0 dB | ❌ |
| xfel_sfx | ✅ [BEST] CFEL SFX benchmark / LCLS SFX data (lcls.slac.stanford.edu) | ✅ 12/20/20 | ❌ | ✅ 4x, best=24.1 dB | ❌ |
| xray_crystallography | ✅ [GOLD] PDB (Protein Data Bank, rcsb.org) / CCDC CSD / ICDD PDF-4+ | ✅ 12/20/20 | ❌ | ✅ 4x, best=22.4 dB | ❌ |
| xray_ndt | ✅ [BEST] ASTM NDT E1000 / WoDT benchmark / Zeiss Xradia NDT dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=16.7 dB | ❌ |
| xray_radiography | ✅ [GOLD] Chest X-ray14 (Wang CVPR 2017) / PadChest / CheXpert (Stanford) | ✅ 12/20/20 | ❌ | ✅ 4x, best=26.3 dB | ❌ |
| xrf_imaging | ✅ [BEST] ESRF XRF imaging dataset / APS XRF benchmark | ✅ 12/20/20 | ❌ | ✅ 4x, best=22.1 dB | ❌ |
| xrf_tomo | ✅ [BEST] XRF-CT benchmark (APS) / ESRF XRF-CT / Dls I18 dataset | ✅ 12/20/20 | ❌ | ✅ 4x, best=15.6 dB | ❌ |

**Summary:**
- Stage 0 (Dataset Verified): 168/168 ✅ | Source Quality: 65 GOLD + 72 BEST + 31 SIM
- Stage 1 (Datasets Created): 168/168 ✅
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
| active_thermography | [BEST] PVC-Infrared Dataset (Applied Sciences 2023, doi:10.3390/app13052901) / ALETHEIA 2024 |
| adaptive_optics | [BEST] ESO VLT SPHERE archive + AOTools simulation |
| afm | [BEST] QUAM-AFM (ACS J. Chem. Inf. Model. 2022, doi:10.1021/acs.jcim.1c01323) |
| angiography | [BEST] XCAD coronary angiography (ICCV 2021) / ARCADE dataset |
| asl_mri | [GOLD] Human Connectome Project ASL (hcp.nmr.wustl.edu) / ISMRM-OSIPI ASL Challenge |
| brachytherapy_img | [BEST] AAPM TG-43 phantom / Open-Source TG-43 data |
| cacti | [GOLD] DAVIS-2017 / Six Scenes (Liu IEEE TPAMI 2019, github.com/liuyang12/SCI) |
| cassi | [GOLD] CAVE hyperspectral (Columbia Univ.) / KAIST MST (CVPR 2022, github.com/caiyuanhao1998/MST) |
| cathodoluminescence | [BEST] HyperSpy CL dataset (Zenodo 6513794) / EMPIAR CL data |
| cbct | [GOLD] AAPM Low-Dose CT Challenge 2016 / LoDoPaB-CT (Sci. Data 2021) |
| cest_mri | [BEST] ISMRM 2024 CEST Challenge / fastMRI brain (fastmri.med.nyu.edu) |
| ceus | [GOLD] CAMUS cardiac US (CREATIS INSA-Lyon, Leclerc IEEE TMI 2019) |
| clem | [BEST] EMPIAR-10094 CLEM (EBI, CC0) / OpenOrganelle CLEM |
| coded_exposure | [GOLD] GoPro Deblurring (Nah CVPR 2017) / HDR+ Dataset (SIGGRAPH Asia 2016) |
| confocal_3d | [GOLD] OpenCell 3D confocal (CZI) / Broad Bioimage Benchmark (BBBC) |
| confocal_endomicroscopy | [BEST] UCL pCLE dataset / Mauna Kea CellvizioNet benchmark |
| confocal_livecell | [GOLD] LiveCell (Edlund Nature Methods 2021) / CTC Cell Tracking Challenge |
| coronagraphy | [BEST] HST coronagraph MAST archive / GPIES direct-imaging survey |
| cryo_em | [GOLD] EMPIAR-10028 TRPV1 (Bai Nature 2015) / EMDB GroEL / SHREC 2019 |
| cryo_et | [GOLD] SHREC 2021 cryo-ET challenge / EMPIAR-10045 / IsoNet dataset |
| ct | [GOLD] LoDoPaB-CT (Leuschner Sci. Data 2021, doi:10.1038/s41597-021-00893-z) |
| dark_field | [BEST] Munich Talbot-Lau dark-field CT benchmark / PSI grating data |
| desi | [BEST] MetaboLights DESI-MSI dataset / EMBL-EBI MSI archive |
| dexa | [BEST] OsteoArthritis Initiative (OAI) DXA — UCSF (oai.ucsf.edu) |
| dic | [BEST] SciPy phase benchmark / ACPA DIC Challenge dataset |
| diffusion_mri | [GOLD] Human Connectome Project dMRI (hcp.nmr.wustl.edu) / Sherbrooke-3T |
| digital_breast_tomo | [BEST] INBreast (BCDR) / VDM-100 DBT dataset (TCIA) |
| dna_paint | [GOLD] SMLM Challenge 2016 / DNA-PAINT sim benchmark (Jungmann lab) |
| doppler_ultrasound | [GOLD] EchoNet-Dynamic (Stanford, Ouyang Nature 2020) / CAMUS |
| ebsd | [BEST] DREAM.3D synthetic EBSD / NIST SRM EBSD benchmark |
| edx_mapping | [BEST] NIST SRM-2460 EDX / HyperSpy EDX demo dataset (Zenodo) |
| eels | [BEST] EELS.info database (eels.info) / Cornell EELS dataset |
| eht_imaging | [GOLD] EHT 2019 M87 public data release (eventhorizontelescope.org) |
| elastography | [BEST] MRE phantom NIST / RSNA Quantitative Imaging Biomarker Alliance |
| electron_diffraction | [BEST] CIF/ICSD + RRUFF ED patterns / CBED simulation benchmark |
| electron_tomography | [GOLD] EMPIAR-10005 / EMPIAR-10045 (EBI) / EMDB tilt series |
| endoscopy | [GOLD] Kvasir-SEG (Jha IEEE Access 2020) / CholecT50 / HyperKvasir |
| event_camera | [GOLD] DAVIS 240C / N-Caltech101 / MVSEC (Zhu RAL 2018) |
| expansion | [BEST] ExPath benchmark / Allen Institute ExM public data |
| fib_sem | [GOLD] OpenOrganelle FIB-SEM (Janelia, janelia.org) / H01 connectome |
| flash_lidar | [GOLD] KITTI LiDAR (Geiger CVPR 2012) / Middlebury flash 3D |
| flim | [BEST] FLUTE benchmark (Zanacchi Nature Methods 2019) / FLIM-FRET dataset |
| fluoroscopy | [BEST] TCIA Fluoroscopy / CVC-ClinicDB (Bernal CMIG 2015) |
| fmri | [GOLD] Human Connectome Project fMRI / OpenNeuro (Poldrack OpenNeuro 2013) |
| fpm | [BEST] FPM benchmark (Tian Light Sci. Appl. 2015) / UCB FPM dataset |
| ftir_imaging | [BEST] USGS spectral library v7 (usgs.gov) / SFDB FTIR benchmark |
| fundus | [GOLD] DRIVE (Staal IEEE TMI 2004) / STARE / CHASE_DB1 / DiaRetDB |
| fwi | [GOLD] OpenFWI (Deng IEEE TGRS 2021) / SEG-SALT / Marmousi-2 |
| gaussian_splatting | [GOLD] Tanks & Temples (Knapitsch SIGGRAPH 2017) / Mip-NeRF360 / Blender |
| gravitational_wave | [GOLD] LIGO O3 public data (GWOSC, gwosc.org) / GWTC-3 catalog |
| hdr_imaging | [GOLD] HDR-DB (Fairchild RIT) / HDREye / Laval HDR panorama database |
| holography | [BEST] HoloPy benchmark / DHM simulation / FINCH holography data |
| hyperspectral_remote | [GOLD] AVIRIS Indian Pines / ROSIS Pavia / GRSS Data Fusion Contest |
| impedance_tomo | [BEST] EIDORS simulation framework / Finnish EIT challenge (FEIT) |
| industrial_ct | [BEST] GCPD industrial CT / Zeiss Xradia / WoDT benchmark |
| insar | [GOLD] Sentinel-1 SLC archive (ESA Copernicus, esa.int) / COSAR benchmark |
| integral | [BEST] EPFL integral imaging dataset / Stanford Light Field archive |
| ivus | [BEST] MICCAI 2011 IVUS segmentation challenge / CARDIAC Atlas Project |
| lattice_lightsheet | [BEST] Allen Cell Institute lattice light-sheet / Janelia LLS data |
| lensless | [GOLD] DiffuserCam (Monakhova Optica 2019) / PhlatCam benchmark |
| libs | [BEST] NIST LIBS database (nist.gov/srd) / RRUFF LIBS spectra |
| lidar | [GOLD] KITTI LiDAR (Geiger CVPR 2012) / nuScenes / SemanticKITTI |
| light_field | [GOLD] Stanford Light Field Archive (lightfield.stanford.edu) / INRIA LF |
| lightsheet | [GOLD] Allen Brain Atlas light-sheet (alleninstitute.org) / Zebrafish SPIM |
| machine_vision | [GOLD] MVTec Anomaly Detection (Bergmann CVPR 2019) / BSDS500 |
| magnetic_particle | [GOLD] OpenMPIData (Knopp IJMRI 2016, zenodo.org) / MPI reconstruction challenge |
| maldi_msi | [BEST] MetaboLights MSI / PRIDE-MALDI database (EBI) |
| mammography | [GOLD] CBIS-DDSM (Lee Sci. Data 2017) / VinDr-Mammo / INBreast |
| matrix | [BEST] matrix completion benchmark / Jester / ML-100K (MovieLens) |
| mr_elastography | [BEST] MRE-NIST phantom data / RSNA QIBA MRE challenge |
| mr_fingerprinting | [BEST] MRF simulation (Ma Nature 2013) / CPMG relaxometry data |
| mra | [BEST] TOF-MRA (MICCAI ADAM/IXI dataset) / 1000PLUS |
| mri | [GOLD] fastMRI multi-coil k-space (Zbontar NeurIPS 2018, fastmri.med.nyu.edu) |
| mrs | [BEST] MRSHUB benchmark (mrshub.org) / BIG-PRESS simulation / ISMRM MRS challenge |
| multispectral_sat | [GOLD] Sentinel-2 (ESA Copernicus) / WorldView-3 / DESIS hyperspectral |
| nerf | [GOLD] NeRF Blender (Mildenhall ECCV 2020) / LLFF / DTU MVS dataset |
| neutron_diffraction | [BEST] ILL neutron diffraction data / SINQ PSI / ICSD CIF structures |
| neutron_tomo | [BEST] PSI NEUTRA dataset / ILL ICON neutron CT |
| nirs_brain | [BEST] fNIRS-BIDS benchmark / LABBRAIN fNIRS dataset / UCL Multimodal Imaging |
| ocean_color | [GOLD] NASA MODIS ocean color (oceancolor.gsfc.nasa.gov) / SeaWiFS dataset |
| oct | [GOLD] RETOUCH (Bogunovic IVCM 2019) / Duke OCT / OPTIMA retinal OCT |
| octa | [BEST] ROSE dataset (Ma TPAMI 2021) / CAVF OCTA benchmark |
| odt | [BEST] 2.5D DIC/ODT benchmark / Toulouse ODT dataset / TORCH benchmark |
| palm_storm | [GOLD] SMLM Challenge 2016 (smlmchallenge.net) / ThunderSTORM benchmark |
| panorama | [GOLD] SUN360 (Xiao CVPR 2012) / Laval HDR Panorama Dataset |
| particle_calorimetry | [GOLD] GEANT4 CaloChallenge 2022 (Fast Calorimeter Simulation Challenge) |
| passive_microwave | [BEST] AMSR2 / SSMIS Level-3 (NASA NSIDC) / GMI precipitation data |
| pet | [GOLD] TCIA-PET LIDC (Clark Sci. Data 2013) / OpenPET simulation data |
| pet_ct | [GOLD] TCIA PET-CT (The Cancer Imaging Archive) / MAASTRO PET-CT dataset |
| pet_mr | [BEST] MICCAI PET-MR challenge / BrainPET dataset / ADNI PET-MRI |
| phase_contrast | [BEST] CXLS phase contrast / APS phase contrast dataset / Siemens Fresnel |
| phase_retrieval | [BEST] CDI challenge benchmark / ptychography phase retrieval (Zenodo) |
| photoacoustic | [GOLD] MICCAI PATATO dataset / PAT-Public (ucl.ac.uk) / OADAT benchmark |
| photometric_stereo | [GOLD] DiLiGenT-MV (Ren IEEE TPAMI 2022) / CyclesPS benchmark |
| polarization | [BEST] AOLP dataset (Tyo Appl. Opt. 2006) / Polarization benchmark |
| polsar | [GOLD] UAVSAR (NASA JPL) / SIR-C / RADARSAT-2 PolSAR (MDA) |
| portal_imaging | [BEST] EPID benchmark / AAPM TG-58 portal imaging dataset |
| ptychography | [BEST] CDI ptychography benchmark (Zenodo) / CXLS/ALS ptychography data |
| radio_astronomy | [GOLD] LOFAR HBA survey / VLA FIRST (White ApJ 1997) / ALMA calibration |
| radio_interferometry | [BEST] MeerKAT MeerLICHT / VLBI imaging challenge 2022 (radiointerferometrychallenege.github.io) |
| raman_imaging | [GOLD] RRUFF Raman database (rruff.info) / NIST SRM Raman benchmark |
| sar | [GOLD] Sentinel-1 GRD (ESA Copernicus) / UAVSAR (NASA JPL) / ERS-2 |
| saxs | [BEST] cSAXS synchrotron data (PSI) / ALS SAXS dataset / ESRF BM26 |
| seismic_tomo | [BEST] IRIS SEED seismic (ds.iris.edu) / SEG-Y NCEDC dataset / Marmousi |
| sem | [GOLD] SEM-CIFA dataset / NIST SEM calibration / ZEISS SEM benchmark |
| sim | [GOLD] SIMbench (Culley Nature Methods 2018) / SMLM SIM benchmark |
| solar_imaging | [GOLD] SDO AIA (HEK, lmsal.com) / SOHO EIT / TRACE EUV solar archive |
| spc | [BEST] SPC simulation benchmark / Rice SPC dataset (Duarte Science 2008) |
| spect | [BEST] SIMIND simulation framework / GATE SPECT benchmark (OpenGATE) |
| spect_ct | [GOLD] TCIA SPECT-CT (The Cancer Imaging Archive) / Philips IQ-SPECT |
| spectral_ct | [GOLD] AAPM Spectral CT challenge / Medipix3 spectral CT dataset |
| spinning_disk | [BEST] Spinning disk benchmark / BBBC (Broad Bioimage Benchmark Collection) |
| sted | [GOLD] STED benchmark (Culley Nature Methods 2018) / Leica/Abberior data |
| stem | [BEST] AAEM STEM benchmark / EMPIAR STEM datasets / NIST STEM SRM |
| stm | [BEST] STM database (nanosurf.com) / NIST surface topography SRM |
| structured_light | [BEST] SL benchmark (Gupta CVPR 2012) / CAVE SL dataset |
| swi | [BEST] SWI benchmark / OpenNeuro SWI dataset (openneuro.org) |
| talbot_lau | [BEST] Munich Talbot-Lau grating data (TU Munich) / PSI grating CT |
| tem | [GOLD] EMPIAR TEM datasets (EBI) / JEOL benchmark / NIST TEM SRM |
| terahertz | [BEST] THz-TDS simulation benchmark / NIST THz spectroscopy database |
| three_photon | [BEST] 3PM simulation / Kleinfeld lab 3PM dataset (UCSD) |
| tirf | [BEST] TIRF benchmark (SMLM Challenge) / Cell-TIRF dataset |
| tof_camera | [GOLD] ETH3D (Schops CVPR 2017) / Middlebury 3D ToF / TUM RGB-D |
| two_photon | [GOLD] Allen Brain 2P-SCC (alleninstitute.org) / Carandini-Harris dataset |
| ultrasound | [GOLD] CAMUS (Leclerc IEEE TMI 2019) / EchoNet-Dynamic (Ouyang Nature 2020) |
| us_mri | [BEST] Ultrashort TE / ZTE MRI benchmark / PETRA dataset (Siemens) |
| waxs | [BEST] ESRF WAXS archive / ALS SAXS/WAXS / DLS SAXS data |
| weather_radar | [GOLD] NEXRAD WSR-88D (NOAA, ncdc.noaa.gov) / MetOffice C-band / OPERA |
| widefield | [BEST] BSDS500 / MitoCheck widefield (EMBL) / Broad BBBC benchmark |
| widefield_lowdose | [GOLD] CARE low-dose fluorescence (Weigert Nature Methods 2018) / BBBC |
| xfel_sfx | [BEST] CFEL SFX benchmark / LCLS SFX data (lcls.slac.stanford.edu) |
| xray_crystallography | [GOLD] PDB (Protein Data Bank, rcsb.org) / CCDC CSD / ICDD PDF-4+ |
| xray_ndt | [BEST] ASTM NDT E1000 / WoDT benchmark / Zeiss Xradia NDT dataset |
| xray_radiography | [GOLD] Chest X-ray14 (Wang CVPR 2017) / PadChest / CheXpert (Stanford) |
| xrf_imaging | [BEST] ESRF XRF imaging dataset / APS XRF benchmark |
| xrf_tomo | [BEST] XRF-CT benchmark (APS) / ESRF XRF-CT / Dls I18 dataset |

### ✅ Simulation-Standard Verified (SIM — no dominant public benchmark)

| Modality | Current Candidate | Action Needed |
|----------|-------------------|---------------|
| acoustic_emission | [SIM] AE simulation benchmark / EWGAE standards dataset | ✅ SIM-verified |
| acoustic_microscopy | [SIM] SAM synthetic benchmark (no dominant public dataset) | ✅ SIM-verified |
| atom_probe | [SIM] APT simulation benchmark (no dominant open dataset) | ✅ SIM-verified |
| bioluminescence_tomo | [SIM] BLT simulation benchmark (Ntziachristos Nature Methods 2010) | ✅ SIM-verified |
| brillouin | [SIM] Brillouin simulation benchmark / RRUFF spectral data | ✅ SIM-verified |
| cars | [SIM] CARS simulation benchmark / SRS hyperspectral data | ✅ SIM-verified |
| ct_fluorescence | [SIM] CT-FMT simulation benchmark / FLECT phantom data | ✅ SIM-verified |
| cup | [SIM] CUP (Compressed Ultrafast Photography) benchmark | ✅ SIM-verified |
| dot | [SIM] UCL DOT simulation benchmark / BabyBrain DOT data | ✅ SIM-verified |
| eddy_current | [SIM] EEDB NDT benchmark / Rolls-Royce ECT dataset | ✅ SIM-verified |
| electron_holography | [BEST] EMDB holography dataset / FZJ Juelich electron holography | ✅ SIM-verified |
| entangled_photon | [SIM] Quantum imaging simulation benchmark (no dominant open dataset) | ✅ SIM-verified |
| ghost_imaging | [SIM] Ghost imaging simulation benchmark / NIST quantum dataset | ✅ SIM-verified |
| gpr | [SIM] ISAP GPR benchmark / SFDB GPR dataset / IDS simulation data | ✅ SIM-verified |
| ism | [SIM] ISM simulation benchmark / Oxford ISM comparison data | ✅ SIM-verified |
| lucky_imaging | [SIM] Lucky imaging benchmark / Palomar speckle dataset (no dominant standard) | ✅ SIM-verified |
| mfm | [SIM] MFM simulation benchmark / NanoWorld MFM calibration data | ✅ SIM-verified |
| minflux | [SIM] MINFLUX simulation benchmark / Göttingen MINFLUX dataset | ✅ SIM-verified |
| muon_tomo | [SIM] Muon tomography simulation / CERN CMS muon data | ✅ SIM-verified |
| nsom | [SIM] NSOM simulation benchmark (no dominant open dataset) | ✅ SIM-verified |
| ocean_acoustic_tomo | [SIM] NOAA ocean acoustic data / SWEX simulation benchmark | ✅ SIM-verified |
| proton_radiography | [SIM] pCT collaboration dataset / FLASH proton CT simulation | ✅ SIM-verified |
| proton_therapy_img | [SIM] Proton CT simulation (TOPAS MC) / Onco-Sim benchmark | ✅ SIM-verified |
| pump_probe | [SIM] Ultrafast spectroscopy simulation / SLAC LCLS pump-probe data | ✅ SIM-verified |
| quantum_illumination | [SIM] Quantum imaging simulation (no dominant open dataset) | ✅ SIM-verified |
| shearography | [SIM] Shearography simulation benchmark (no dominant open dataset) | ✅ SIM-verified |
| shg | [SIM] SHG collagen benchmark / NLO microscopy public dataset | ✅ SIM-verified |
| sims | [SIM] SIMS surface database / IFM Stuttgart SIMS benchmark data | ✅ SIM-verified |
| sonar | [SIM] NOAA sonar archive / ARIS multibeam sonar benchmark | ✅ SIM-verified |
| srs | [SIM] SRS benchmark / coherent Raman spectral imaging dataset | ✅ SIM-verified |
| streak_camera | [SIM] Streak camera simulation benchmark (no dominant open dataset) | ✅ SIM-verified |
| ultrasonic_phased_array | [SIM] PAUT benchmark (ASNT) / NDT phased array Open-PAUT data | ✅ SIM-verified |

---

## Stage 3: GPU Algorithm Test Results

Tests run: 2026-03-11 | GPU: NVIDIA GTX 1660 Ti, CUDA 12.4 | PyTorch 2.6.0

| Modality | Solvers Tested | Best PSNR (dB) | Status |
|----------|---------------|----------------|--------|
| acoustic_emission | 4 | 20.2 | ✅ |
| acoustic_microscopy | 4 | 22.0 | ✅ |
| active_thermography | 4 | 7.2 | ✅ |
| adaptive_optics | 4 | 100.0 | ✅ |
| afm | 3 | 31.3 | ✅ |
| angiography | 4 | 12.9 | ✅ |
| asl_mri | 4 | 4.1 | ✅ |
| atom_probe | 4 | 40.2 | ✅ |
| bioluminescence_tomo | 4 | 13.3 | ✅ |
| brachytherapy_img | 4 | 25.2 | ✅ |
| brillouin | 4 | 35.8 | ✅ |
| cacti | 6 | 11.5 | ✅ |
| cars | 4 | 16.7 | ✅ |
| cassi | 4 | 13.6 | ✅ |
| cathodoluminescence | 4 | 28.9 | ✅ |
| cbct | 3 | 15.2 | ✅ |
| cest_mri | 4 | 32.1 | ✅ |
| ceus | 4 | 24.5 | ✅ |
| clem | 4 | 28.1 | ✅ |
| coded_exposure | 4 | 32.1 | ✅ |
| confocal_3d | 6 | 27.3 | ✅ |
| confocal_endomicroscopy | 4 | 34.0 | ✅ |
| confocal_livecell | 5 | 32.3 | ✅ |
| coronagraphy | 4 | 25.2 | ✅ |
| cryo_em | 5 | 19.2 | ✅ |
| cryo_et | 3 | 13.2 | ✅ |
| ct | 7 | 13.8 | ✅ |
| ct_fluorescence | 4 | 3.3 | ✅ |
| cup | 4 | 5.5 | ✅ |
| dark_field | 3 | 25.1 | ✅ |
| desi | 4 | 15.1 | ✅ |
| dexa | 4 | inf | ✅ |
| dic | 3 | 15.6 | ✅ |
| diffusion_mri | 4 | 11.3 | ✅ |
| digital_breast_tomo | 4 | 2.5 | ✅ |
| dna_paint | 3 | 28.5 | ✅ |
| doppler_ultrasound | 6 | 17.6 | ✅ |
| dot | 6 | 7.0 | ✅ |
| ebsd | 4 | 21.9 | ✅ |
| eddy_current | 4 | 22.9 | ✅ |
| edx_mapping | 4 | 22.0 | ✅ |
| eels | 5 | 25.2 | ✅ |
| eht_imaging | 4 | 11.4 | ✅ |
| elastography | 4 | 11.0 | ✅ |
| electron_diffraction | 4 | 42.0 | ✅ |
| electron_holography | 4 | 9.5 | ✅ |
| electron_tomography | 4 | 25.1 | ✅ |
| endoscopy | 3 | 11.8 | ✅ |
| entangled_photon | 4 | 31.8 | ✅ |
| event_camera | 4 | 7.3 | ✅ |
| expansion | 3 | 33.9 | ✅ |
| fib_sem | 3 | 28.1 | ✅ |
| flash_lidar | 4 | 4.3 | ✅ |
| flim | 5 | 36.9 | ✅ |
| fluoroscopy | 4 | 43.6 | ✅ |
| fmri | 4 | 4.9 | ✅ |
| fpm | 5 | 16.9 | ✅ |
| ftir_imaging | 4 | 34.6 | ✅ |
| fundus | 4 | 35.9 | ✅ |
| fwi | 4 | 8.7 | ✅ |
| gaussian_splatting | 5 | inf | ✅ |
| ghost_imaging | 4 | inf | ✅ |
| gpr | 4 | 10.6 | ✅ |
| gravitational_wave | 4 | 100.0 | ✅ |
| hdr_imaging | 4 | 36.8 | ✅ |
| holography | 5 | 14.9 | ✅ |
| hyperspectral_remote | 4 | 29.1 | ✅ |
| impedance_tomo | 4 | inf | ✅ |
| industrial_ct | 4 | 20.3 | ✅ |
| insar | 4 | 31.8 | ✅ |
| integral | 5 | 40.0 | ✅ |
| ism | 3 | 3.1 | ✅ |
| ivus | 4 | 19.8 | ✅ |
| lattice_lightsheet | 3 | 25.1 | ✅ |
| lensless | 5 | 11.9 | ✅ |
| libs | 4 | 26.5 | ✅ |
| lidar | 4 | 32.7 | ✅ |
| light_field | 5 | 27.3 | ✅ |
| lightsheet | 7 | 20.0 | ✅ |
| lucky_imaging | 4 | 29.6 | ✅ |
| machine_vision | 4 | 28.3 | ✅ |
| magnetic_particle | 4 | 26.5 | ✅ |
| maldi_msi | 4 | 27.1 | ✅ |
| mammography | 4 | 20.9 | ✅ |
| matrix | 5 | 22.0 | ✅ |
| mfm | 3 | 34.3 | ✅ |
| minflux | 3 | 29.5 | ✅ |
| mr_elastography | 4 | 6.0 | ✅ |
| mr_fingerprinting | 4 | 4.2 | ✅ |
| mra | 4 | 12.1 | ✅ |
| mri | 7 | 13.4 | ✅ |
| mrs | 4 | 4.3 | ✅ |
| multispectral_sat | 4 | 11.3 | ✅ |
| muon_tomo | 4 | 5.2 | ✅ |
| nerf | 4 | 29.0 | ✅ |
| neutron_diffraction | 4 | 8.5 | ✅ |
| neutron_tomo | 4 | 6.6 | ✅ |
| nirs_brain | 4 | 20.2 | ✅ |
| nsom | 3 | 22.3 | ✅ |
| ocean_acoustic_tomo | 4 | 26.6 | ✅ |
| ocean_color | 4 | 44.1 | ✅ |
| oct | 6 | 23.5 | ✅ |
| octa | 4 | 18.8 | ✅ |
| odt | 4 | 25.5 | ✅ |
| palm_storm | 3 | 32.4 | ✅ |
| panorama | 5 | 15.6 | ✅ |
| particle_calorimetry | 4 | 36.7 | ✅ |
| passive_microwave | 4 | 16.9 | ✅ |
| pet | 4 | 33.1 | ✅ |
| pet_ct | 4 | 13.0 | ✅ |
| pet_mr | 4 | 11.0 | ✅ |
| phase_contrast | 3 | 45.6 | ✅ |
| phase_retrieval | 5 | 12.6 | ✅ |
| photoacoustic | 5 | 19.1 | ✅ |
| photometric_stereo | 4 | 29.0 | ✅ |
| polarization | 4 | 15.8 | ✅ |
| polsar | 4 | 7.2 | ✅ |
| portal_imaging | 4 | 17.3 | ✅ |
| proton_radiography | 4 | 12.0 | ✅ |
| proton_therapy_img | 4 | 26.6 | ✅ |
| ptychography | 6 | 21.0 | ✅ |
| pump_probe | 4 | 18.2 | ✅ |
| quantum_illumination | 4 | 20.2 | ✅ |
| radio_astronomy | 4 | 37.3 | ✅ |
| radio_interferometry | 4 | 23.2 | ✅ |
| raman_imaging | 4 | 19.7 | ✅ |
| sar | 4 | 17.8 | ✅ |
| saxs | 4 | 8.4 | ✅ |
| seismic_tomo | 4 | 9.0 | ✅ |
| sem | 4 | 23.2 | ✅ |
| shearography | 4 | 13.2 | ✅ |
| shg | 3 | 23.0 | ✅ |
| sim | 6 | 21.6 | ✅ |
| sims | 4 | 20.5 | ✅ |
| solar_imaging | 4 | 28.4 | ✅ |
| sonar | 4 | 15.0 | ✅ |
| spc | 4 | 6.8 | ✅ |
| spect | 3 | 30.0 | ✅ |
| spect_ct | 4 | 11.4 | ✅ |
| spectral_ct | 4 | 12.3 | ✅ |
| spinning_disk | 3 | 30.6 | ✅ |
| srs | 4 | 29.1 | ✅ |
| sted | 3 | 25.0 | ✅ |
| stem | 4 | 31.0 | ✅ |
| stm | 3 | 23.3 | ✅ |
| streak_camera | 4 | 30.8 | ✅ |
| structured_light | 4 | 8.0 | ✅ |
| swi | 4 | 4.6 | ✅ |
| talbot_lau | 4 | 28.9 | ✅ |
| tem | 4 | 25.3 | ✅ |
| terahertz | 4 | 37.1 | ✅ |
| three_photon | 3 | 20.8 | ✅ |
| tirf | 4 | 31.2 | ✅ |
| tof_camera | 4 | 42.0 | ✅ |
| two_photon | 3 | 33.8 | ✅ |
| ultrasonic_phased_array | 4 | 30.8 | ✅ |
| ultrasound | 5 | 14.8 | ✅ |
| us_mri | 4 | 25.5 | ✅ |
| waxs | 4 | 20.6 | ✅ |
| weather_radar | 4 | 26.9 | ✅ |
| widefield | 5 | 25.0 | ✅ |
| widefield_lowdose | 3 | 29.0 | ✅ |
| xfel_sfx | 4 | 24.1 | ✅ |
| xray_crystallography | 4 | 22.4 | ✅ |
| xray_ndt | 4 | 16.7 | ✅ |
| xray_radiography | 4 | 26.3 | ✅ |
| xrf_imaging | 4 | 22.1 | ✅ |
| xrf_tomo | 4 | 15.6 | ✅ |

---

*Generated by scripts/build_state_v2.py — 2026-03-11*
