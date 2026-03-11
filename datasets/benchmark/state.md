# PWM Benchmark — Dataset & Pipeline State

Last updated: 2026-03-11 — 168 modalities

## Pipeline States

Each modality tracks 4 pipeline stages:

1. **Dataset** — public (≥10 samples), dev (20), hidden (20) created
2. **Benchmark** — modality page live at https://pwm.platformai.org/benchmark
3. **GPU Tests** — all YAML-defined solvers tested on GPU server
4. **SpecLab** — full reconstruction suite running on main server

Icons: ✅ done | 🔄 in progress | ❌ pending

---

## Quick Status Table

| Modality | Public Dataset | Stage 1: Dataset | Stage 2: Benchmark | Stage 3: GPU Tests | Stage 4: SpecLab |
|----------|---------------|------------------|-------------------|--------------------|-----------------|
| acoustic_emission | CAME (Component Analysis for Micro-Earthquake) / AEDB | ❌ | ❌ | ✅ 1 solvers, best=20.2 dB | ❌ |
| acoustic_microscopy | Zenodo AM benchmark / custom lab datasets | ❌ | ❌ | ✅ 1 solvers, best=10.0 dB | ❌ |
| active_thermography | IR-thermography IRT benchmark / EXTRACT dataset | ❌ | ❌ | ✅ 1 solvers, best=6.5 dB | ❌ |
| adaptive_optics | SciPy/WFS simulation data / SAXO+ datasets | ❌ | ❌ | ✅ 1 solvers, best=100.0 dB | ❌ |
| afm | OpenAFM benchmark / NIST AFM standard | ❌ | ❌ | ✅ 3 solvers, best=31.3 dB | ❌ |
| angiography | CTA/DSA datasets — VESSEL12 / CAT08 | ❌ | ❌ | ✅ 2 solvers, best=12.9 dB | ❌ |
| asl_mri | Human Connectome Project ASL / OpenNeuro | ❌ | ❌ | ✅ 1 solvers, best=2.7 dB | ❌ |
| atom_probe | IVAS dataset / FAU Erlangen dataset | ❌ | ❌ | ✅ 1 solvers, best=40.2 dB | ❌ |
| bioluminescence_tomo | BLT simulation benchmark / Caltech BLT | ❌ | ❌ | ✅ 1 solvers, best=13.3 dB | ❌ |
| brachytherapy_img | TG-43 phantom / AAPM TG-186 dataset | ❌ | ❌ | ✅ 1 solvers, best=20.5 dB | ❌ |
| brillouin | Zenodo Brillouin benchmark / stimulated data | ❌ | ❌ | ✅ 1 solvers, best=35.8 dB | ❌ |
| cacti | Six Scenes / DAVIS / CODED dataset (Hitomi et al. 2011) | ❌ | ❌ | ✅ 5 solvers, best=11.5 dB | ❌ |
| cars | CARS spectral benchmark / SRS-CARS dataset | ❌ | ❌ | ✅ 1 solvers, best=14.2 dB | ❌ |
| cassi | CAVE hyperspectral / KAIST scene dataset | ❌ | ❌ | ✅ 1 solvers, best=-5.3 dB | ❌ |
| cathodoluminescence | CL-EM benchmark / Zeiss CL dataset | ❌ | ❌ | ✅ 1 solvers, best=28.9 dB | ❌ |
| cbct | TCIA Head-Neck CBCT / CBCT-AAPM challenge | ❌ | ❌ | ✅ 3 solvers, best=15.2 dB | ❌ |
| cest_mri | CEST simulation benchmark / CEST MRI Challenge | ❌ | ❌ | ✅ 1 solvers, best=31.0 dB | ❌ |
| ceus | CAMUS ultrasound / MICCAI CEUS challenge | ❌ | ❌ | ✅ 1 solvers, best=24.5 dB | ❌ |
| clem | OpenOrganelle CLEM / CryoET Data Portal CLEM | ❌ | ❌ | ✅ 1 solvers, best=17.0 dB | ❌ |
| coded_exposure | Deblurring dataset (Levin 2009) / Kohler 2012 | ❌ | ❌ | ✅ 1 solvers, best=19.9 dB | ❌ |
| confocal_3d | OpenCell 3D confocal / HPA 3D confocal dataset | ❌ | ❌ | ✅ 6 solvers, best=27.3 dB | ❌ |
| confocal_endomicroscopy | UCL pCLE dataset / Mauna Kea nCLE data | ❌ | ❌ | ✅ 1 solvers, best=34.0 dB | ❌ |
| confocal_livecell | LiveCell dataset / CTC Cell Tracking Challenge | ❌ | ❌ | ✅ 5 solvers, best=32.3 dB | ❌ |
| coronagraphy | HST coronagraph archive / GPI GPIES dataset | ❌ | ❌ | ✅ 1 solvers, best=25.2 dB | ❌ |
| cryo_em | EMPIAR-10028 (TRPV1) / EMDB GroEL / SHREC19 | ❌ | ❌ | ✅ 2 solvers, best=19.2 dB | ❌ |
| cryo_et | SHREC 2021 / EMPIAR-10045 / IsoNet dataset | ❌ | ❌ | ✅ 3 solvers, best=13.2 dB | ❌ |
| ct | LoDoPaB-CT (Leuschner 2021, Zenodo 3384092) | ✅ | ❌ | ✅ 6 solvers, best=13.8 dB | ❌ |
| ct_fluorescence | CT-FMT simulation benchmark / FLECT dataset | ❌ | ❌ | ✅ 1 solvers, best=-37.6 dB | ❌ |
| cup | CUP benchmark / STAMP dataset | ❌ | ❌ | ✅ 1 solvers, best=-2.3 dB | ❌ |
| dark_field | Talbot-Lau dark-field benchmark / Munich DFI data | ❌ | ❌ | ✅ 3 solvers, best=25.1 dB | ❌ |
| desi | MetaboLights DESI dataset / METLIN-DESI | ❌ | ❌ | ✅ 1 solvers, best=15.1 dB | ❌ |
| dexa | OsteoArthritis Initiative (OAI) DXA / NHANES DXA | ❌ | ❌ | ✅ 1 solvers, best=9.5 dB | ❌ |
| dic | SciPy phase benchmark / ACPA DIC dataset | ❌ | ❌ | ✅ 3 solvers, best=15.6 dB | ❌ |
| diffusion_mri | Human Connectome Project dMRI / Sherbrooke-3T | ❌ | ❌ | ✅ 1 solvers, best=11.3 dB | ❌ |
| digital_breast_tomo | INBreast / VDM-100 DBT dataset | ❌ | ❌ | ✅ 1 solvers, best=-36.0 dB | ❌ |
| dna_paint | DNA-PAINT sim benchmark / Jungmann lab data | ❌ | ❌ | ✅ 3 solvers, best=28.5 dB | ❌ |
| doppler_ultrasound | PHANTOMNET / EchoNet-Dynamic / MICCAI Doppler | ❌ | ❌ | ✅ 3 solvers, best=17.6 dB | ❌ |
| dot | DOT simulation benchmark / UCL DOT dataset | ❌ | ❌ | ✅ 3 solvers, best=7.0 dB | ❌ |
| ebsd | DREAM.3D synthetic / Neper EBSD benchmark | ❌ | ❌ | ✅ 1 solvers, best=21.8 dB | ❌ |
| eddy_current | ECT benchmark / Rolls-Royce NDT dataset | ❌ | ❌ | ✅ 1 solvers, best=4.8 dB | ❌ |
| edx_mapping | NIST EDX SRM / Hyperspy EDX demo dataset | ❌ | ❌ | ✅ 2 solvers, best=22.0 dB | ❌ |
| eels | EELS.info database / Cornell EELS dataset | ❌ | ❌ | ✅ 1 solvers, best=24.6 dB | ❌ |
| eht_imaging | EHT 2019 M87 data / ngEHT simulated dataset | ❌ | ❌ | ✅ 1 solvers, best=11.3 dB | ❌ |
| elastography | MRE NIST phantom / MICCAI Elastography dataset | ❌ | ❌ | ✅ 1 solvers, best=5.7 dB | ❌ |
| electron_diffraction | CIF/ICSD simulation / RRUFF electron diffraction | ❌ | ❌ | ✅ 2 solvers, best=42.0 dB | ❌ |
| electron_holography | EMDB holo dataset / FZJ Juelich holography data | ❌ | ❌ | ✅ 2 solvers, best=9.5 dB | ❌ |
| electron_tomography | EMPIAR-10005 / EMPIAR-10045 / EMDB tilt series | ❌ | ❌ | ✅ 2 solvers, best=25.1 dB | ❌ |
| endoscopy | Kvasir-SEG / CholecT50 / Hyper-Kvasir | ❌ | ❌ | ✅ 3 solvers, best=11.8 dB | ❌ |
| entangled_photon | NIST quantum imaging data / simulation benchmark | ❌ | ❌ | ✅ 1 solvers, best=31.8 dB | ❌ |
| event_camera | DAVIS 240C / N-MNIST / MVSEC event dataset | ❌ | ❌ | ✅ 1 solvers, best=7.3 dB | ❌ |
| expansion | ExPath benchmark / Allen Institute ExM dataset | ❌ | ❌ | ✅ 3 solvers, best=33.9 dB | ❌ |
| fib_sem | OpenOrganelle FIB-SEM / Janelia FIB-SEM (H01) | ❌ | ❌ | ✅ 3 solvers, best=28.1 dB | ❌ |
| flash_lidar | KITTI LiDAR / Middlebury flash dataset | ❌ | ❌ | ✅ 1 solvers, best=4.3 dB | ❌ |
| flim | FLIM-FRET benchmark / FLUTE dataset | ❌ | ❌ | ✅ 2 solvers, best=30.7 dB | ❌ |
| fluoroscopy | TCIA Fluoroscopy / CVC-ClinicDB fluoroscopy | ❌ | ❌ | ✅ 2 solvers, best=43.5 dB | ❌ |
| fmri | Human Connectome Project fMRI / OpenNeuro BOLD | ❌ | ❌ | ✅ 1 solvers, best=4.9 dB | ❌ |
| fpm | FPM benchmark (Tian 2014) / UCB FPM dataset | ❌ | ❌ | ✅ 2 solvers, best=16.9 dB | ❌ |
| ftir_imaging | USGS spectral library / SFDB FTIR benchmark | ❌ | ❌ | ✅ 1 solvers, best=14.8 dB | ❌ |
| fundus | DRIVE / STARE / CHASE_DB1 fundus dataset | ❌ | ❌ | ✅ 4 solvers, best=35.9 dB | ❌ |
| fwi | OpenFWI / SEG-SALT / SEAM dataset | ❌ | ❌ | ✅ 1 solvers, best=8.7 dB | ❌ |
| gaussian_splatting | Tanks & Temples / Mip-NeRF360 / Blender NeRF | ❌ | ❌ | ✅ 5 solvers, best=inf dB | ❌ |
| ghost_imaging | Ghost imaging simulation / NIST quantum dataset | ❌ | ❌ | ✅ 1 solvers, best=6.6 dB | ❌ |
| gpr | GPR simulation benchmark / ISAP GPR dataset | ❌ | ❌ | ✅ 1 solvers, best=10.6 dB | ❌ |
| gravitational_wave | LIGO O3 public data / GW event catalog GWTC-3 | ❌ | ❌ | ✅ 1 solvers, best=100.0 dB | ❌ |
| hdr_imaging | HDR-DB (Fairchild) / HDREye / EMPA HDR dataset | ❌ | ❌ | ✅ 1 solvers, best=36.8 dB | ❌ |
| holography | HoloPy dataset / DHM benchmark dataset | ❌ | ❌ | ✅ 5 solvers, best=14.9 dB | ❌ |
| hyperspectral_remote | AVIRIS Indian Pines / ROSIS Pavia / HSRS-MT | ❌ | ❌ | ✅ 1 solvers, best=29.1 dB | ❌ |
| impedance_tomo | EIDORS simulation / Finnish EIT challenge | ❌ | ❌ | ✅ 1 solvers, best=11.2 dB | ❌ |
| industrial_ct | GCPD industrial CT / Zeiss Xradia dataset | ❌ | ❌ | ✅ 1 solvers, best=20.3 dB | ❌ |
| insar | Sentinel-1 SAR archive / COSAR benchmark | ❌ | ❌ | ✅ 1 solvers, best=31.8 dB | ❌ |
| integral | EPFL integral imaging / Stanford LF archive | ❌ | ❌ | ✅ 2 solvers, best=40.0 dB | ❌ |
| ism | ISM benchmark / Oxford ISM simulation data | ❌ | ❌ | ✅ 3 solvers, best=3.1 dB | ❌ |
| ivus | IVUS segmentation challenge / MICCAI 2011 IVUS | ❌ | ❌ | ✅ 1 solvers, best=19.8 dB | ❌ |
| lattice_lightsheet | Allen Institute lattice LS / Janelia LLC data | ❌ | ❌ | ✅ 3 solvers, best=25.1 dB | ❌ |
| lensless | DiffuserCam (Monakhova 2019) / PhlatCam dataset | ❌ | ❌ | ✅ 5 solvers, best=11.9 dB | ❌ |
| libs | LIBS spectral database / NIST LIBS database | ❌ | ❌ | ✅ 1 solvers, best=18.0 dB | ❌ |
| lidar | KITTI LiDAR / nuScenes LiDAR / SemanticKITTI | ❌ | ❌ | ✅ 1 solvers, best=32.6 dB | ❌ |
| light_field | Stanford Light Field Archive / INRIA LF dataset | ❌ | ❌ | ✅ 5 solvers, best=27.3 dB | ❌ |
| lightsheet | Allen Brain Atlas LS / Zebrafish SPIM dataset | ❌ | ❌ | ✅ 7 solvers, best=20.0 dB | ❌ |
| lucky_imaging | Lucky imaging benchmark / Palomar speckle data | ❌ | ❌ | ✅ 1 solvers, best=29.2 dB | ❌ |
| machine_vision | MVTec Anomaly Detection / BSDS500 | ❌ | ❌ | ✅ 1 solvers, best=26.5 dB | ❌ |
| magnetic_particle | MPI reconstruction challenge / OpenMPIData | ❌ | ❌ | ✅ 1 solvers, best=26.5 dB | ❌ |
| maldi_msi | MetaboLights MSI / METLIN imaging dataset | ❌ | ❌ | ✅ 1 solvers, best=26.3 dB | ❌ |
| mammography | VinDr-Mammo / CBIS-DDSM / INBreast | ❌ | ❌ | ✅ 2 solvers, best=20.9 dB | ❌ |
| matrix | Matrix completion benchmark / CASC dataset | ❌ | ❌ | ✅ 1 solvers, best=22.0 dB | ❌ |
| mfm | MFM simulation / NanoWorld MFM calibration | ❌ | ❌ | ✅ 3 solvers, best=34.3 dB | ❌ |
| minflux | MINFLUX benchmark / Gottingen MINFLUX data | ❌ | ❌ | ✅ 3 solvers, best=29.5 dB | ❌ |
| mr_elastography | MRE-NIST phantom / RSNA MRE challenge | ❌ | ❌ | ✅ 1 solvers, best=6.0 dB | ❌ |
| mr_fingerprinting | MRF simulation (Ma 2013) / CPMG relaxometry | ❌ | ❌ | ✅ 1 solvers, best=1.8 dB | ❌ |
| mra | TOF-MRA dataset / MICCAI vessel challenge | ❌ | ❌ | ✅ 1 solvers, best=12.1 dB | ❌ |
| mri | FastMRI (Zbontar 2018) multi-coil k-space | ❌ | ❌ | ✅ 3 solvers, best=13.0 dB | ❌ |
| mrs | MRSHUB dataset / BIG-PRESS simulation | ❌ | ❌ | ✅ 1 solvers, best=1.9 dB | ❌ |
| multispectral_sat | Sentinel-2 / WorldView-3 / DESIS hyperspectral | ❌ | ❌ | ✅ 1 solvers, best=10.8 dB | ❌ |
| muon_tomo | Muon tomography simulation / CERN muon data | ❌ | ❌ | ✅ 2 solvers, best=5.2 dB | ❌ |
| nerf | NeRF Blender / LLFF / DTU MVS dataset | ❌ | ❌ | ✅ 2 solvers, best=29.0 dB | ❌ |
| neutron_diffraction | SINQ / ILL neutron diffraction / CrysAlis | ❌ | ❌ | ✅ 1 solvers, best=8.5 dB | ❌ |
| neutron_tomo | ILL ICON neutron CT / PSI NEUTRA dataset | ❌ | ❌ | ✅ 2 solvers, best=4.3 dB | ❌ |
| nirs_brain | fNIRS benchmark / LABBRAIN fNIRS dataset | ❌ | ❌ | ✅ 1 solvers, best=14.5 dB | ❌ |
| nsom | NSOM simulation / Witec NSOM benchmark data | ❌ | ❌ | ✅ 3 solvers, best=22.3 dB | ❌ |
| ocean_acoustic_tomo | OAT simulation benchmark / NOAA acoustic data | ❌ | ❌ | ✅ 1 solvers, best=5.6 dB | ❌ |
| ocean_color | NASA MODIS ocean color / SeaWiFS dataset | ❌ | ❌ | ✅ 1 solvers, best=44.1 dB | ❌ |
| oct | RETOUCH / Duke OCT / OPTIMA OCT dataset | ❌ | ❌ | ✅ 6 solvers, best=23.5 dB | ❌ |
| octa | ROSE dataset / CAVF OCTA benchmark | ❌ | ❌ | ✅ 2 solvers, best=16.8 dB | ❌ |
| odt | 2D DIC/ODT benchmark / Toulouse ODT dataset | ❌ | ❌ | ✅ 1 solvers, best=25.5 dB | ❌ |
| palm_storm | SMLM Challenge 2016 / Thunderstorm benchmark | ❌ | ❌ | ✅ 3 solvers, best=32.4 dB | ❌ |
| panorama | SUN360 / LAVAL HDR Panorama dataset | ❌ | ❌ | ✅ 2 solvers, best=15.1 dB | ❌ |
| particle_calorimetry | GEANT4 simulation / CERN CaloChallenge 2022 | ❌ | ❌ | ✅ 1 solvers, best=36.2 dB | ❌ |
| passive_microwave | AMSR2 / SSMIS passive microwave NASA | ❌ | ❌ | ✅ 1 solvers, best=9.2 dB | ❌ |
| pet | TCIA-PET LIDC / OpenPET simulation data | ❌ | ❌ | ✅ 4 solvers, best=33.1 dB | ❌ |
| pet_ct | TCIA PET-CT / MAASTRO PET-CT dataset | ❌ | ❌ | ✅ 1 solvers, best=13.0 dB | ❌ |
| pet_mr | MICCAI PET-MR / BrainPET dataset | ❌ | ❌ | ✅ 1 solvers, best=11.0 dB | ❌ |
| phase_contrast | CXLS phase contrast / APS phase contrast data | ❌ | ❌ | ✅ 3 solvers, best=45.6 dB | ❌ |
| phase_retrieval | CDI benchmark / Phase retrieval algorithm tests | ❌ | ❌ | ✅ 2 solvers, best=12.6 dB | ❌ |
| photoacoustic | MICCAI PATATO / PAT-Public dataset | ❌ | ❌ | ✅ 2 solvers, best=19.1 dB | ❌ |
| photometric_stereo | DiLiGenT-MV / CyclesPS benchmark | ❌ | ❌ | ✅ 1 solvers, best=29.0 dB | ❌ |
| polarization | Polarization benchmark / AOLP dataset | ❌ | ❌ | ✅ 2 solvers, best=15.8 dB | ❌ |
| polsar | UAVSAR / SIR-C PolSAR / AIRSAR dataset | ❌ | ❌ | ✅ 1 solvers, best=3.5 dB | ❌ |
| portal_imaging | EPID benchmark / AAPM portal imaging TG58 | ❌ | ❌ | ✅ 1 solvers, best=10.5 dB | ❌ |
| proton_radiography | FLASH proton CT simulation / GSI proton data | ❌ | ❌ | ✅ 2 solvers, best=10.9 dB | ❌ |
| proton_therapy_img | Proton CT simulation / TOPAS benchmark | ❌ | ❌ | ✅ 1 solvers, best=17.8 dB | ❌ |
| ptychography | CDI / FXI ptychography benchmark / CXLS data | ❌ | ❌ | ✅ 3 solvers, best=21.0 dB | ❌ |
| pump_probe | Ultrafast spectroscopy simulation data | ❌ | ❌ | ✅ 1 solvers, best=18.2 dB | ❌ |
| quantum_illumination | Quantum imaging simulation benchmark | ❌ | ❌ | ✅ 1 solvers, best=20.2 dB | ❌ |
| radio_astronomy | LOFAR / VLA FIRST / ALMA calibration data | ❌ | ❌ | ✅ 1 solvers, best=16.1 dB | ❌ |
| radio_interferometry | MeerKAT / VLBI imaging challenge 2022 | ❌ | ❌ | ✅ 1 solvers, best=23.2 dB | ❌ |
| raman_imaging | RRUFF Raman database / NIST Raman benchmark | ❌ | ❌ | ✅ 1 solvers, best=14.1 dB | ❌ |
| sar | Sentinel-1 GRD / UAVSAR / ERS-2 SAR archive | ❌ | ❌ | ✅ 2 solvers, best=17.3 dB | ❌ |
| saxs | cSAXS synchrotron / ALS SAXS dataset | ❌ | ❌ | ✅ 1 solvers, best=8.4 dB | ❌ |
| seismic_tomo | IRIS seismic / SEG-Y NCEDC dataset | ❌ | ❌ | ✅ 1 solvers, best=9.0 dB | ❌ |
| sem | SEM-CIFA dataset / NIST SEM calibration | ❌ | ❌ | ✅ 2 solvers, best=23.2 dB | ❌ |
| shearography | Shearography simulation / LTI lab dataset | ❌ | ❌ | ✅ 1 solvers, best=8.0 dB | ❌ |
| shg | SHG microscopy benchmark / collagen data | ❌ | ❌ | ✅ 3 solvers, best=23.0 dB | ❌ |
| sim | SIMbench dataset / Allen SIM data | ❌ | ❌ | ✅ 3 solvers, best=21.6 dB | ❌ |
| sims | SIMS imaging database / IFM-Stuttgart SIMS | ❌ | ❌ | ✅ 1 solvers, best=20.5 dB | ❌ |
| solar_imaging | SDO AIA EUV / SOHO EIT / TRACE solar data | ❌ | ❌ | ✅ 1 solvers, best=28.4 dB | ❌ |
| sonar | NOAA sonar archive / ARIS multibeam sonar | ❌ | ❌ | ✅ 1 solvers, best=10.3 dB | ❌ |
| spc | SPC simulation benchmark / Rice SPC dataset | ❌ | ❌ | ✅ 1 solvers, best=-19.3 dB | ❌ |
| spect | SIMIND simulation / GATE SPECT benchmark | ❌ | ❌ | ✅ 3 solvers, best=30.0 dB | ❌ |
| spect_ct | TCIA SPECT-CT / Philips IQ-SPECT dataset | ❌ | ❌ | ✅ 1 solvers, best=11.4 dB | ❌ |
| spectral_ct | AAPM Spectral CT / Medipix spectral CT data | ❌ | ❌ | ✅ 1 solvers, best=12.3 dB | ❌ |
| spinning_disk | Spinning disk benchmark / Zeiss LSM dataset | ❌ | ❌ | ✅ 3 solvers, best=30.6 dB | ❌ |
| srs | SRS benchmark / coherent Raman dataset | ❌ | ❌ | ✅ 1 solvers, best=29.1 dB | ❌ |
| sted | STED benchmark / Leica Abberior STED data | ❌ | ❌ | ✅ 3 solvers, best=25.0 dB | ❌ |
| stem | AAEM STEM benchmark / NIST STEM dataset | ❌ | ❌ | ✅ 2 solvers, best=31.0 dB | ❌ |
| stm | STM database / NIST surface topography data | ❌ | ❌ | ✅ 3 solvers, best=23.3 dB | ❌ |
| streak_camera | Streak camera simulation benchmark | ❌ | ❌ | ✅ 1 solvers, best=14.3 dB | ❌ |
| structured_light | SL benchmark (Gupta 2012) / CAVE SL dataset | ❌ | ❌ | ✅ 1 solvers, best=8.0 dB | ❌ |
| swi | SWI benchmark / OpenNeuro SWI dataset | ❌ | ❌ | ✅ 1 solvers, best=1.9 dB | ❌ |
| talbot_lau | Munich Talbot-Lau grating dataset / PSI data | ❌ | ❌ | ✅ 1 solvers, best=6.6 dB | ❌ |
| tem | EMPIAR TEM benchmark / JEOL TEM data | ❌ | ❌ | ✅ 1 solvers, best=25.3 dB | ❌ |
| terahertz | THz-TDS benchmark / NIST THz dataset | ❌ | ❌ | ✅ 1 solvers, best=37.1 dB | ❌ |
| three_photon | 3PM simulation / Kleinfeld lab 3PM dataset | ❌ | ❌ | ✅ 3 solvers, best=20.8 dB | ❌ |
| tirf | TIRF benchmark / Cell-TIRF dataset | ❌ | ❌ | ✅ 2 solvers, best=31.2 dB | ❌ |
| tof_camera | ETH3D / Middlebury 3D ToF dataset | ❌ | ❌ | ✅ 1 solvers, best=42.0 dB | ❌ |
| two_photon | Allen Brain 2P / Carandini-Harris 2P dataset | ❌ | ❌ | ✅ 3 solvers, best=33.8 dB | ❌ |
| ultrasonic_phased_array | PAUT benchmark / NDT phased array dataset | ❌ | ❌ | ✅ 1 solvers, best=29.6 dB | ❌ |
| ultrasound | CAMUS / Echonet-Dynamic / CARDIAC US dataset | ❌ | ❌ | ✅ 2 solvers, best=14.6 dB | ❌ |
| us_mri | Ultrashort TE MRI benchmark / PETRA dataset | ❌ | ❌ | ✅ 1 solvers, best=7.6 dB | ❌ |
| waxs | SAXS/WAXS synchrotron / ESRF WAXS archive | ❌ | ❌ | ✅ 1 solvers, best=20.6 dB | ❌ |
| weather_radar | NEXRAD WSR-88D / MetOffice C-band radar data | ❌ | ❌ | ✅ 1 solvers, best=26.9 dB | ❌ |
| widefield | BSDS / MitoCheck widefield benchmark | ❌ | ❌ | ✅ 5 solvers, best=25.0 dB | ❌ |
| widefield_lowdose | Low-dose fluorescence benchmark / CARE dataset | ❌ | ❌ | ✅ 3 solvers, best=29.0 dB | ❌ |
| xfel_sfx | CFEL SFX benchmark / LCLS SFX dataset | ❌ | ❌ | ✅ 1 solvers, best=24.1 dB | ❌ |
| xray_crystallography | CIF / PDB (Protein Data Bank) / CSD dataset | ❌ | ❌ | ✅ 1 solvers, best=22.4 dB | ❌ |
| xray_ndt | ASTM NDT benchmark / Zeiss Xradia NDT data | ❌ | ❌ | ✅ 1 solvers, best=16.7 dB | ❌ |
| xray_radiography | RSNA Bone Age / Chest X-ray14 / PadChest | ❌ | ❌ | ✅ 2 solvers, best=26.3 dB | ❌ |
| xrf_imaging | XRF benchmark / ESRF XRF dataset | ❌ | ❌ | ✅ 1 solvers, best=22.1 dB | ❌ |
| xrf_tomo | XRF-CT benchmark / APS XRF-CT dataset | ❌ | ❌ | ✅ 1 solvers, best=15.6 dB | ❌ |

**Summary:** 1/168 datasets done | 168/168 GPU tests done

---

## CT Dataset (Reference Implementation)

- Public: 11 samples (LoDoPaB-CT Shepp-Logan fallback — need real zips)
- Dev: 20 samples
- Hidden: 20 samples
- Structure: per-sample dirs with groundtruth.npy, measurement.npy, images/
- GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ct/

**NOTE:** Public tier currently uses Shepp-Logan synthetic fallback.
To use real LoDoPaB-CT data, download to `datasets/benchmark/ct/lodopab_src/`:
```bash
wget 'https://zenodo.org/api/records/3384092/files/ground_truth_test.zip/content' \
     -O datasets/benchmark/ct/lodopab_src/ground_truth_test.zip
python datasets/benchmark/ct/generate_dataset.py
```

---

## Modalities Needing Datasets (167 pending)

All modalities except CT need public/dev/hidden datasets generated.
Dataset generation scripts: `scripts/generate_batch{1-12}_datasets.py`

| Modality | Public Dataset (Canonical) | Notes |
|----------|---------------------------|-------|
| acoustic_emission | CAME (Component Analysis for Micro-Earthquake) / AEDB | needs generation |
| acoustic_microscopy | Zenodo AM benchmark / custom lab datasets | needs generation |
| active_thermography | IR-thermography IRT benchmark / EXTRACT dataset | needs generation |
| adaptive_optics | SciPy/WFS simulation data / SAXO+ datasets | needs generation |
| afm | OpenAFM benchmark / NIST AFM standard | needs generation |
| angiography | CTA/DSA datasets — VESSEL12 / CAT08 | needs generation |
| asl_mri | Human Connectome Project ASL / OpenNeuro | needs generation |
| atom_probe | IVAS dataset / FAU Erlangen dataset | needs generation |
| bioluminescence_tomo | BLT simulation benchmark / Caltech BLT | needs generation |
| brachytherapy_img | TG-43 phantom / AAPM TG-186 dataset | needs generation |
| brillouin | Zenodo Brillouin benchmark / stimulated data | needs generation |
| cacti | Six Scenes / DAVIS / CODED dataset (Hitomi et al. 2011) | needs generation |
| cars | CARS spectral benchmark / SRS-CARS dataset | needs generation |
| cassi | CAVE hyperspectral / KAIST scene dataset | needs generation |
| cathodoluminescence | CL-EM benchmark / Zeiss CL dataset | needs generation |
| cbct | TCIA Head-Neck CBCT / CBCT-AAPM challenge | needs generation |
| cest_mri | CEST simulation benchmark / CEST MRI Challenge | needs generation |
| ceus | CAMUS ultrasound / MICCAI CEUS challenge | needs generation |
| clem | OpenOrganelle CLEM / CryoET Data Portal CLEM | needs generation |
| coded_exposure | Deblurring dataset (Levin 2009) / Kohler 2012 | needs generation |
| confocal_3d | OpenCell 3D confocal / HPA 3D confocal dataset | needs generation |
| confocal_endomicroscopy | UCL pCLE dataset / Mauna Kea nCLE data | needs generation |
| confocal_livecell | LiveCell dataset / CTC Cell Tracking Challenge | needs generation |
| coronagraphy | HST coronagraph archive / GPI GPIES dataset | needs generation |
| cryo_em | EMPIAR-10028 (TRPV1) / EMDB GroEL / SHREC19 | needs generation |
| cryo_et | SHREC 2021 / EMPIAR-10045 / IsoNet dataset | needs generation |
| ct_fluorescence | CT-FMT simulation benchmark / FLECT dataset | needs generation |
| cup | CUP benchmark / STAMP dataset | needs generation |
| dark_field | Talbot-Lau dark-field benchmark / Munich DFI data | needs generation |
| desi | MetaboLights DESI dataset / METLIN-DESI | needs generation |
| dexa | OsteoArthritis Initiative (OAI) DXA / NHANES DXA | needs generation |
| dic | SciPy phase benchmark / ACPA DIC dataset | needs generation |
| diffusion_mri | Human Connectome Project dMRI / Sherbrooke-3T | needs generation |
| digital_breast_tomo | INBreast / VDM-100 DBT dataset | needs generation |
| dna_paint | DNA-PAINT sim benchmark / Jungmann lab data | needs generation |
| doppler_ultrasound | PHANTOMNET / EchoNet-Dynamic / MICCAI Doppler | needs generation |
| dot | DOT simulation benchmark / UCL DOT dataset | needs generation |
| ebsd | DREAM.3D synthetic / Neper EBSD benchmark | needs generation |
| eddy_current | ECT benchmark / Rolls-Royce NDT dataset | needs generation |
| edx_mapping | NIST EDX SRM / Hyperspy EDX demo dataset | needs generation |
| eels | EELS.info database / Cornell EELS dataset | needs generation |
| eht_imaging | EHT 2019 M87 data / ngEHT simulated dataset | needs generation |
| elastography | MRE NIST phantom / MICCAI Elastography dataset | needs generation |
| electron_diffraction | CIF/ICSD simulation / RRUFF electron diffraction | needs generation |
| electron_holography | EMDB holo dataset / FZJ Juelich holography data | needs generation |
| electron_tomography | EMPIAR-10005 / EMPIAR-10045 / EMDB tilt series | needs generation |
| endoscopy | Kvasir-SEG / CholecT50 / Hyper-Kvasir | needs generation |
| entangled_photon | NIST quantum imaging data / simulation benchmark | needs generation |
| event_camera | DAVIS 240C / N-MNIST / MVSEC event dataset | needs generation |
| expansion | ExPath benchmark / Allen Institute ExM dataset | needs generation |
| fib_sem | OpenOrganelle FIB-SEM / Janelia FIB-SEM (H01) | needs generation |
| flash_lidar | KITTI LiDAR / Middlebury flash dataset | needs generation |
| flim | FLIM-FRET benchmark / FLUTE dataset | needs generation |
| fluoroscopy | TCIA Fluoroscopy / CVC-ClinicDB fluoroscopy | needs generation |
| fmri | Human Connectome Project fMRI / OpenNeuro BOLD | needs generation |
| fpm | FPM benchmark (Tian 2014) / UCB FPM dataset | needs generation |
| ftir_imaging | USGS spectral library / SFDB FTIR benchmark | needs generation |
| fundus | DRIVE / STARE / CHASE_DB1 fundus dataset | needs generation |
| fwi | OpenFWI / SEG-SALT / SEAM dataset | needs generation |
| gaussian_splatting | Tanks & Temples / Mip-NeRF360 / Blender NeRF | needs generation |
| ghost_imaging | Ghost imaging simulation / NIST quantum dataset | needs generation |
| gpr | GPR simulation benchmark / ISAP GPR dataset | needs generation |
| gravitational_wave | LIGO O3 public data / GW event catalog GWTC-3 | needs generation |
| hdr_imaging | HDR-DB (Fairchild) / HDREye / EMPA HDR dataset | needs generation |
| holography | HoloPy dataset / DHM benchmark dataset | needs generation |
| hyperspectral_remote | AVIRIS Indian Pines / ROSIS Pavia / HSRS-MT | needs generation |
| impedance_tomo | EIDORS simulation / Finnish EIT challenge | needs generation |
| industrial_ct | GCPD industrial CT / Zeiss Xradia dataset | needs generation |
| insar | Sentinel-1 SAR archive / COSAR benchmark | needs generation |
| integral | EPFL integral imaging / Stanford LF archive | needs generation |
| ism | ISM benchmark / Oxford ISM simulation data | needs generation |
| ivus | IVUS segmentation challenge / MICCAI 2011 IVUS | needs generation |
| lattice_lightsheet | Allen Institute lattice LS / Janelia LLC data | needs generation |
| lensless | DiffuserCam (Monakhova 2019) / PhlatCam dataset | needs generation |
| libs | LIBS spectral database / NIST LIBS database | needs generation |
| lidar | KITTI LiDAR / nuScenes LiDAR / SemanticKITTI | needs generation |
| light_field | Stanford Light Field Archive / INRIA LF dataset | needs generation |
| lightsheet | Allen Brain Atlas LS / Zebrafish SPIM dataset | needs generation |
| lucky_imaging | Lucky imaging benchmark / Palomar speckle data | needs generation |
| machine_vision | MVTec Anomaly Detection / BSDS500 | needs generation |
| magnetic_particle | MPI reconstruction challenge / OpenMPIData | needs generation |
| maldi_msi | MetaboLights MSI / METLIN imaging dataset | needs generation |
| mammography | VinDr-Mammo / CBIS-DDSM / INBreast | needs generation |
| matrix | Matrix completion benchmark / CASC dataset | needs generation |
| mfm | MFM simulation / NanoWorld MFM calibration | needs generation |
| minflux | MINFLUX benchmark / Gottingen MINFLUX data | needs generation |
| mr_elastography | MRE-NIST phantom / RSNA MRE challenge | needs generation |
| mr_fingerprinting | MRF simulation (Ma 2013) / CPMG relaxometry | needs generation |
| mra | TOF-MRA dataset / MICCAI vessel challenge | needs generation |
| mri | FastMRI (Zbontar 2018) multi-coil k-space | needs generation |
| mrs | MRSHUB dataset / BIG-PRESS simulation | needs generation |
| multispectral_sat | Sentinel-2 / WorldView-3 / DESIS hyperspectral | needs generation |
| muon_tomo | Muon tomography simulation / CERN muon data | needs generation |
| nerf | NeRF Blender / LLFF / DTU MVS dataset | needs generation |
| neutron_diffraction | SINQ / ILL neutron diffraction / CrysAlis | needs generation |
| neutron_tomo | ILL ICON neutron CT / PSI NEUTRA dataset | needs generation |
| nirs_brain | fNIRS benchmark / LABBRAIN fNIRS dataset | needs generation |
| nsom | NSOM simulation / Witec NSOM benchmark data | needs generation |
| ocean_acoustic_tomo | OAT simulation benchmark / NOAA acoustic data | needs generation |
| ocean_color | NASA MODIS ocean color / SeaWiFS dataset | needs generation |
| oct | RETOUCH / Duke OCT / OPTIMA OCT dataset | needs generation |
| octa | ROSE dataset / CAVF OCTA benchmark | needs generation |
| odt | 2D DIC/ODT benchmark / Toulouse ODT dataset | needs generation |
| palm_storm | SMLM Challenge 2016 / Thunderstorm benchmark | needs generation |
| panorama | SUN360 / LAVAL HDR Panorama dataset | needs generation |
| particle_calorimetry | GEANT4 simulation / CERN CaloChallenge 2022 | needs generation |
| passive_microwave | AMSR2 / SSMIS passive microwave NASA | needs generation |
| pet | TCIA-PET LIDC / OpenPET simulation data | needs generation |
| pet_ct | TCIA PET-CT / MAASTRO PET-CT dataset | needs generation |
| pet_mr | MICCAI PET-MR / BrainPET dataset | needs generation |
| phase_contrast | CXLS phase contrast / APS phase contrast data | needs generation |
| phase_retrieval | CDI benchmark / Phase retrieval algorithm tests | needs generation |
| photoacoustic | MICCAI PATATO / PAT-Public dataset | needs generation |
| photometric_stereo | DiLiGenT-MV / CyclesPS benchmark | needs generation |
| polarization | Polarization benchmark / AOLP dataset | needs generation |
| polsar | UAVSAR / SIR-C PolSAR / AIRSAR dataset | needs generation |
| portal_imaging | EPID benchmark / AAPM portal imaging TG58 | needs generation |
| proton_radiography | FLASH proton CT simulation / GSI proton data | needs generation |
| proton_therapy_img | Proton CT simulation / TOPAS benchmark | needs generation |
| ptychography | CDI / FXI ptychography benchmark / CXLS data | needs generation |
| pump_probe | Ultrafast spectroscopy simulation data | needs generation |
| quantum_illumination | Quantum imaging simulation benchmark | needs generation |
| radio_astronomy | LOFAR / VLA FIRST / ALMA calibration data | needs generation |
| radio_interferometry | MeerKAT / VLBI imaging challenge 2022 | needs generation |
| raman_imaging | RRUFF Raman database / NIST Raman benchmark | needs generation |
| sar | Sentinel-1 GRD / UAVSAR / ERS-2 SAR archive | needs generation |
| saxs | cSAXS synchrotron / ALS SAXS dataset | needs generation |
| seismic_tomo | IRIS seismic / SEG-Y NCEDC dataset | needs generation |
| sem | SEM-CIFA dataset / NIST SEM calibration | needs generation |
| shearography | Shearography simulation / LTI lab dataset | needs generation |
| shg | SHG microscopy benchmark / collagen data | needs generation |
| sim | SIMbench dataset / Allen SIM data | needs generation |
| sims | SIMS imaging database / IFM-Stuttgart SIMS | needs generation |
| solar_imaging | SDO AIA EUV / SOHO EIT / TRACE solar data | needs generation |
| sonar | NOAA sonar archive / ARIS multibeam sonar | needs generation |
| spc | SPC simulation benchmark / Rice SPC dataset | needs generation |
| spect | SIMIND simulation / GATE SPECT benchmark | needs generation |
| spect_ct | TCIA SPECT-CT / Philips IQ-SPECT dataset | needs generation |
| spectral_ct | AAPM Spectral CT / Medipix spectral CT data | needs generation |
| spinning_disk | Spinning disk benchmark / Zeiss LSM dataset | needs generation |
| srs | SRS benchmark / coherent Raman dataset | needs generation |
| sted | STED benchmark / Leica Abberior STED data | needs generation |
| stem | AAEM STEM benchmark / NIST STEM dataset | needs generation |
| stm | STM database / NIST surface topography data | needs generation |
| streak_camera | Streak camera simulation benchmark | needs generation |
| structured_light | SL benchmark (Gupta 2012) / CAVE SL dataset | needs generation |
| swi | SWI benchmark / OpenNeuro SWI dataset | needs generation |
| talbot_lau | Munich Talbot-Lau grating dataset / PSI data | needs generation |
| tem | EMPIAR TEM benchmark / JEOL TEM data | needs generation |
| terahertz | THz-TDS benchmark / NIST THz dataset | needs generation |
| three_photon | 3PM simulation / Kleinfeld lab 3PM dataset | needs generation |
| tirf | TIRF benchmark / Cell-TIRF dataset | needs generation |
| tof_camera | ETH3D / Middlebury 3D ToF dataset | needs generation |
| two_photon | Allen Brain 2P / Carandini-Harris 2P dataset | needs generation |
| ultrasonic_phased_array | PAUT benchmark / NDT phased array dataset | needs generation |
| ultrasound | CAMUS / Echonet-Dynamic / CARDIAC US dataset | needs generation |
| us_mri | Ultrashort TE MRI benchmark / PETRA dataset | needs generation |
| waxs | SAXS/WAXS synchrotron / ESRF WAXS archive | needs generation |
| weather_radar | NEXRAD WSR-88D / MetOffice C-band radar data | needs generation |
| widefield | BSDS / MitoCheck widefield benchmark | needs generation |
| widefield_lowdose | Low-dose fluorescence benchmark / CARE dataset | needs generation |
| xfel_sfx | CFEL SFX benchmark / LCLS SFX dataset | needs generation |
| xray_crystallography | CIF / PDB (Protein Data Bank) / CSD dataset | needs generation |
| xray_ndt | ASTM NDT benchmark / Zeiss Xradia NDT data | needs generation |
| xray_radiography | RSNA Bone Age / Chest X-ray14 / PadChest | needs generation |
| xrf_imaging | XRF benchmark / ESRF XRF dataset | needs generation |
| xrf_tomo | XRF-CT benchmark / APS XRF-CT dataset | needs generation |

---

## GPU Algorithm Test Results

Tests run: 2026-03-11 | GPU: NVIDIA GTX 1660 Ti, CUDA 12.4

| Modality | Solvers Completed | Best PSNR (dB) |
|----------|------------------|---------------|
| acoustic_emission | 1 | 20.2 |
| acoustic_microscopy | 1 | 10.0 |
| active_thermography | 1 | 6.5 |
| adaptive_optics | 1 | 100.0 |
| afm | 3 | 31.3 |
| angiography | 2 | 12.9 |
| asl_mri | 1 | 2.7 |
| atom_probe | 1 | 40.2 |
| bioluminescence_tomo | 1 | 13.3 |
| brachytherapy_img | 1 | 20.5 |
| brillouin | 1 | 35.8 |
| cacti | 5 | 11.5 |
| cars | 1 | 14.2 |
| cassi | 1 | -5.3 |
| cathodoluminescence | 1 | 28.9 |
| cbct | 3 | 15.2 |
| cest_mri | 1 | 31.0 |
| ceus | 1 | 24.5 |
| clem | 1 | 17.0 |
| coded_exposure | 1 | 19.9 |
| confocal_3d | 6 | 27.3 |
| confocal_endomicroscopy | 1 | 34.0 |
| confocal_livecell | 5 | 32.3 |
| coronagraphy | 1 | 25.2 |
| cryo_em | 2 | 19.2 |
| cryo_et | 3 | 13.2 |
| ct | 6 | 13.8 |
| ct_fluorescence | 1 | -37.6 |
| cup | 1 | -2.3 |
| dark_field | 3 | 25.1 |
| desi | 1 | 15.1 |
| dexa | 1 | 9.5 |
| dic | 3 | 15.6 |
| diffusion_mri | 1 | 11.3 |
| digital_breast_tomo | 1 | -36.0 |
| dna_paint | 3 | 28.5 |
| doppler_ultrasound | 3 | 17.6 |
| dot | 3 | 7.0 |
| ebsd | 1 | 21.8 |
| eddy_current | 1 | 4.8 |
| edx_mapping | 2 | 22.0 |
| eels | 1 | 24.6 |
| eht_imaging | 1 | 11.3 |
| elastography | 1 | 5.7 |
| electron_diffraction | 2 | 42.0 |
| electron_holography | 2 | 9.5 |
| electron_tomography | 2 | 25.1 |
| endoscopy | 3 | 11.8 |
| entangled_photon | 1 | 31.8 |
| event_camera | 1 | 7.3 |
| expansion | 3 | 33.9 |
| fib_sem | 3 | 28.1 |
| flash_lidar | 1 | 4.3 |
| flim | 2 | 30.7 |
| fluoroscopy | 2 | 43.5 |
| fmri | 1 | 4.9 |
| fpm | 2 | 16.9 |
| ftir_imaging | 1 | 14.8 |
| fundus | 4 | 35.9 |
| fwi | 1 | 8.7 |
| gaussian_splatting | 5 | inf |
| ghost_imaging | 1 | 6.6 |
| gpr | 1 | 10.6 |
| gravitational_wave | 1 | 100.0 |
| hdr_imaging | 1 | 36.8 |
| holography | 5 | 14.9 |
| hyperspectral_remote | 1 | 29.1 |
| impedance_tomo | 1 | 11.2 |
| industrial_ct | 1 | 20.3 |
| insar | 1 | 31.8 |
| integral | 2 | 40.0 |
| ism | 3 | 3.1 |
| ivus | 1 | 19.8 |
| lattice_lightsheet | 3 | 25.1 |
| lensless | 5 | 11.9 |
| libs | 1 | 18.0 |
| lidar | 1 | 32.6 |
| light_field | 5 | 27.3 |
| lightsheet | 7 | 20.0 |
| lucky_imaging | 1 | 29.2 |
| machine_vision | 1 | 26.5 |
| magnetic_particle | 1 | 26.5 |
| maldi_msi | 1 | 26.3 |
| mammography | 2 | 20.9 |
| matrix | 1 | 22.0 |
| mfm | 3 | 34.3 |
| minflux | 3 | 29.5 |
| mr_elastography | 1 | 6.0 |
| mr_fingerprinting | 1 | 1.8 |
| mra | 1 | 12.1 |
| mri | 3 | 13.0 |
| mrs | 1 | 1.9 |
| multispectral_sat | 1 | 10.8 |
| muon_tomo | 2 | 5.2 |
| nerf | 2 | 29.0 |
| neutron_diffraction | 1 | 8.5 |
| neutron_tomo | 2 | 4.3 |
| nirs_brain | 1 | 14.5 |
| nsom | 3 | 22.3 |
| ocean_acoustic_tomo | 1 | 5.6 |
| ocean_color | 1 | 44.1 |
| oct | 6 | 23.5 |
| octa | 2 | 16.8 |
| odt | 1 | 25.5 |
| palm_storm | 3 | 32.4 |
| panorama | 2 | 15.1 |
| particle_calorimetry | 1 | 36.2 |
| passive_microwave | 1 | 9.2 |
| pet | 4 | 33.1 |
| pet_ct | 1 | 13.0 |
| pet_mr | 1 | 11.0 |
| phase_contrast | 3 | 45.6 |
| phase_retrieval | 2 | 12.6 |
| photoacoustic | 2 | 19.1 |
| photometric_stereo | 1 | 29.0 |
| polarization | 2 | 15.8 |
| polsar | 1 | 3.5 |
| portal_imaging | 1 | 10.5 |
| proton_radiography | 2 | 10.9 |
| proton_therapy_img | 1 | 17.8 |
| ptychography | 3 | 21.0 |
| pump_probe | 1 | 18.2 |
| quantum_illumination | 1 | 20.2 |
| radio_astronomy | 1 | 16.1 |
| radio_interferometry | 1 | 23.2 |
| raman_imaging | 1 | 14.1 |
| sar | 2 | 17.3 |
| saxs | 1 | 8.4 |
| seismic_tomo | 1 | 9.0 |
| sem | 2 | 23.2 |
| shearography | 1 | 8.0 |
| shg | 3 | 23.0 |
| sim | 3 | 21.6 |
| sims | 1 | 20.5 |
| solar_imaging | 1 | 28.4 |
| sonar | 1 | 10.3 |
| spc | 1 | -19.3 |
| spect | 3 | 30.0 |
| spect_ct | 1 | 11.4 |
| spectral_ct | 1 | 12.3 |
| spinning_disk | 3 | 30.6 |
| srs | 1 | 29.1 |
| sted | 3 | 25.0 |
| stem | 2 | 31.0 |
| stm | 3 | 23.3 |
| streak_camera | 1 | 14.3 |
| structured_light | 1 | 8.0 |
| swi | 1 | 1.9 |
| talbot_lau | 1 | 6.6 |
| tem | 1 | 25.3 |
| terahertz | 1 | 37.1 |
| three_photon | 3 | 20.8 |
| tirf | 2 | 31.2 |
| tof_camera | 1 | 42.0 |
| two_photon | 3 | 33.8 |
| ultrasonic_phased_array | 1 | 29.6 |
| ultrasound | 2 | 14.6 |
| us_mri | 1 | 7.6 |
| waxs | 1 | 20.6 |
| weather_radar | 1 | 26.9 |
| widefield | 5 | 25.0 |
| widefield_lowdose | 3 | 29.0 |
| xfel_sfx | 1 | 24.1 |
| xray_crystallography | 1 | 22.4 |
| xray_ndt | 1 | 16.7 |
| xray_radiography | 2 | 26.3 |
| xrf_imaging | 1 | 22.1 |
| xrf_tomo | 1 | 15.6 |

---

*Generated by scripts/build_state_v2.py*
