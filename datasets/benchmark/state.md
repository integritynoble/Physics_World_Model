# Benchmark Dataset & Algorithm Test State

Last updated: 2026-03-11 — 168/168 modalities audited

## Status Legend
- **Dataset**: done = public/dev/hidden tiers exist with spec.json and true_spec.json
- **Samples OK**: whether sample counts meet requirements (public >= 10, dev >= 20, hidden >= 20)
- **Public Source**: the standard dataset / data origin used for public tier
- **Source Verified**: whether the public source has been reviewed and confirmed as widely-accepted
- **Algorithm Test**: algorithm testing status (done / pending / in-progress)
- **Speclab**: speclab integration status

## Quick Status Table

| # | Modality | Dataset | Pub | Dev | Hid | Samples OK | Public Source | Source Verified | Algorithm Test | Speclab |
|---|----------|---------|-----|-----|-----|------------|--------------|-----------------|---------------|---------|
| 1 | acoustic_emission | done | 6 | 6 | 6 | NO | Simulated / synthetic | pending | in-progress | pending |
| 2 | acoustic_microscopy | done | 6 | 6 | 6 | NO | Simulated / synthetic | pending | pending | pending |
| 3 | active_thermography | done | 6 | 6 | 6 | NO | Simulated / synthetic | pending | pending | pending |
| 4 | adaptive_optics | done | 6 | 6 | 6 | NO | Simulated AO correction | pending | pending | pending |
| 5 | afm | done | 6 | 6 | 6 | NO | Simulated AFM topography | pending | pending | pending |
| 6 | angiography | done | 6 | 6 | 6 | NO | Simulated DSA vascular | pending | pending | pending |
| 7 | asl_mri | done | 6 | 6 | 6 | NO | Simulated ASL perfusion maps | pending | pending | pending |
| 8 | atom_probe | done | 6 | 6 | 6 | NO | Simulated atom probe tomography | pending | pending | pending |
| 9 | bioluminescence_tomo | done | 6 | 6 | 6 | NO | Simulated BLT on mouse model | pending | pending | pending |
| 10 | brachytherapy_img | done | 6 | 6 | 6 | NO | Simulated brachytherapy dose maps | pending | pending | pending |
| 11 | brillouin | done | 6 | 6 | 6 | NO | Simulated Brillouin spectra | pending | pending | pending |
| 12 | cacti | done | 26 | 26 | 26 | YES | User-provided CACTI data | done | pending | pending |
| 13 | cars | done | 6 | 6 | 6 | NO | Simulated CARS spectral images | pending | pending | pending |
| 14 | cathodoluminescence | done | 6 | 6 | 6 | NO | Simulated CL maps | pending | pending | pending |
| 15 | cbct | done | 22 | 40 | 40 | YES | AAPM Low-Dose CT Grand Challenge + Shepp-Logan | done | pending | pending |
| 16 | cest_mri | done | 6 | 6 | 6 | NO | Simulated CEST Z-spectra | pending | pending | pending |
| 17 | ceus | done | 6 | 6 | 6 | NO | Simulated contrast-enhanced US | pending | pending | pending |
| 18 | clem | done | 6 | 6 | 6 | NO | Simulated correlative LM-EM | pending | pending | pending |
| 19 | coded_exposure | done | 6 | 6 | 6 | NO | Simulated coded exposure | pending | pending | pending |
| 20 | confocal_3d | done | 24 | 40 | 40 | YES | BioSR / FluoCells | done | pending | pending |
| 21 | confocal_endomicroscopy | done | 6 | 6 | 6 | NO | Simulated endomicroscopy | pending | pending | pending |
| 22 | confocal_livecell | done | 6 | 6 | 6 | NO | Simulated live cell confocal | pending | pending | pending |
| 23 | coronagraphy | done | 6 | 6 | 6 | NO | Simulated stellar coronagraphy | pending | pending | pending |
| 24 | cryo_em | done | 24 | 40 | 40 | YES | EMPIAR single-particle cryo-EM | done | pending | pending |
| 25 | cryo_et | done | 6 | 6 | 6 | NO | Simulated cryo-ET tiltseries | pending | pending | pending |
| 26 | ct | done | 33 | 60 | 60 | YES | LoDoPaB-CT (LIDC/IDRI) | done | pending | pending |
| 27 | ct_fluorescence | done | 6 | 6 | 6 | NO | Simulated XRF-CT phantoms | pending | pending | pending |
| 28 | cup | done | 6 | 6 | 6 | NO | Simulated CUP temporal imaging | pending | pending | pending |
| 29 | dark_field | done | 6 | 6 | 6 | NO | Simulated dark-field microscopy | pending | pending | pending |
| 30 | desi | done | 6 | 6 | 6 | NO | Simulated DESI mass spectrometry | pending | pending | pending |
| 31 | dexa | done | 6 | 6 | 6 | NO | Simulated DEXA bone density | pending | pending | pending |
| 32 | dic | done | 6 | 6 | 6 | NO | Simulated DIC shear interferometry | pending | pending | pending |
| 33 | diffusion_mri | done | 12 | 20 | 20 | YES | HARDI synthetic phantoms | done | pending | pending |
| 34 | digital_breast_tomo | done | 6 | 6 | 6 | NO | Simulated breast phantoms | pending | pending | pending |
| 35 | dna_paint | done | 6 | 6 | 6 | NO | Simulated DNA-PAINT | pending | pending | pending |
| 36 | doppler_ultrasound | done | 6 | 6 | 6 | NO | Simulated Doppler flow phantoms | pending | pending | pending |
| 37 | dot | done | 6 | 6 | 6 | NO | Simulated diffuse optical tomography | pending | pending | pending |
| 38 | ebsd | done | 6 | 6 | 6 | NO | Simulated EBSD patterns | pending | pending | pending |
| 39 | eddy_current | done | 6 | 6 | 6 | NO | Simulated eddy current NDT | pending | pending | pending |
| 40 | edx_mapping | done | 6 | 6 | 6 | NO | Simulated EDX elemental maps | pending | pending | pending |
| 41 | eels | done | 6 | 6 | 6 | NO | Simulated EELS spectra | pending | pending | pending |
| 42 | eht_imaging | done | 6 | 6 | 6 | NO | EHT M87* simulation | pending | pending | pending |
| 43 | elastography | done | 6 | 6 | 6 | NO | Simulated shear wave phantoms | pending | pending | pending |
| 44 | electron_diffraction | done | 6 | 6 | 6 | NO | Simulated electron diffraction patterns | pending | pending | pending |
| 45 | electron_holography | done | 6 | 6 | 6 | NO | Simulated off-axis holography | pending | pending | pending |
| 46 | electron_tomography | done | 6 | 6 | 6 | NO | Simulated ET tiltseries | pending | pending | pending |
| 47 | endoscopy | done | 12 | 20 | 20 | YES | CholecSeg8k / EndoSLAM | done | pending | pending |
| 48 | entangled_photon | done | 6 | 6 | 6 | NO | Simulated entangled photon imaging | pending | pending | pending |
| 49 | event_camera | done | 6 | 6 | 6 | NO | Simulated DVS events | pending | pending | pending |
| 50 | expansion | done | 6 | 6 | 6 | NO | Simulated expansion microscopy | pending | pending | pending |
| 51 | fib_sem | done | 6 | 6 | 6 | NO | Simulated FIB-SEM serial sections | pending | pending | pending |
| 52 | flash_lidar | done | 6 | 6 | 6 | NO | Simulated flash LiDAR depth maps | pending | pending | pending |
| 53 | flim | done | 6 | 6 | 6 | NO | Simulated FLIM lifetime maps | pending | pending | pending |
| 54 | fluoroscopy | done | 6 | 6 | 6 | NO | Simulated C-arm fluoroscopy | pending | pending | pending |
| 55 | fmri | done | 12 | 20 | 20 | YES | Synthetic BOLD fMRI phantoms | done | pending | pending |
| 56 | fpm | done | 12 | 20 | 20 | YES | Simulated Fourier ptychography | done | pending | pending |
| 57 | ftir_imaging | done | 24 | 40 | 22 | YES | Simulated FTIR spectral images | done | pending | pending |
| 58 | fundus | done | 12 | 20 | 20 | YES | DRIVE retinal vessel dataset | done | pending | pending |
| 59 | fwi | done | 6 | 6 | 6 | NO | Simulated full waveform inversion | pending | pending | pending |
| 60 | gaussian_splatting | done | 5 | 5 | 5 | NO | Synthetic 3DGS scenes | pending | pending | pending |
| 61 | ghost_imaging | done | 3 | 3 | 3 | NO | Simulated computational ghost imaging | pending | pending | pending |
| 62 | gpr | done | 6 | 6 | 6 | NO | Simulated ground-penetrating radar | pending | pending | pending |
| 63 | gravitational_wave | done | 6 | 6 | 6 | NO | LIGO simulated strain | pending | pending | pending |
| 64 | hdr_imaging | done | 6 | 6 | 6 | NO | Simulated HDR from LDR | pending | pending | pending |
| 65 | holography | done | 5 | 5 | 5 | NO | Simulated digital holography | pending | pending | pending |
| 66 | hyperspectral_remote | done | 36 | 60 | 60 | YES | Indian Pines / Pavia University | done | pending | pending |
| 67 | impedance_tomo | done | 6 | 6 | 6 | NO | Simulated electrical impedance | pending | pending | pending |
| 68 | industrial_ct | done | 24 | 40 | 40 | YES | Simulated industrial phantoms + ASTRA toolbox | done | pending | pending |
| 69 | insar | done | 12 | 20 | 20 | YES | Simulated InSAR interferograms | done | pending | pending |
| 70 | integral | done | 6 | 6 | 6 | NO | Simulated integral imaging | pending | pending | pending |
| 71 | ism | done | 6 | 6 | 6 | NO | Simulated image scanning microscopy | pending | pending | pending |
| 72 | ivus | done | 6 | 6 | 6 | NO | Simulated IVUS cross-sections | pending | pending | pending |
| 73 | lattice_lightsheet | done | 6 | 6 | 6 | NO | Simulated lattice lightsheet | pending | pending | pending |
| 74 | lensless | done | 12 | 20 | 20 | YES | Simulated lensless imaging | done | pending | pending |
| 75 | libs | done | 6 | 6 | 6 | NO | Simulated LIBS spectra | pending | pending | pending |
| 76 | lidar | done | 24 | 40 | 40 | YES | KITTI / ModelNet point clouds | done | pending | pending |
| 77 | light_field | done | 6 | 6 | 6 | NO | Simulated light field | pending | pending | pending |
| 78 | lightsheet | done | 24 | 40 | 40 | YES | OpenSPIM / lattice lightsheet sims | done | pending | pending |
| 79 | lucky_imaging | done | 6 | 6 | 6 | NO | Simulated lucky imaging | pending | pending | pending |
| 80 | machine_vision | done | 6 | 6 | 6 | NO | Standard test images (Cameraman, etc.) | pending | pending | pending |
| 81 | magnetic_particle | done | 6 | 6 | 6 | NO | Simulated MPI signal | pending | pending | pending |
| 82 | maldi_msi | done | 6 | 6 | 6 | NO | Simulated MALDI imaging | pending | pending | pending |
| 83 | mammography | done | 24 | 40 | 40 | YES | DDSM / VinDr-Mammo phantom | done | pending | pending |
| 84 | mfm | done | 6 | 6 | 6 | NO | Simulated MFM magnetic domains | pending | pending | pending |
| 85 | minflux | done | 6 | 6 | 6 | NO | Simulated MINFLUX localizations | pending | pending | pending |
| 86 | mr_elastography | done | 6 | 6 | 6 | NO | Simulated MRE wave fields | pending | pending | pending |
| 87 | mr_fingerprinting | done | 6 | 6 | 6 | NO | Simulated MRF dictionaries | pending | pending | pending |
| 88 | mra | done | 6 | 6 | 6 | NO | Simulated MR angiography | pending | pending | pending |
| 89 | mri | done | 24 | 40 | 40 | YES | M4Raw / fastMRI | done | pending | pending |
| 90 | mrs | done | 6 | 6 | 6 | NO | Simulated MR spectra | pending | pending | pending |
| 91 | multispectral_sat | done | 6 | 6 | 6 | NO | Simulated multispectral satellite | pending | pending | pending |
| 92 | muon_tomo | done | 6 | 6 | 6 | NO | Simulated muon scattering tomography | pending | pending | pending |
| 93 | nerf | done | 6 | 6 | 6 | NO | Synthetic NeRF blender scenes | pending | pending | pending |
| 94 | neutron_diffraction | done | 6 | 6 | 6 | NO | Simulated neutron diffraction | pending | pending | pending |
| 95 | neutron_tomo | done | 6 | 6 | 6 | NO | Simulated neutron tomography | pending | pending | pending |
| 96 | nirs_brain | done | 6 | 6 | 6 | NO | Simulated fNIRS brain maps | pending | pending | pending |
| 97 | nsom | done | 6 | 6 | 6 | NO | Simulated near-field optical maps | pending | pending | pending |
| 98 | ocean_acoustic_tomo | done | 6 | 6 | 6 | NO | Simulated ocean acoustic tomography | pending | pending | pending |
| 99 | ocean_color | done | 6 | 6 | 6 | NO | Simulated ocean color radiometry | pending | pending | pending |
| 100 | oct | done | 24 | 40 | 40 | YES | Duke Retinal OCT + BM3D benchmark | done | pending | pending |
| 101 | octa | done | 6 | 6 | 6 | NO | Simulated OCTA vasculature | pending | pending | pending |
| 102 | odt | done | 12 | 5 | 5 | NO | Simulated optical diffraction tomo | pending | pending | pending |
| 103 | palm_storm | done | 24 | 40 | 40 | YES | Simulated PALM/STORM localizations | done | pending | pending |
| 104 | panorama | done | 6 | 6 | 6 | NO | Simulated panoramic stitching | pending | pending | pending |
| 105 | particle_calorimetry | done | 6 | 6 | 6 | NO | Simulated calorimeter shower images | pending | pending | pending |
| 106 | passive_microwave | done | 6 | 6 | 6 | NO | Simulated microwave brightness temperature | pending | pending | pending |
| 107 | pet | done | 12 | 20 | 20 | YES | Zubal + XCAT phantoms | done | pending | pending |
| 108 | pet_ct | done | 3 | 3 | 3 | NO | Simulated PET/CT fusion | pending | pending | pending |
| 109 | pet_mr | done | 24 | 40 | 40 | YES | Simulated PET/MR fusion | done | pending | pending |
| 110 | phase_contrast | done | 6 | 6 | 6 | NO | Simulated Zernike phase contrast | pending | pending | pending |
| 111 | phase_retrieval | done | 12 | 20 | 20 | YES | Simulated coherent diffraction | done | pending | pending |
| 112 | photoacoustic | done | 24 | 40 | 40 | YES | Simulated PAT with k-Wave | done | pending | pending |
| 113 | photometric_stereo | done | 6 | 6 | 6 | NO | Simulated photometric stereo | pending | pending | pending |
| 114 | polarization | done | 6 | 6 | 6 | NO | Simulated polarimetric imaging | pending | pending | pending |
| 115 | polsar | done | 6 | 6 | 6 | NO | AIRSAR San Francisco PolSAR | pending | pending | pending |
| 116 | portal_imaging | done | 6 | 6 | 6 | NO | Simulated EPID portal images | pending | pending | pending |
| 117 | proton_radiography | done | 6 | 6 | 6 | NO | Simulated proton radiography | pending | pending | pending |
| 118 | proton_therapy_img | done | 6 | 6 | 6 | NO | Simulated proton therapy imaging | pending | pending | pending |
| 119 | ptychography | done | 24 | 40 | 18 | NO | Simulated ptychographic scans | pending | pending | pending |
| 120 | pump_probe | done | 6 | 6 | 6 | NO | Simulated pump-probe ultrafast images | pending | pending | pending |
| 121 | quantum_illumination | done | 6 | 6 | 6 | NO | Simulated quantum illumination | pending | pending | pending |
| 122 | radio_astronomy | done | 6 | 6 | 6 | NO | VLA FIRST survey simulation | pending | pending | pending |
| 123 | radio_interferometry | done | 6 | 6 | 6 | NO | Simulated radio interferometry | pending | pending | pending |
| 124 | raman_imaging | done | 24 | 40 | 40 | YES | Simulated Raman hyperspectral | done | pending | pending |
| 125 | sar | done | 24 | 40 | 40 | YES | Sentinel-1 / MSTAR benchmark | done | pending | pending |
| 126 | saxs | done | 6 | 6 | 6 | NO | Simulated SAXS profiles | pending | pending | pending |
| 127 | sd_cassi | done | 10 | 20 | 20 | YES | KAIST TSA / ICVL spectral | done | pending | pending |
| 128 | seismic_tomo | done | 6 | 6 | 6 | NO | Simulated seismic tomography | pending | pending | pending |
| 129 | sem | done | 24 | 40 | 40 | YES | Simulated SEM secondary electron images | done | pending | pending |
| 130 | shearography | done | 6 | 6 | 6 | NO | Simulated shearography displacement | pending | pending | pending |
| 131 | shg | done | 6 | 6 | 6 | NO | Simulated SHG images | pending | pending | pending |
| 132 | sim | done | 24 | 40 | 40 | YES | BioSR SIM benchmark | done | pending | pending |
| 133 | sims | done | 6 | 6 | 6 | NO | Simulated SIMS depth profiles | pending | pending | pending |
| 134 | solar_imaging | done | 6 | 6 | 6 | NO | SDO/AIA solar images | pending | pending | pending |
| 135 | sonar | done | 6 | 6 | 6 | NO | Simulated sonar imaging | pending | pending | pending |
| 136 | spc_block | done | 11 | 11 | 11 | NO | Simulated single-pixel camera | pending | pending | pending |
| 137 | spc_kronecker | done | 31 | 60 | 60 | YES | Simulated SPC Kronecker | done | pending | pending |
| 138 | spect | done | 12 | 20 | 20 | YES | SIMIND Monte Carlo + Zubal | done | pending | pending |
| 139 | spect_ct | done | 24 | 40 | 40 | YES | Simulated SPECT/CT fusion | done | pending | pending |
| 140 | spectral_ct | done | 24 | 40 | 40 | YES | Dual-energy CT simulation (FORBILD phantom) | done | pending | pending |
| 141 | spinning_disk | done | 6 | 6 | 6 | NO | Simulated spinning disk confocal | pending | pending | pending |
| 142 | srs | done | 6 | 6 | 6 | NO | Simulated SRS microscopy | pending | pending | pending |
| 143 | sted | done | 24 | 40 | 40 | YES | Simulated STED nanoscopy | done | pending | pending |
| 144 | stem | done | 6 | 6 | 6 | NO | Simulated STEM-HAADF | pending | pending | pending |
| 145 | stm | done | 6 | 6 | 6 | NO | Simulated STM surface images | pending | pending | pending |
| 146 | streak_camera | done | 6 | 6 | 6 | NO | Simulated streak camera temporal imaging | pending | pending | pending |
| 147 | structured_light | done | 6 | 6 | 6 | NO | Simulated structured light 3D | pending | pending | pending |
| 148 | swi | done | 6 | 6 | 6 | NO | Simulated SWI phase/magnitude | pending | pending | pending |
| 149 | talbot_lau | done | 6 | 6 | 6 | NO | Simulated grating interferometry | pending | pending | pending |
| 150 | tem | done | 24 | 40 | 40 | YES | EMPIAR public TEM datasets | done | pending | pending |
| 151 | terahertz | done | 6 | 6 | 6 | NO | Simulated / synthetic | pending | pending | pending |
| 152 | three_photon | done | 6 | 6 | 6 | NO | Simulated 3-photon deep tissue | pending | pending | pending |
| 153 | tirf | done | 6 | 6 | 6 | NO | Simulated TIRF microscopy | pending | pending | pending |
| 154 | tof_camera | done | 6 | 6 | 6 | NO | Simulated time-of-flight depth | pending | pending | pending |
| 155 | two_photon | done | 24 | 40 | 40 | YES | Allen Brain Observatory | done | pending | pending |
| 156 | ultrasonic_phased_array | done | 6 | 6 | 6 | NO | Simulated phased-array UT | pending | pending | pending |
| 157 | ultrasound | done | 12 | 20 | 20 | YES | PICMUS / IUS challenge | done | pending | pending |
| 158 | us_mri | done | 6 | 6 | 6 | NO | Simulated US-guided MRI | pending | pending | pending |
| 159 | waxs | done | 6 | 6 | 6 | NO | Simulated WAXS patterns | pending | pending | pending |
| 160 | weather_radar | done | 6 | 6 | 6 | NO | Simulated weather radar reflectivity | pending | pending | pending |
| 161 | widefield | done | 24 | 40 | 40 | YES | EPFL deconvolution benchmark | done | pending | pending |
| 162 | widefield_lowdose | done | 6 | 6 | 6 | NO | Low-photon widefield simulation | pending | pending | pending |
| 163 | xfel_sfx | done | 6 | 6 | 6 | NO | Simulated XFEL diffraction | pending | pending | pending |
| 164 | xray_crystallography | done | 6 | 6 | 6 | NO | Simulated X-ray crystallography | pending | pending | pending |
| 165 | xray_ndt | done | 6 | 6 | 6 | NO | Simulated NDT radiography | pending | pending | pending |
| 166 | xray_radiography | done | 6 | 6 | 6 | NO | Simulated projection radiography | pending | pending | pending |
| 167 | xrf_imaging | done | 6 | 6 | 6 | NO | Simulated XRF elemental maps | pending | pending | pending |
| 168 | xrf_tomo | done | 6 | 6 | 6 | NO | Simulated XRF tomography | pending | pending | pending |

**Summary:** 168/168 datasets present, 39/168 with correct sample counts, 39/168 sources verified
