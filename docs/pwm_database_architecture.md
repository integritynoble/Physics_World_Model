# PWM Database Architecture — Global Primitive Basis + Operator Families

> **Design principle**: One global primitive basis (11 canonical types). Modalities differ by **template DAG + operator families + parameters + mismatch models**, not by introducing new primitive types.
>
> **Theoretical backing**: The Finite Primitive Basis (FPB) Theorem guarantees ε-approximate representation for all modalities in the defined operator class C_img. The 11-primitive basis is **universal under a falsifiable extension protocol**, not "absolutely final forever."

---

## Strict Vocabulary

| Term | Definition | Example |
|------|-----------|---------|
| **Primitive** | Canonical operator type (11 only, global, stable) | `C` (Convolve) |
| **Operator family** | Physics-specific version under a primitive | `Gaussian_PSF`, `Airy_PSF`, `CTF` |
| **Operator instance** | Family + concrete parameters | `Gaussian_PSF(σ=2.0, mode=reflect)` |
| **Modality template** | DAG pattern + default operator choices + priors | `widefield: C → D` |
| **Design** | Instantiated DAG for a user prompt | `C(Gaussian_PSF, σ=1.8) → D(sCMOS, η=5.0)` |

These five levels must remain distinct in code, documentation, and product messaging.

---

## 1. Primitive Registry (global, small, stable)

One row per canonical primitive type. This is the **typed primitive library** from the FPB Theorem.

### Schema: `primitives`

| Field | Type | Description |
|-------|------|-------------|
| `primitive_id` | `enum` | One of: `P`, `M`, `Π`, `F`, `C`, `Σ`, `D`, `S`, `W`, `R`, `Λ` |
| `name` | `str` | Human-readable name |
| `physics_stage` | `enum` | `propagation`, `interaction`, `encoding_projection`, `detection_readout` |
| `type_signature` | `dict` | Input/output tensor types, dimensions, units |
| `is_linear` | `bool` | Whether the primitive is linear (enables adjoint) |
| `forward_contract` | `str` | Mathematical specification of `forward(x, θ)` |
| `adjoint_contract` | `str` | Mathematical specification of `adjoint(y, θ)` |
| `constraints` | `list` | Compositional constraints (e.g., D must be terminal) |
| `validation_tests` | `list` | Required validation tests (e.g., dot-product adjoint test) |

### The 11 Primitives

| ID | Name | Stage | Forward Contract | Adjoint |
|----|------|-------|-----------------|---------|
| `P` | Propagate | propagation | Free-space wave propagation: `y = H_prop(θ) · x` | `H_prop†(θ) · y` |
| `M` | Modulate | interaction | Element-wise multiplication: `y = diag(m(θ)) · x` | `diag(m(θ))† · y` |
| `Π` | Project | encoding_projection | Line-integral projection: `y = R(θ) · x` | `R†(θ) · y` (backprojection) |
| `F` | Encode | encoding_projection | Fourier/frequency encoding: `y = F(θ) · x` | `F†(θ) · y` |
| `C` | Convolve | propagation | Spatial convolution: `y = h(θ) * x` | `h(-θ) * y` |
| `Σ` | Accumulate | detection_readout | Summation over axis: `y = Σ_k x_k` | Broadcast/expand |
| `D` | Detect | detection_readout | Detector response: `y = η(g · x) + noise` | — (terminal, non-invertible) |
| `S` | Sample | detection_readout | Sub-sampling: `y = P_Ω · x` | Zero-fill: `P_Ω† · y` |
| `W` | Disperse | detection_readout | Wavelength-dependent shift: `y_λ = T_λ(θ) · x` | `T_λ†(θ) · y` |
| `R` | Scatter | interaction | Scattering interaction: `y = S(θ) · x` | `S†(θ) · y` |
| `Λ` | Filter | detection_readout | Energy/wavelength selection: `y = Λ(E, θ) · x` | `Λ†(E, θ) · y` |

### 4 Physics-Stage Families

| Family | Primitives | Description |
|--------|-----------|-------------|
| `propagation` | P, C | Wave/field transport through free space or media |
| `interaction` | M, R | Scene interaction — modulation and scattering |
| `encoding_projection` | Π, F | Geometric/frequency-domain encoding |
| `detection_readout` | Σ, S, W, Λ, D | Signal integration, sampling, dispersion, detection |

### 5 Detect Response Families

| Family | Mathematical Form | Example Modalities |
|--------|------------------|-------------------|
| `linear_intensity` | `η(x) = g·|x|²` | Most photon/X-ray detectors |
| `logarithmic` | `η(x) = g·log(1 + |x|²/x₀)` | Ultrasound, OCT |
| `sigmoid` | `η(x) = g·σ(|x|² - x₀)` | Saturating sensors |
| `poisson_rate` | `η(x) = g·|x|²` (Poisson draw) | Photon-counting, PET, SPECT |
| `coherent_field` | `η(x) = g·Re[x·e^(iφ)]` | Holography, interferometry |

### Extension Protocol

When the 11-primitive basis fails to represent a new modality within ε < 0.01:

1. **Forward/adjoint validation** — new primitive passes dot-product adjoint test
2. **Representation gap** — prove `min_G ε_tier2 > ε` without the new primitive
3. **Error reduction** — show `ε_tier2 < ε` with the new primitive
4. **Multi-modality need** — ≥2 modalities require it
5. **Closure test** — all existing decompositions remain valid

History: `R` (Scatter) was added via this protocol for Compton-like scattering. `Λ` (Filter) was added for energy-selective imaging (DEXA, EELS).

---

## 2. Operator Family Registry (large, grows over time)

Each primitive can have many operator families and implementations. This is where modality-specific physics lives.

### Schema: `operator_families`

| Field | Type | Description |
|-------|------|-------------|
| `operator_family_id` | `str` | Unique ID (e.g., `c_gaussian_psf_v1`) |
| `primitive_id` | `enum` | Parent primitive (e.g., `C`) |
| `physics_family` | `str` | Physics category (e.g., `optical_psf`) |
| `display_name` | `str` | Human-readable label (e.g., `Gaussian PSF`) |
| `analytic_form` | `str` | Mathematical expression or `"learned_surrogate"` or `"lookup_table"` |
| `parameter_schema` | `dict` | Parameter names, types, units, valid ranges |
| `default_params` | `dict` | Default parameter values |
| `is_linear` | `bool` | Inherited from primitive, may be overridden |
| `adjoint_available` | `bool` | Whether adjoint is implemented |
| `backend_implementations` | `list` | Available backends: `pytorch`, `jax`, `cuda`, `cpu` |
| `adjoint_validation_status` | `enum` | `validated`, `pending`, `not_applicable` |
| `compatible_carriers` | `list` | Carrier types this family applies to |
| `compatible_primitives_before` | `list` | Primitives that can precede this in a DAG |
| `compatible_primitives_after` | `list` | Primitives that can follow this in a DAG |
| `reference` | `str` | Citation or source |

### Operator Families per Primitive

#### C — Convolve (30+ families)

| Family ID | Display Name | Analytic Form | Parameters | Modalities |
|-----------|-------------|---------------|------------|------------|
| `c_gaussian_psf` | Gaussian PSF | `exp(-r²/2σ²)` | `σ` (px) | widefield, widefield_lowdose |
| `c_airy_psf` | Airy Disk PSF | `[2J₁(x)/x]²` | `NA`, `λ` | widefield (theoretical) |
| `c_confocal_psf` | Confocal PSF | `PSF_ill × PSF_det` | `σ`, `pinhole` | confocal_livecell |
| `c_3d_confocal_psf` | 3D Confocal PSF | `PSF_xy × PSF_z` | `σ_xy`, `σ_z`, `RI` | confocal_3d |
| `c_lightsheet_psf` | Light-Sheet PSF | Sheet × Detection PSF | `thickness`, `tilt` | lightsheet |
| `c_two_photon_psf` | Two-Photon PSF | `|PSF_exc|⁴` | `σ`, `scattering` | two_photon |
| `c_three_photon_psf` | Three-Photon PSF | `|PSF_exc|⁶` | `σ`, `scattering` | three_photon |
| `c_sted_psf` | STED Effective PSF | `PSF_exc × (1-PSF_dep)` | `depletion`, `saturation` | sted |
| `c_tirf_psf` | Evanescent PSF | `exp(-z/d) × PSF_xy` | `angle`, `depth` | tirf |
| `c_optic_psf` | Ophthalmic PSF | Zernike-based | `aberrations`, `pupil` | fundus |
| `c_fiber_psf` | Fiber Bundle PSF | Fiber coupling model | `pitch`, `crosstalk` | endoscopy, confocal_endomicroscopy |
| `c_depth_psf` | Depth-Varying PSF | `PSF(z)` | `focus_stack` | panorama |
| `c_expansion_psf` | Expansion-Scaled PSF | `PSF / expansion_factor` | `factor`, `distortion` | expansion |
| `c_minflux_psf` | MINFLUX Donut PSF | `1 - exp(-r²/2σ²)` | `beam_center` | minflux |
| `c_ism_psf` | ISM Detector-Array PSF | Multi-element PSF | `offset`, `mag` | ism |
| `c_spinning_disk_psf` | Spinning Disk PSF | Pinhole array × PSF | `crosstalk`, `wobble` | spinning_disk |
| `c_lattice_psf` | Lattice Light-Sheet PSF | Bessel lattice | `period`, `NA` | lattice_lightsheet |
| `c_phase_contrast_psf` | Phase Contrast PSF | `CTF_phase_ring` | `ring_absorption` | phase_contrast |
| `c_dark_field_psf` | Dark-Field PSF | High-NA condenser | `NA_ratio`, `stray` | dark_field |
| `c_ctf` | Contrast Transfer Function | `CTF(f, Δf, Cs)` | `defocus`, `Cs`, `astigmatism` | tem, cryo_em |
| `c_electron_probe` | Electron Probe | Probe formation | `aberrations`, `convergence` | sem, stem |
| `c_ao_psf` | AO-Corrected PSF | `Residual_WF × PSF` | `r0`, `residual` | adaptive_optics |
| `c_lucky_psf` | Lucky Imaging PSF | Best-seeing selection | `r0`, `threshold` | lucky_imaging |
| `c_machine_vision_psf` | Machine Vision Optics | Industrial lens PSF | `MTF`, `distortion` | machine_vision |
| `c_nsom_psf` | Near-Field PSF | Aperture near-field | `tip_distance`, `aperture` | nsom |
| `c_motion_blur_psf` | Motion Blur PSF | Linear motion kernel | `velocity`, `exposure` | coded_exposure |
| `c_structured_light_psf` | Structured Light PSF | Projector × camera | `gamma`, `defocus` | structured_light, photometric_stereo |
| `c_dic_psf` | DIC Shear PSF | Gradient × PSF | `shear`, `bias` | dic |
| `c_microlens_psf` | Microlens Array PSF | Micro-optic PSF | `pitch`, `f-number` | light_field, integral |
| `c_fib_sem_psf` | FIB-SEM Imaging PSF | Electron + ion PSF | `curtaining`, `charging` | fib_sem |
| `c_acoustic_lens_psf` | Acoustic Lens PSF | Focused acoustic | `speed`, `focus` | acoustic_microscopy |

#### P — Propagate (22+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `p_fresnel` | Fresnel Propagation | `λ`, `z`, `pixel_size` | holography |
| `p_angular_spectrum` | Angular Spectrum | `λ`, `z`, `pixel_size` | phase_retrieval |
| `p_acoustic` | Acoustic Propagation | `c₀`, `attenuation`, `freq` | ultrasound, doppler_ultrasound, sonar, ivus, ceus, elastography, acoustic_microscopy |
| `p_electron` | Electron Wave | `voltage`, `Cs`, `defocus` | tem, sem, stem, electron_holography, electron_diffraction |
| `p_diffuse` | Diffuse Photon Transport | `μ_a`, `μ_s'` | dot, nirs_brain, bioluminescence_tomo |
| `p_low_coherence` | Low-Coherence Interferometric | `bandwidth`, `center_λ` | oct, octa |
| `p_modulated` | Modulated CW | `mod_freq`, `phase` | tof_camera |
| `p_pulsed` | Pulsed Laser | `pulse_width`, `rep_rate` | lidar, flash_lidar |
| `p_probe` | Ptychographic Probe | `probe_size`, `overlap` | ptychography, fpm |
| `p_diffuser` | Diffuser Propagation | `PSF_diffuser` | lensless |
| `p_rf` | RF/EM Propagation | `freq`, `permittivity` | gpr, weather_radar |
| `p_thz` | Terahertz Propagation | `freq`, `water_vapor` | terahertz |
| `p_thermal` | Thermal Diffusion | `diffusivity`, `emissivity` | active_thermography |
| `p_seismic` | Seismic Wave | `velocity_model`, `Q` | seismic_tomo, fwi, ocean_acoustic_tomo |
| `p_gravitational` | Gravitational Wave | `strain`, `freq` | gravitational_wave |
| `p_coherent_scatter` | Coherent Scattering | `grating_period`, `distance` | odt, talbot_lau |
| `p_shear_wave` | Shear Wave | `shear_speed`, `attenuation` | elastography |
| `p_coronagraph` | Coronagraphic | `IWA`, `WFE` | coronagraphy |
| `p_solar` | Solar Photon | `wavelength`, `stray_light` | solar_imaging |
| `p_shearography` | Shearographic Interferometric | `shear_amount` | shearography |
| `p_nerf_ray` | Neural Radiance Ray | `pose`, `focal` | nerf |
| `p_splat_ray` | Gaussian Splatting Ray | `pose`, `focal` | gaussian_splatting |

#### M — Modulate (34+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `m_coded_aperture` | Binary Coded Aperture | `mask`, `shift` | cassi, sd_cassi |
| `m_sensing_matrix` | Sensing Matrix | `Φ`, `sampling_rate` | spc, spc_block, matrix |
| `m_kronecker` | Kronecker Sensing | `H`, `W` | spc_kronecker |
| `m_temporal_mask` | Temporal Mask | `masks_per_frame` | cacti |
| `m_polarizer` | Polarizer/Analyzer | `angle`, `extinction` | polarization |
| `m_sim_grating` | SIM Grating Pattern | `freq`, `angle`, `phase` | sim |
| `m_led_array` | LED Array Illumination | `positions`, `intensities` | fpm |
| `m_dmd` | DMD Pattern | `pattern`, `refresh_rate` | cup, ghost_imaging |
| `m_stochastic` | Stochastic Activation | `density`, `brightness` | palm_storm, dna_paint |
| `m_pump_beam` | Pump/Excitation Beam | `power`, `wavelength`, `duration` | flim, shg, pump_probe, cars, srs, libs, brillouin, xfel_sfx |
| `m_entangled_source` | Entangled Photon Source | `pair_rate`, `concurrence` | quantum_illumination, entangled_photon |
| `m_exposure_code` | Temporal Exposure Code | `code_sequence` | coded_exposure |
| `m_event_threshold` | Event Threshold | `contrast_threshold`, `refractory` | event_camera |
| `m_exposure_bracket` | Multi-Exposure Bracket | `exposure_ratios` | hdr_imaging |
| `m_coronagraph_mask` | Coronagraph Stop | `IWA`, `throughput` | coronagraphy |
| `m_frame_selection` | Lucky Frame Selection | `threshold`, `r0` | lucky_imaging |
| `m_rf_pulse` | RF Pulse Sequence | `flip_angle`, `TR`, `TE` | mri, fmri, mrs, diffusion_mri, mr_elastography, cest_mri, asl_mri, mra, swi, mr_fingerprinting |
| `m_fiber_coupling` | Fiber-Optic Coupling | `NA_fiber`, `transmission` | endoscopy, confocal_endomicroscopy |
| `m_deformable_mirror` | Wavefront Modulator | `actuator_count`, `stroke` | adaptive_optics |
| `m_electrode_pattern` | Electrode Drive | `impedance`, `pattern` | impedance_tomo |
| `m_xrf_excitation` | X-ray/Electron Excitation | `energy`, `current` | xrf_imaging, edx_mapping, cathodoluminescence |
| `m_ffp` | Focus Field Pattern | `field_gradient`, `drive_freq` | magnetic_particle |
| `m_spectral_filter` | Spectral Filter | `bands`, `bandwidth` | multispectral_sat, ocean_color, ftir_imaging, streak_camera |
| `m_solar_filter` | Solar Wavelength Filter | `wavelength`, `bandwidth` | solar_imaging |
| `m_projected_pattern` | Projected Pattern | `type`, `period` | structured_light |
| `m_photometric` | Photometric Light Source | `direction`, `intensity` | photometric_stereo |
| `m_nsom_probe` | Near-Field Probe | `aperture`, `distance` | nsom |
| `m_shear_optic` | Shearing Optic | `shear_amount`, `direction` | shearography |
| `m_polsar` | Polarimetric SAR | `HH`, `HV`, `VH`, `VV` | polsar |
| `m_aperture` | Aperture Function | `shape`, `size` | nerf, gaussian_splatting |
| `m_scan_electron` | Electron Scan Pattern | `raster`, `step_size` | electron_diffraction |

#### Π — Project (16+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `pi_fan_beam` | Fan-Beam Projection | `n_angles`, `det_count` | ct, industrial_ct |
| `pi_cone_beam` | Cone-Beam Projection | `source_distance`, `det_distance` | cbct |
| `pi_parallel` | Parallel-Hole Collimator | `hole_diameter`, `septa` | spect |
| `pi_lor` | Line-of-Response | `ring_diameter`, `TOF_resolution` | pet |
| `pi_xray_proj` | Generic X-ray Projection | `SDD`, `kVp` | xray_radiography, angiography, fluoroscopy, dexa, xray_ndt, portal_imaging |
| `pi_contact` | Contact/Compression | `compression_force`, `thickness` | mammography, digital_breast_tomo |
| `pi_neutron` | Neutron Attenuation | `spectrum`, `scattering_fraction` | neutron_tomo |
| `pi_proton` | Proton Transmission | `energy`, `MCS_model` | proton_radiography, proton_therapy_img |
| `pi_muon` | Muon Scattering | `angular_resolution`, `track_model` | muon_tomo |
| `pi_electron_tilt` | Electron Tilt-Series | `tilt_range`, `tilt_step`, `missing_wedge` | electron_tomography, cryo_et |
| `pi_brachytherapy` | Brachytherapy Dose | `source_model`, `geometry` | brachytherapy_img |
| `pi_spectral_ct` | Energy-Resolved Projection | `energy_bins`, `thresholds` | spectral_ct |
| `pi_xrf_sinogram` | XRF Sinogram | `fluorescence_yield`, `self_absorption` | xrf_tomo |
| `pi_ray_cast` | Volume Ray Casting | `step_size`, `density_field` | nerf |
| `pi_gaussian_splat` | Gaussian Splatting | `splat_params`, `opacity` | gaussian_splatting |
| `pi_microlens` | Microlens Array | `pitch`, `f_number` | light_field, integral |

#### F — Encode (18+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `f_cartesian_kspace` | Cartesian k-Space | `accel`, `calib_size` | mri |
| `f_epi` | Echo-Planar Imaging | `echo_spacing`, `bandwidth` | fmri, diffusion_mri |
| `f_fid` | Free Induction Decay | `dwell_time`, `spectral_width` | mrs |
| `f_mre_meg` | Motion-Encoding Gradient | `freq`, `amplitude` | mr_elastography |
| `f_cest_sat` | CEST Saturation | `offset`, `power`, `duration` | cest_mri |
| `f_asl_label` | Arterial Spin Labeling | `label_duration`, `PLD` | asl_mri |
| `f_mra_encode` | Angiographic Encoding | `VENC`, `contrast_timing` | mra |
| `f_swi_phase` | SWI Phase | `TE`, `filter_size` | swi |
| `f_mrf_schedule` | MRF Schedule | `flip_angles`, `TR_pattern` | mr_fingerprinting |
| `f_range_doppler` | Range-Doppler | `PRF`, `bandwidth`, `velocity` | sar |
| `f_insar_phase` | InSAR Phase | `baseline`, `wavelength` | insar |
| `f_polsar_encode` | PolSAR Encoding | `polarization_basis` | polsar |
| `f_diffraction` | Electron Diffraction | `camera_length`, `voltage` | electron_diffraction |
| `f_eddy_impedance` | Eddy Current | `frequency`, `lift_off` | eddy_current |
| `f_mpi_ffp` | MPI Frequency Encoding | `drive_freq`, `gradient` | magnetic_particle |
| `f_visibility` | Radio Visibility | `baseline`, `freq` | radio_interferometry, radio_astronomy, eht_imaging |
| `f_bragg` | Bragg Diffraction | `wavelength`, `crystal_params` | xray_crystallography |
| `f_neutron_tof` | Neutron TOF | `wavelength`, `chopper` | neutron_diffraction |

#### R — Scatter (21+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `r_raman` | Spontaneous Raman | `shift`, `cross_section` | raman_imaging |
| `r_cars` | Coherent Anti-Stokes Raman | `pump_stokes_offset`, `non_resonant` | cars |
| `r_srs` | Stimulated Raman | `lock_in_phase`, `XPM` | srs |
| `r_brillouin` | Brillouin Scattering | `shift`, `linewidth` | brillouin |
| `r_shg` | Second Harmonic Generation | `phase_matching`, `chi2` | shg |
| `r_libs` | Laser-Induced Breakdown | `energy`, `matrix_effect` | libs |
| `r_xrf` | X-ray Fluorescence | `yield`, `self_absorption` | xrf_imaging, xrf_tomo |
| `r_cathodoluminescence` | Cathodoluminescence | `beam_current`, `collection` | cathodoluminescence |
| `r_edx` | Energy-Dispersive X-ray | `solid_angle`, `peak_overlap` | edx_mapping |
| `r_fluorescence_lifetime` | Fluorescence + Lifetime | `IRF`, `lifetime` | flim |
| `r_pump_probe` | Pump-Probe Transient | `time_zero`, `chirp` | pump_probe |
| `r_xfel` | XFEL Diffraction | `hit_rate`, `partiality` | xfel_sfx |
| `r_quantum_correlation` | Quantum Correlation | `concurrence`, `dark_count` | quantum_illumination, entangled_photon |
| `r_kikuchi` | Kikuchi/EBSD Pattern | `pattern_center`, `tilt` | ebsd |
| `r_saxs` | Small-Angle X-ray | `divergence`, `parasitic` | saxs |
| `r_waxs` | Wide-Angle X-ray | `distance`, `polarization` | waxs |
| `r_calorimetry` | Particle Shower | `intercalibration`, `nonlinearity` | particle_calorimetry |
| `r_bubble_harmonic` | Microbubble Nonlinear | `concentration`, `harmonic` | ceus |
| `r_weather` | Hydrometeor Scattering | `reflectivity`, `attenuation` | weather_radar |
| `r_bioluminescence` | Bioluminescent Emission | `optical_properties`, `depth` | bioluminescence_tomo |
| `r_diffuse_optical` | Diffuse Optical | `μ_a`, `μ_s'`, `g` | dot, nirs_brain |

#### Σ — Accumulate (20+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `sigma_temporal` | Temporal Integration | `T`, `dt` | ultrasound, pet, fluoroscopy, cacti, fmri, sonar, flim |
| `sigma_spectral` | Spectral Sum | `n_bands` | cassi, sd_cassi, hyperspectral_remote |
| `sigma_angular` | Angular Sum | `n_angles` | fpm |
| `sigma_phase` | Phase-Shift Sum | `n_phases` | sim |
| `sigma_focus` | Focus Stack Sum | `n_planes` | panorama |
| `sigma_energy` | Energy Window Sum | `energy_range` | spect |
| `sigma_interference` | Interferometric Sum | `reference_path` | oct, octa, electron_holography |
| `sigma_volume` | Volume Rendering | `step_size` | nerf |
| `sigma_alpha` | Alpha Compositing | `depth_order` | gaussian_splatting |
| `sigma_correlation` | Correlation Integration | `n_correlations` | tof_camera |
| `sigma_return` | Return Signal Integration | `gate_width` | lidar |
| `sigma_gw` | GW Integration | `duration` | gravitational_wave |
| `sigma_calorimetry` | Shower Energy Sum | `layers` | particle_calorimetry |
| `sigma_ghost` | Ghost Imaging Correlation | `n_measurements` | ghost_imaging |
| `sigma_cup` | CUP Temporal Sum | `DMD_pattern` | cup |
| `sigma_hdr` | Multi-Exposure Merge | `exposure_ratios` | hdr_imaging |
| `sigma_ftir` | Interferogram Integration | `OPD_range` | ftir_imaging |
| `sigma_streak` | Streak Temporal Sum | `sweep_rate` | streak_camera |
| `sigma_passive` | Passive Radiometric | `integration_time` | passive_microwave |
| `sigma_multispectral` | Multi-Band Sum | `bands` | multispectral_sat, ocean_color |

#### S — Sample (10+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `s_raster` | Raster Scan | `step_size`, `dwell_time` | stem, fib_sem |
| `s_spectral_bin` | Spectral Binning | `energy_range`, `resolution` | eels, sims, desi, maldi_msi |
| `s_field_evaporation` | Field Evaporation | `voltage`, `rate` | atom_probe |
| `s_lidar_scan` | LiDAR Scan | `angular_rate`, `pattern` | lidar |
| `s_acoustic_array` | Acoustic Array | `element_positions`, `coupling` | acoustic_emission |
| `s_visibility_sample` | Visibility Sampling | `uv_coverage`, `baseline` | radio_interferometry, insar, radio_astronomy, eht_imaging |
| `s_diffraction_spot` | Diffraction Spot Integration | `resolution`, `mosaicity` | xray_crystallography, neutron_diffraction |
| `s_cantilever` | Cantilever Raster | `speed`, `setpoint` | afm |
| `s_tunnel_current` | Tunneling Current | `bias`, `setpoint` | stm |
| `s_block_illumination` | Block Illumination Sampling | `block_size` | spc_block |

#### W — Disperse (5+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `w_prism` | Prism Dispersion | `angle`, `aperture`, `slope` | cassi, sd_cassi |
| `w_grating` | Diffraction Grating | `groove_density`, `order` | hyperspectral_remote |
| `w_energy_bin` | Energy-Dispersive Binning | `thresholds`, `charge_sharing` | spectral_ct |

#### Λ — Filter (3+ families)

| Family ID | Display Name | Parameters | Modalities |
|-----------|-------------|------------|------------|
| `lambda_dual_energy` | Dual-Energy Selection | `E1`, `E2` | dexa |
| `lambda_energy_disperser` | Energy Loss Disperser | `spectrometer`, `aperture` | eels |
| `lambda_k_edge` | K-Edge Subtraction | `element`, `edge_energy` | spectral_ct (variant) |

#### D — Detect (12+ families)

| Family ID | Display Name | Response Family | Modalities |
|-----------|-------------|----------------|------------|
| `d_scmos` | Scientific CMOS | linear_intensity | widefield, SIM, lightsheet, ... |
| `d_ccd` | CCD | linear_intensity | TEM, SEM, ... |
| `d_emccd` | EMCCD | poisson_rate | PALM/STORM, TIRF, ... |
| `d_pmt` | Photomultiplier Tube | poisson_rate | confocal, two-photon, STED |
| `d_spad` | SPAD Array | poisson_rate | flash_lidar, FLIM |
| `d_flat_panel` | Flat-Panel Detector | linear_intensity | CT, X-ray, mammography |
| `d_rf_coil` | RF Coil | coherent_field | MRI, fMRI, MRS |
| `d_piezo_transducer` | Piezo Transducer | linear_intensity | ultrasound, sonar |
| `d_scintillation` | Scintillation Detector | poisson_rate | PET, SPECT |
| `d_direct_electron` | Direct Electron Detector | poisson_rate | cryo_em, 4D-STEM |
| `d_coded_aperture_det` | Coded-Aperture Detector | linear_intensity | CASSI, CACTI |
| `d_single_pixel` | Single-Pixel (Bucket) | linear_intensity | SPC, ghost imaging |
| `d_ion_detector` | Ion Detector | poisson_rate | SIMS, MALDI, atom probe |
| `d_bolometer` | Bolometer | linear_intensity | THz, passive microwave |
| `d_cantilever` | Cantilever Deflection | linear_intensity | AFM, MFM |

---

## 3. Modality Template Registry (canonical DAGs + priors)

Each modality stores a **template over the 11 primitives** — not a separate primitive set. This is the link between the abstract primitive basis and concrete imaging systems.

### Schema: `modality_templates`

| Field | Type | Description |
|-------|------|-------------|
| `modality_id` | `str` | Unique key (e.g., `cassi`, `mri`, `ct`) |
| `display_name` | `str` | Human-readable name |
| `category` | `str` | One of 19 categories |
| `carrier_family` | `str` | `photon`, `electron`, `spin_rf`, `acoustic`, `xray`, `gamma`, `rf`, `ion`, ... |
| `canonical_dag` | `str` | Primitive chain (e.g., `M → W → Σ → D`) |
| `dag_nodes` | `list[dict]` | Ordered list of `{primitive_id, default_operator_family, label}` |
| `dag_edges` | `list[tuple]` | Edge list for non-linear DAGs (fan-in, branches) |
| `parameter_priors` | `dict` | Plausible parameter ranges per node |
| `mismatch_parameters` | `list[dict]` | B2 mismatch parameters with nominal/range/unit |
| `benchmark_mappings` | `dict` | Pointers to B1/B2/B3/B4 benchmark specs |
| `solver_compatibility` | `list[str]` | Compatible reconstruction solvers |
| `maturity_level` | `enum` | `M0`–`M4` |
| `e_tier2` | `float` | Measured ε-fidelity (if validated) |
| `validation_level` | `str` | `full`, `held_out`, `exotic`, `template` |
| `keywords` | `list[str]` | Search keywords for prompt matching |
| `description` | `str` | Natural-language description of the imaging system |

### Example: CASSI Template

```yaml
cassi:
  display_name: "Coded Aperture Snapshot Spectral Imaging"
  category: compressive_imaging
  carrier_family: photon
  canonical_dag: "M → W → Σ → D"
  dag_nodes:
    - primitive_id: M
      default_operator_family: m_coded_aperture
      label: "Coded Aperture Mask"
    - primitive_id: W
      default_operator_family: w_prism
      label: "Prism Dispersion"
    - primitive_id: Σ
      default_operator_family: sigma_spectral
      label: "Spectral Integration"
    - primitive_id: D
      default_operator_family: d_coded_aperture_det
      label: "Focal Plane Array"
  dag_edges:
    - [0, 1]
    - [1, 2]
    - [2, 3]
  parameter_priors:
    M:
      mask_shift_dx: {nominal: 0, range: [-3.0, 3.0], unit: px}
      mask_shift_dy: {nominal: 0, range: [-3.0, 3.0], unit: px}
      mask_rotation: {nominal: 0, range: [-2.0, 2.0], unit: deg}
    W:
      dispersion_slope: {nominal: 2.0, range: [1.5, 2.5], unit: px/band}
      dispersion_offset: {nominal: 0, range: [-0.5, 0.5], unit: px}
    D:
      gain: {nominal: 1.0, range: [0.9, 1.1]}
      read_noise: {nominal: 5.0, range: [1.0, 15.0], unit: e-}
  mismatch_parameters:
    - {name: mask_shift_dx, nominal: 0, range: [-3.0, 3.0], unit: px}
    - {name: mask_shift_dy, nominal: 0, range: [-3.0, 3.0], unit: px}
    - {name: mask_rotation, nominal: 0, range: [-2.0, 2.0], unit: deg}
    - {name: dispersion_slope, nominal: 2.0, range: [1.5, 2.5], unit: px/band}
    - {name: gain, nominal: 1.0, range: [0.9, 1.1]}
    - {name: read_noise, nominal: 5.0, range: [1.0, 15.0], unit: e-}
  solver_compatibility: [gap_tv, admm_tv, deep_unfolding, pnp_hqs]
  maturity_level: M3
  e_tier2: 1.0e-4
  validation_level: full
```

### Example: MRI Template

```yaml
mri:
  display_name: "Magnetic Resonance Imaging"
  category: medical_imaging
  carrier_family: spin_rf
  canonical_dag: "M → F → S → D"
  dag_nodes:
    - primitive_id: M
      default_operator_family: m_rf_pulse
      label: "RF Pulse Sequence"
    - primitive_id: F
      default_operator_family: f_cartesian_kspace
      label: "k-Space Encoding"
    - primitive_id: S
      default_operator_family: s_kspace_undersample
      label: "Undersampling Pattern"
    - primitive_id: D
      default_operator_family: d_rf_coil
      label: "RF Coil Array"
  parameter_priors:
    M:
      flip_angle: {nominal: 90, range: [5, 180], unit: deg}
      TR: {nominal: 500, range: [5, 10000], unit: ms}
      TE: {nominal: 30, range: [1, 500], unit: ms}
    F:
      acceleration: {nominal: 4, range: [2, 8]}
      calib_size: {nominal: 24, range: [16, 48], unit: lines}
    S:
      sampling_pattern: {nominal: random, options: [random, equispaced, poisson_disk]}
    D:
      n_coils: {nominal: 8, range: [1, 64]}
      coil_sensitivity_error: {nominal: 0, range: [0, 15], unit: "%"}
  maturity_level: M3
  e_tier2: 1.0e-6
  validation_level: full
```

---

## 4. Prompt-to-Graph Layer (retrieval + composition)

**Soft retrieval, not hard classification.** The system retrieves top-K modality templates, instantiates DAGs, scores them, and returns the best with alternatives.

### Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PARSE PROMPT                                             │
│    Extract constraints: carrier, resolution, FOV, speed,    │
│    dose, wavelength, geometry, budget, sample type          │
├─────────────────────────────────────────────────────────────┤
│ 2. RETRIEVE TOP-K TEMPLATES                                 │
│    Soft match against modality_templates.keywords,          │
│    carrier_family, parameter_priors. Return K=5 candidates. │
├─────────────────────────────────────────────────────────────┤
│ 3. INSTANTIATE DAGs                                         │
│    For each template:                                       │
│    - Select operator families per node                      │
│    - Fill parameters from priors + prompt constraints       │
│    - Resolve any branching / fan-in                         │
├─────────────────────────────────────────────────────────────┤
│ 4. SCORE CANDIDATES                                         │
│    - Feasibility (are parameters physically valid?)         │
│    - Recoverability (condition number, compression ratio)   │
│    - Noise performance (expected SNR)                       │
│    - Mismatch risk (sensitivity to parameter errors)        │
│    - Cost/complexity                                        │
├─────────────────────────────────────────────────────────────┤
│ 5. RETURN BEST + ALTERNATIVES                               │
│    Best graph with full operator metadata.                  │
│    Alternative templates with trade-off explanations.       │
├─────────────────────────────────────────────────────────────┤
│ 6. USER REFINES                                             │
│    "More compact" / "Lower dose" / "Higher spectral res"   │
│    → Re-score / swap operators / reorganize DAG             │
│    → Return updated graph                                   │
└─────────────────────────────────────────────────────────────┘
```

### Example: Prompt-Driven Design Session

```
User: "Design a system to image brain tissue with two-photon
       excitation and add a deformable mirror for adaptive optics"

Step 1 — Parse:
    carrier: photon
    sample: brain tissue (scattering)
    excitation: two-photon
    extras: adaptive optics (deformable mirror)

Step 2 — Retrieve:
    #1: two_photon      → C → D
    #2: adaptive_optics → M → C → D
    #3: confocal_3d     → C → D

Step 3 — Instantiate:
    Merge two_photon + adaptive_optics:
    M(m_deformable_mirror) → C(c_two_photon_psf, with AO correction) → D(d_pmt)

Step 4 — Score:
    Feasibility: OK (AO + 2P is standard)
    SNR: ~25 dB at 200 um depth
    Mismatch risk: medium (scattering model sensitive)

Step 5 — Return:
    Design: M(DM, 140 actuators) → C(PSF_2P_AO, σ=0.8 px) → D(PMT, g=1e6)
    Alternative: Without AO → C(PSF_2P, σ=1.5 px) → D(PMT) (simpler, worse resolution)

Step 6 — User refines:
    "Also add temporal gating for FLIM"
    → M(DM) → C(PSF_2P_AO) → R(fluorescence_lifetime, τ) → D(TCSPC)
    New operator family r_fluorescence_lifetime inserted;
    detector swapped from PMT to TCSPC.
```

---

## 5. Practical Data Model (Minimal Schema)

```
┌──────────────────────┐    ┌──────────────────────────┐
│     primitives       │    │    operator_families     │
│ (11 rows, stable)    │◄───│ (200+ rows, growing)     │
│                      │    │                          │
│ primitive_id (PK)    │    │ operator_family_id (PK)  │
│ name                 │    │ primitive_id (FK)        │
│ physics_stage        │    │ physics_family           │
│ type_signature       │    │ analytic_form            │
│ forward_contract     │    │ parameter_schema         │
│ adjoint_contract     │    │ backend_implementations  │
│ constraints          │    │ adjoint_validation       │
└──────────────────────┘    └──────────────────────────┘
         │                              │
         │                              ▼
         │                  ┌──────────────────────────┐
         │                  │ operator_implementations  │
         │                  │ (backend-specific code)   │
         │                  │                          │
         │                  │ impl_id (PK)             │
         │                  │ operator_family_id (FK)  │
         │                  │ backend (torch/jax/cuda) │
         │                  │ code_path                │
         │                  │ tested                   │
         │                  └──────────────────────────┘
         │
         ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  modality_templates  │    │     template_nodes       │
│ (168 rows)           │◄───│ (ordered per template)   │
│                      │    │                          │
│ modality_id (PK)     │    │ template_id (FK)         │
│ display_name         │    │ node_index               │
│ category             │    │ primitive_id (FK)        │
│ carrier_family       │    │ operator_family_id (FK)  │
│ canonical_dag        │    │ label                    │
│ maturity_level       │    │ parameter_priors         │
│ e_tier2              │    └──────────────────────────┘
│ validation_level     │
│ description          │    ┌──────────────────────────┐
│ keywords             │    │     template_edges       │
└──────────────────────┘    │ (DAG connectivity)       │
         │                  │                          │
         │                  │ template_id (FK)         │
         ├─────────────────►│ source_node_index        │
         │                  │ target_node_index        │
         │                  └──────────────────────────┘
         │
         ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  mismatch_models     │    │   calibration_routines   │
│ (per modality)       │    │ (per modality)           │
│                      │    │                          │
│ modality_id (FK)     │    │ modality_id (FK)         │
│ parameter_name       │    │ routine_name             │
│ nominal              │    │ parameters_estimated     │
│ mismatch_range       │    │ algorithm                │
│ unit                 │    │ convergence_criteria     │
│ sensitivity          │    └──────────────────────────┘
└──────────────────────┘
         │
         ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  parameter_schemas   │    │   parameter_priors       │
│ (per operator family)│    │ (per modality + node)    │
│                      │    │                          │
│ operator_family_id   │    │ modality_id (FK)         │
│ param_name           │    │ node_index               │
│ param_type           │    │ param_name               │
│ unit                 │    │ prior_distribution       │
│ valid_range          │    │ nominal                  │
│ default_value        │    │ range                    │
└──────────────────────┘    └──────────────────────────┘

┌──────────────────────┐    ┌──────────────────────────┐
│   solver_adapters    │    │   benchmark_specs        │
│                      │    │ (B1/B2/B3/B4)            │
│ solver_id            │    │                          │
│ modality_id (FK)     │    │ modality_id (FK)         │
│ algorithm            │    │ benchmark_type           │
│ hyperparameters      │    │ data_file                │
│ expected_psnr        │    │ metric                   │
│ expected_rho         │    │ expected_range           │
└──────────────────────┘    └──────────────────────────┘

┌──────────────────────┐
│   runbundles /       │
│   triad_reports      │
│                      │
│ bundle_id            │
│ modality_id (FK)     │
│ solver_id (FK)       │
│ metrics (PSNR, SSIM) │
│ artifacts            │
│ provenance           │
└──────────────────────┘
```

### Table Counts

| Table | Rows (initial) | Growth Rate |
|-------|:--------------:|:-----------:|
| `primitives` | 11 | Rare (extension protocol) |
| `operator_families` | ~200 | Moderate (new physics models) |
| `operator_implementations` | ~300 | Active (new backends, optimizations) |
| `modality_templates` | 168 | Moderate (new modalities) |
| `template_nodes` | ~500 | Follows modality_templates |
| `template_edges` | ~500 | Follows modality_templates |
| `mismatch_models` | ~794 | Follows modality_templates |
| `parameter_schemas` | ~1000 | Follows operator_families |
| `parameter_priors` | ~2000 | Follows modality_templates |
| `calibration_routines` | ~168 | Follows modality_templates |
| `solver_adapters` | ~300 | Active |
| `benchmark_specs` | ~672 | 168 × 4 benchmarks |
| `runbundles` | Growing | Active |

---

## 6. Mapping to Existing Code

| Architecture Layer | Existing File | Status |
|-------------------|--------------|--------|
| Primitive types | `packages/pwm_core/pwm_core/graph/ir_types.py` → `CanonicalPrimitive` enum | Planned (Phase 1 of `plan_canonical_primitives.md`) |
| Implementation primitives | `packages/pwm_core/contrib/primitives.yaml` | Exists (~30 primitives) |
| Canonical decompositions | `packages/pwm_core/pwm_core/graph/canonical_decompositions.py` | Planned (Phase 2) |
| Graph templates | `packages/pwm_core/contrib/graph_templates.yaml` | Exists (65 modalities) |
| Modality definitions | `packages/pwm_core/contrib/modalities.yaml` | Exists (65 modalities) |
| Mismatch models | `packages/pwm_core/contrib/mismatch_db.yaml` | Exists |
| Solver registry | `packages/pwm_core/contrib/solver_registry.yaml` | Exists |
| Benchmark data | `platform/pwm_platform/static/benchmark-data/v1.0/` | Exists (65 variants) |
| Benchmark specs | `docs/pwm_modality_benchmarks_detailed.md` | Exists (168 modalities) |
| Registry conventions | `docs/contracts/registry_conventions.md` | FROZEN |
| RunBundle schema | `docs/contracts/runbundle_schema.md` | FROZEN |

### What Needs to Be Built

1. **Extend `primitives.yaml`**: Add `canonical_id` field to each entry, mapping to one of 11 canonical types
2. **Create `operator_families.yaml`**: New registry with the ~200 operator families listed above
3. **Extend `graph_templates.yaml`**: Add 103 new modality templates (168 - 65 existing)
4. **Extend `modalities.yaml`**: Add 103 new modality definitions
5. **Implement `CanonicalPrimitive` enum**: Phase 1 of `plan_canonical_primitives.md`
6. **Implement prompt-to-graph retrieval**: New module in `packages/pwm_core/pwm_core/agents/`
7. **Add ε-fidelity validation**: Phase 3 of `plan_canonical_primitives.md`
8. **Add extension protocol**: Phase 4 of `plan_canonical_primitives.md`

---

## 7. Theoretical Caveat

The correct claim:

> **"The 11-primitive basis is sufficient for all imaging modalities in the defined operator class C_img (finite-stage compositions with bounded linear/Lipschitz stages), and empirically covers all 168 modalities tested. An extension protocol exists for when a representation gap appears."**

This means:
- The FPB Theorem gives **ε-approximate** representation, not exact
- The operator class C_img is **defined** (bounded, finite-stage) — pathological operators outside this class may require extension
- The 11-primitive count has **saturated empirically** (no new primitives needed from modality 10 through modality 168)
- The extension protocol is **falsifiable** — if a new modality genuinely can't be represented within ε, the protocol adds a new primitive (as happened with `R` for Compton scattering and `Λ` for energy selection)

The right product claim is: **"Universal under a falsifiable extension protocol"**, not "absolutely final forever."

---

## 8. Alignment with Existing PWM Plans

This architecture is consistent with and extends:

| Document | Relationship |
|----------|-------------|
| `plan_canonical_primitives.md` | Phases 1–6 implement the code-level primitive registry and canonical decompositions |
| `plan_operatorgraph_foundation.md` | Typed DAG semantics and graph compilation |
| `contracts/registry_conventions.md` | ID format (`<family>_<name>_v<N>`) applies to all new entries |
| `contracts/runbundle_schema.md` | RunBundles link to modality_templates via `spec_id` |
| `pwm_modality_benchmarks_detailed.md` | Source of truth for 168 modality specs, mismatch ranges, benchmark definitions |
| `primitives_operators_database.md` | Flat catalogue of all operators (reference data, not architecture) |
