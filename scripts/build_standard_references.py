#!/usr/bin/env python3
"""Generate reference.json for each modality's standard/ directory.

Parses standard_references.md and writes per-modality reference.json files
with: canonical_reference, doi_link, forward_model, measurement_matrix,
ground_truth description, and reconstruction info.
"""
import json, re, os, sys
from pathlib import Path
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
REF_MD = ROOT / "datasets" / "benchmark" / "standard_references.md"
BENCH_DIR = ROOT / "datasets" / "benchmark"

# Ground truth descriptions per modality (what x represents physically)
GROUND_TRUTHS = {
    "acoustic_emission": "Acoustic emission source locations and amplitudes in material",
    "acoustic_microscopy": "Acoustic reflectance map R(x,y) of sample surface",
    "active_thermography": "Subsurface defect map (thermal diffusivity distribution)",
    "adaptive_optics": "True astronomical object brightness distribution",
    "afm": "Surface topography z(x,y) at atomic resolution",
    "angiography": "Vascular contrast map (vessel attenuation distribution)",
    "asl_mri": "Cerebral blood flow (CBF) map in mL/100g/min",
    "atom_probe": "3D atomic composition map (element identity at each position)",
    "bioluminescence_tomo": "3D bioluminescent source distribution S(r)",
    "brachytherapy_img": "3D dose distribution around radioactive seeds",
    "brillouin": "Brillouin frequency shift map (elastic modulus distribution)",
    "cacti": "3D spatiotemporal video volume x(x,y,t)",
    "cars": "CARS spectral image (molecular vibration distribution)",
    "cassi": "3D hyperspectral data cube x(x,y,lambda)",
    "cathodoluminescence": "Hyperspectral CL map (photon emission vs position and wavelength)",
    "cbct": "3D attenuation volume mu(x,y,z)",
    "cest_mri": "CEST contrast map (amide proton transfer ratio)",
    "ceus": "Contrast-enhanced ultrasound perfusion map",
    "clem": "Correlated fluorescence + electron microscopy image pair",
    "coded_exposure": "Sharp latent image (deblurred scene)",
    "confocal_3d": "3D fluorescence distribution rho(x,y,z)",
    "confocal_endomicroscopy": "Tissue fluorescence image at cellular resolution",
    "confocal_livecell": "Live cell fluorescence image",
    "coronagraphy": "Exoplanet/circumstellar disk image (high-contrast scene)",
    "cryo_em": "3D electron density map of macromolecular complex",
    "cryo_et": "3D electron density volume from cryo-electron tomography",
    "ct": "2D attenuation cross-section mu(x,y)",
    "ct_fluorescence": "3D fluorophore concentration distribution c(r)",
    "cup": "3D spatiotemporal scene x(x,y,t) at ultrafast timescales",
    "dark_field": "Dark-field scattering image (sub-pixel microstructure)",
    "desi": "2D molecular composition map (mass spectrum per pixel)",
    "dexa": "Bone mineral density (BMD) map in g/cm^2",
    "dic": "Displacement field u(x,y) (surface deformation)",
    "diffusion_mri": "Diffusion tensor field D(x,y,z) (fiber orientation)",
    "digital_breast_tomo": "3D breast tissue volume mu(x,y,z)",
    "dna_paint": "Super-resolution molecular map (emitter positions)",
    "doppler_ultrasound": "Blood flow velocity map v(x,y)",
    "dot": "3D optical absorption coefficient distribution mu_a(r)",
    "ebsd": "Crystal orientation map (Euler angles per pixel)",
    "eddy_current": "Conductivity defect map sigma(r) in material",
    "edx_mapping": "Elemental composition map (element concentration per pixel)",
    "eels": "Electron energy-loss spectrum / elemental map",
    "eht_imaging": "Radio brightness distribution of black hole shadow",
    "elastography": "Tissue shear modulus (stiffness) map mu(r)",
    "electron_diffraction": "Crystal potential V(r) (atomic structure)",
    "electron_holography": "Phase and amplitude of electron wave (electromagnetic potentials)",
    "electron_tomography": "3D object density from electron tilt series",
    "endoscopy": "Tissue surface image (gastrointestinal mucosa)",
    "entangled_photon": "Object transmission function T(x,y)",
    "event_camera": "Latent intensity image L(x,y,t) at high temporal resolution",
    "expansion": "Sub-diffraction fluorescence distribution (expanded sample)",
    "fib_sem": "3D volume of material (serial section electron images)",
    "flash_lidar": "3D depth map d(x,y) + reflectance",
    "flim": "Fluorescence lifetime map tau(x,y)",
    "fluoroscopy": "Real-time X-ray projection (tissue attenuation)",
    "fmri": "Brain activation map (BOLD signal change)",
    "fpm": "High-resolution complex field (amplitude + phase) of sample",
    "ftir_imaging": "Infrared absorption spectrum per pixel (chemical composition)",
    "fundus": "Retinal fundus image (vasculature and pathology)",
    "fwi": "Subsurface velocity model v(x,y,z)",
    "gaussian_splatting": "3D scene as Gaussian primitives (positions, covariances, colors)",
    "ghost_imaging": "Object transmission/reflectance image T(x,y)",
    "gpr": "Subsurface dielectric permittivity profile",
    "gravitational_wave": "Gravitational wave strain waveform h(t)",
    "hdr_imaging": "High dynamic range radiance map E(x,y)",
    "holography": "Complex object field (amplitude + phase)",
    "hyperspectral_remote": "Hyperspectral data cube (endmember abundances)",
    "impedance_tomo": "Conductivity distribution sigma(x,y) inside body",
    "industrial_ct": "3D density/attenuation volume of manufactured part",
    "insar": "Surface displacement / topographic phase map",
    "integral": "4D light field L(u,v,s,t)",
    "ism": "Super-resolution fluorescence image",
    "ivus": "Cross-sectional vessel wall image (intravascular ultrasound)",
    "lattice_lightsheet": "3D fluorescence distribution in live specimen",
    "lensless": "2D scene image x(x,y)",
    "libs": "Elemental composition (concentration map from plasma emission)",
    "lidar": "Range-resolved backscatter profile / 3D point cloud",
    "light_field": "4D light field L(u,v,s,t) from plenoptic camera",
    "lightsheet": "3D fluorescence volume of specimen",
    "lucky_imaging": "Diffraction-limited astronomical image",
    "machine_vision": "Sharp object image (lens-degraded scene)",
    "magnetic_particle": "3D tracer concentration distribution c(r)",
    "maldi_msi": "2D molecular distribution map (mass spectrum per pixel)",
    "mammography": "Breast tissue attenuation projection",
    "matrix": "Complete low-rank matrix X (from partial observations)",
    "mfm": "Magnetic domain structure / stray field gradient",
    "minflux": "Super-resolution emitter positions (1-5 nm precision)",
    "mr_elastography": "Tissue shear modulus map from MRI-encoded displacements",
    "mr_fingerprinting": "Quantitative T1, T2, proton density maps",
    "mra": "Vascular anatomy image (MR angiogram)",
    "mri": "Proton density / tissue contrast image rho(x,y)",
    "mrs": "Metabolite concentration spectrum (chemical shift frequencies)",
    "multispectral_sat": "Multispectral surface reflectance (land cover classification)",
    "muon_tomo": "3D density/atomic-number distribution (nuclear material detection)",
    "nerf": "3D neural radiance field (density + color)",
    "neutron_diffraction": "Crystal structure (atomic positions from neutron scattering)",
    "neutron_tomo": "3D neutron attenuation cross-section volume",
    "nirs_brain": "Cerebral hemoglobin concentration changes (HbO, HbR)",
    "nsom": "Near-field optical image at sub-wavelength resolution",
    "ocean_acoustic_tomo": "Ocean sound speed distribution c(r)",
    "ocean_color": "Inherent optical properties (chlorophyll, CDOM, particulates)",
    "oct": "Depth-resolved reflectivity profile R(z) of tissue",
    "octa": "Retinal vasculature flow map (angiogram)",
    "odt": "3D refractive index distribution n(x,y,z)",
    "palm_storm": "Super-resolution image (single-molecule localizations)",
    "panorama": "360-degree equirectangular panoramic image",
    "particle_calorimetry": "Particle energy and shower profile in calorimeter",
    "passive_microwave": "Brightness temperature map T_B(theta, nu)",
    "pet": "3D radiotracer activity distribution f(x,y,z)",
    "pet_ct": "Co-registered PET activity + CT attenuation volumes",
    "pet_mr": "Co-registered PET activity + MRI tissue contrast",
    "phase_contrast": "X-ray phase shift map phi(x,y)",
    "phase_retrieval": "Complex field amplitude and phase from Fourier magnitudes",
    "photoacoustic": "Initial pressure distribution p0(r) (optical absorption)",
    "photometric_stereo": "Surface normal map n(x,y) and albedo rho(x,y)",
    "polarization": "Stokes vector image S(x,y) = (S0, S1, S2, S3)",
    "polsar": "Polarimetric scattering matrix / land classification",
    "portal_imaging": "MV X-ray portal image (patient position verification)",
    "proton_radiography": "Water-equivalent path length (WEPL) map / material distribution",
    "proton_therapy_img": "3D dose distribution in patient",
    "ptychography": "Complex object transmission function O(r) (amplitude + phase)",
    "pump_probe": "Transient absorption spectra DeltaA(lambda, tau)",
    "quantum_illumination": "Target reflectivity image (low-signal detection)",
    "radio_astronomy": "Radio sky brightness distribution I(l,m)",
    "radio_interferometry": "Radio sky brightness from VLBI sparse visibilities",
    "raman_imaging": "Chemical composition map (Raman spectrum per pixel)",
    "sar": "Complex reflectivity map sigma(x,y) (radar cross-section)",
    "saxs": "Electron density contrast profile rho(r) (nanostructure)",
    "sd_cassi": "3D hyperspectral data cube x(x,y,lambda) (single-disperser CASSI)",
    "seismic_tomo": "Subsurface seismic velocity distribution v(r)",
    "sem": "Surface morphology image (secondary electron yield)",
    "shearography": "Surface strain / displacement derivative map",
    "shg": "Second-harmonic generation image (non-centrosymmetric structures)",
    "sim": "Super-resolution fluorescence image (2x resolution improvement)",
    "sims": "3D elemental/isotopic composition map (depth profiling)",
    "solar_imaging": "Solar EUV/X-ray image (coronal plasma distribution)",
    "sonar": "Underwater acoustic reflectivity map / bathymetry",
    "spc_kronecker": "2D image from Kronecker-structured compressed sensing",
    "spect": "3D radiotracer distribution (gamma-ray emission)",
    "spect_ct": "Co-registered SPECT activity + CT attenuation volumes",
    "spectral_ct": "Material-decomposed images (basis material densities)",
    "spinning_disk": "3D fluorescence volume (fast confocal)",
    "srs": "Stimulated Raman scattering chemical image",
    "sted": "Super-resolution fluorescence image (sub-diffraction)",
    "stem": "Atomic-resolution image (Z-contrast HAADF or BF)",
    "stm": "Surface electronic structure / LDOS at atomic scale",
    "streak_camera": "3D spatiotemporal scene x(x,y,t) (streak imaging)",
    "structured_light": "3D surface depth map from fringe projection",
    "swi": "Susceptibility-weighted MRI image (venous vasculature)",
    "talbot_lau": "X-ray absorption, phase, and dark-field images",
    "tem": "2D projected potential image of specimen",
    "terahertz": "THz absorption/refractive index image of sample",
    "three_photon": "Deep-tissue 3D fluorescence (three-photon excitation)",
    "tirf": "Surface-proximal fluorescence image (evanescent excitation)",
    "tof_camera": "Depth map d(x,y) from time-of-flight sensor",
    "two_photon": "3D fluorescence volume (two-photon excitation)",
    "ultrasonic_phased_array": "Ultrasonic reflectivity image (NDE defect map)",
    "ultrasound": "2D tissue reflectivity image sigma(x,y)",
    "us_mri": "Ultrashort TE MRI image (short T2* tissues)",
    "waxs": "Electron density / crystal structure (wide-angle diffraction)",
    "weather_radar": "Radar reflectivity map Z(r, theta) (precipitation)",
    "widefield": "2D fluorescence or brightfield image",
    "widefield_lowdose": "Low-dose fluorescence image (denoised ground truth)",
    "xfel_sfx": "Crystal electron density from serial femtosecond diffraction",
    "xray_crystallography": "3D electron density rho(x,y,z) from structure factor magnitudes",
    "xray_ndt": "3D attenuation volume of industrial object (defect detection)",
    "xray_radiography": "2D X-ray projection (tissue/material attenuation)",
    "xrf_imaging": "Elemental concentration map from X-ray fluorescence",
}

# Reconstruction methods per modality
RECONSTRUCTIONS = {
    "acoustic_emission": "Time-difference-of-arrival (TDOA) source localization + beamforming",
    "acoustic_microscopy": "Deconvolution with acoustic PSF (Wiener filter)",
    "active_thermography": "Pulse phase thermography + thermal diffusivity inversion",
    "adaptive_optics": "Wavefront sensing + deformable mirror correction + deconvolution",
    "afm": "Tip deconvolution (blind tip estimation + erosion)",
    "angiography": "Digital subtraction angiography (DSA) + vessel enhancement",
    "asl_mri": "Kinetic model fitting (Buxton general kinetic model)",
    "atom_probe": "Reverse point projection + mass-to-charge identification",
    "bioluminescence_tomo": "Diffusion-based inverse solver (Tikhonov / L1 regularization)",
    "brachytherapy_img": "TG-43 dose calculation + optimization",
    "brillouin": "Lorentzian spectral fitting (frequency shift → modulus)",
    "cacti": "GAP-TV (Generalized Alternating Projection with TV regularization)",
    "cars": "Maximum entropy method (MEM) / Kramers-Kronig phase retrieval",
    "cassi": "GAP-TV / TwIST / deep unfolding (MST, CST)",
    "cathodoluminescence": "Spectral unmixing + deconvolution",
    "cbct": "FDK (Feldkamp-Davis-Kress) cone-beam filtered backprojection",
    "cest_mri": "Lorentzian difference analysis + Bloch-McConnell fitting",
    "ceus": "Pulse inversion + contrast-specific beamforming",
    "clem": "Image registration (affine/deformable) + overlay",
    "coded_exposure": "Wiener deconvolution (flutter shutter code has flat spectrum)",
    "confocal_3d": "Deconvolution (Richardson-Lucy / regularized inverse)",
    "confocal_endomicroscopy": "Fiber pattern removal + super-resolution reconstruction",
    "confocal_livecell": "CARE denoising + deconvolution",
    "coronagraphy": "Coronagraphic subtraction (ADI / RDI / PCA-based)",
    "cryo_em": "Single-particle reconstruction (RELION / cryoSPARC back-projection + refinement)",
    "cryo_et": "Weighted back-projection (WBP) / SIRT / IMOD",
    "ct": "Filtered backprojection (FBP) / iterative (SART, ADMM-TV)",
    "ct_fluorescence": "Born-normalized fluorescence reconstruction (Tikhonov)",
    "cup": "GAP-TV (treating temporal as spectral, identical to CASSI reconstruction)",
    "dark_field": "Visibility analysis from phase-stepping curve",
    "desi": "Mass spectral deconvolution + spatial interpolation",
    "dexa": "Dual-energy decomposition (bone/soft tissue separation)",
    "dic": "Subset-based DIC correlation + displacement field smoothing",
    "diffusion_mri": "Diffusion tensor estimation (least-squares fit to DW images)",
    "digital_breast_tomo": "Iterative reconstruction (SART / model-based) for limited-angle",
    "dna_paint": "Single-molecule localization (Gaussian fitting / MLE)",
    "doppler_ultrasound": "Autocorrelation estimator (Kasai method) for velocity",
    "dot": "Diffuse optical tomography inversion (Jacobian + regularization)",
    "ebsd": "Hough transform band detection + dictionary indexing",
    "eddy_current": "Impedance inversion (defect characterization from ΔZ)",
    "edx_mapping": "Background subtraction + Cliff-Lorimer quantification",
    "eels": "Background subtraction (power-law) + deconvolution",
    "eht_imaging": "CLEAN / RML (regularized maximum likelihood) / CHIRP",
    "elastography": "Local frequency estimation + Helmholtz inversion for shear modulus",
    "electron_diffraction": "Direct methods / charge flipping / dynamical refinement",
    "electron_holography": "Fourier filtering of hologram (carrier frequency sideband extraction)",
    "electron_tomography": "Weighted backprojection (WBP) / SIRT from tilt series",
    "endoscopy": "Denoising + color correction + image stitching",
    "entangled_photon": "Coincidence image formation (bucket + reference correlation)",
    "event_camera": "Event integration + motion compensation",
    "expansion": "Deconvolution after expansion factor calibration",
    "fib_sem": "Stack alignment + 3D segmentation",
    "flash_lidar": "Photon counting histogram → depth estimation (cross-correlation)",
    "flim": "Exponential decay fitting (single/multi-exponential, phasor approach)",
    "fluoroscopy": "Logarithmic subtraction + edge enhancement",
    "fmri": "General linear model (GLM) + hemodynamic response convolution",
    "fpm": "Fourier ptychographic phase retrieval (alternating projections in Fourier space)",
    "ftir_imaging": "Inverse Fourier transform of interferogram → spectrum per pixel",
    "fundus": "Illumination correction + vessel segmentation",
    "fwi": "Adjoint-state gradient + L-BFGS optimization (waveform inversion)",
    "gaussian_splatting": "Differentiable rendering + gradient descent on Gaussian parameters",
    "ghost_imaging": "Correlation imaging (second-order) / compressed sensing reconstruction",
    "gpr": "Migration (Kirchhoff / reverse-time) + velocity estimation",
    "gravitational_wave": "Matched filtering with template waveform bank",
    "hdr_imaging": "Camera response function calibration + radiance map fusion",
    "holography": "Angular spectrum method / Fresnel propagation back to object plane",
    "hyperspectral_remote": "Spectral unmixing (VCA endmember extraction + FCLS abundance)",
    "impedance_tomo": "Gauss-Newton iteration on FEM-discretized conductivity",
    "industrial_ct": "FBP / iterative reconstruction (SART / CGLS) at high resolution",
    "insar": "Phase unwrapping (SNAPHU) + geocoding",
    "integral": "Light field refocusing (shift-and-add) / depth estimation",
    "ism": "Pixel reassignment (multiview deconvolution for 2x resolution)",
    "ivus": "Polar-to-Cartesian scan conversion + tissue characterization",
    "lattice_lightsheet": "3D deconvolution (Richardson-Lucy / Wiener)",
    "lensless": "Wiener deconvolution / ADMM with TV regularization",
    "libs": "Spectral line identification + Boltzmann plot quantification",
    "lidar": "Inversion of lidar equation (Klett-Fernald method)",
    "light_field": "Depth estimation + view synthesis from 4D field",
    "lightsheet": "Multiview fusion + deconvolution",
    "lucky_imaging": "Frame selection (top percentile) + shift-and-add registration",
    "machine_vision": "Deconvolution / denoising (classical Wiener or learned priors)",
    "magnetic_particle": "System matrix reconstruction (Tikhonov / Kaczmarz iteration)",
    "maldi_msi": "Peak picking + spatial interpolation + normalization",
    "mammography": "Contrast enhancement + computer-aided detection (CAD)",
    "matrix": "Nuclear norm minimization / SVT (Singular Value Thresholding)",
    "mfm": "Stray field deconvolution (transfer function approach)",
    "minflux": "Maximum likelihood position estimation from photon counts",
    "mr_elastography": "Direct inversion of Helmholtz equation for shear modulus",
    "mr_fingerprinting": "Dictionary matching (Bloch simulation fingerprint library)",
    "mra": "Maximum intensity projection (MIP) + vessel segmentation",
    "mri": "Inverse FFT / GRAPPA / SENSE (parallel imaging) / compressed sensing",
    "mrs": "Spectral fitting (LCModel / QUEST / AMARES)",
    "multispectral_sat": "Atmospheric correction + spectral classification",
    "muon_tomo": "POCA / MLP-based iterative reconstruction (scattering density)",
    "nerf": "Volume rendering integral + MLP optimization (gradient descent)",
    "neutron_diffraction": "Rietveld refinement (crystal structure from diffraction peaks)",
    "neutron_tomo": "FBP / iterative reconstruction from neutron projections",
    "nirs_brain": "Modified Beer-Lambert law inversion → ΔHbO, ΔHbR",
    "nsom": "Near-field deconvolution / finite-element inverse modeling",
    "ocean_acoustic_tomo": "Travel time inversion (ray-based or full-wave) for sound speed",
    "ocean_color": "Bio-optical model inversion (semi-analytical / neural network)",
    "oct": "Inverse FFT of spectral interferogram → depth reflectivity",
    "octa": "Speckle variance / decorrelation analysis between repeated B-scans",
    "odt": "Fourier diffraction theorem reconstruction (Rytov/Born approximation)",
    "palm_storm": "Single-molecule localization (Gaussian MLE) + density rendering",
    "panorama": "Image stitching + homography estimation + blending",
    "particle_calorimetry": "Energy sum + shower shape analysis / ML regression",
    "passive_microwave": "Antenna deconvolution + retrieval algorithms",
    "pet": "MLEM / OSEM iterative reconstruction with attenuation correction",
    "pet_ct": "OSEM (PET) + FBP (CT) + CT-based attenuation correction",
    "pet_mr": "MLEM/OSEM (PET) + MR-based attenuation correction",
    "phase_contrast": "Phase retrieval from stepping curve + phase integration",
    "phase_retrieval": "Gerchberg-Saxton / HIO (Hybrid Input-Output) iterative projection",
    "photoacoustic": "Universal back-projection / time-reversal reconstruction",
    "photometric_stereo": "Least-squares normal estimation from multiple light directions",
    "polarization": "Stokes vector calculation from analyzer measurements",
    "polsar": "Polarimetric decomposition (Freeman-Durden / H-A-alpha)",
    "portal_imaging": "Flat-field correction + dose verification comparison",
    "proton_radiography": "Most likely path (MLP) estimation + iterative WEPL reconstruction",
    "proton_therapy_img": "Monte Carlo dose calculation + range verification",
    "ptychography": "ePIE (extended Ptychographic Iterative Engine) / difference map",
    "pump_probe": "Global analysis (multi-exponential kinetic fitting)",
    "quantum_illumination": "Quantum receiver (OPA + photon counting) / classical correlation",
    "radio_astronomy": "CLEAN deconvolution / maximum entropy method (MEM)",
    "radio_interferometry": "CLEAN / MEM / compressed sensing imaging",
    "raman_imaging": "Spectral unmixing + MCR-ALS (multivariate curve resolution)",
    "sar": "Range-Doppler algorithm / omega-k / chirp scaling",
    "saxs": "Indirect Fourier transform (GNOM) / ab initio shape reconstruction (DAMMIN)",
    "sd_cassi": "GAP-TV / TwIST / deep unfolding (same as CASSI)",
    "seismic_tomo": "LSQR / conjugate gradient travel time inversion",
    "sem": "Brightness/contrast adjustment + deconvolution with beam PSF",
    "shearography": "Phase unwrapping + spatial derivative integration",
    "shg": "Nonlinear optical deconvolution / direct imaging",
    "sim": "Frequency-domain reconstruction (3 orientations × 3 phases → 2x resolution)",
    "sims": "Depth profile reconstruction + 3D composition mapping",
    "solar_imaging": "Multi-channel DEM (differential emission measure) analysis",
    "sonar": "Beamforming + matched filter processing",
    "spc_kronecker": "FISTA / OMP with Kronecker-structured sensing matrix",
    "spect": "OSEM with collimator response modeling + attenuation correction",
    "spect_ct": "OSEM (SPECT) + FBP (CT) + CT-based attenuation correction",
    "spectral_ct": "Material decomposition (basis material method) + iterative recon",
    "spinning_disk": "Deconvolution (Richardson-Lucy) + temporal denoising",
    "srs": "Lock-in detection + spectral fitting",
    "sted": "Deconvolution with effective STED PSF",
    "stem": "Deconvolution / exit-wave reconstruction / multislice inversion",
    "stm": "Tersoff-Hamann simulation matching / LDOS extraction",
    "streak_camera": "GAP-TV reconstruction (same as CUP/CASSI, temporal dispersion)",
    "structured_light": "Phase unwrapping + triangulation → 3D point cloud",
    "swi": "Minimum intensity projection + phase mask multiplication",
    "talbot_lau": "Phase stepping analysis (Fourier coefficients of stepping curve)",
    "tem": "CTF correction (Wiener filter / phase flipping) + 3D reconstruction",
    "terahertz": "Deconvolution in time/frequency domain + material parameter extraction",
    "three_photon": "3D deconvolution with three-photon PSF",
    "tirf": "TIRF-specific deconvolution / single-molecule analysis",
    "tof_camera": "Phase-to-depth conversion + multi-frequency unwrapping",
    "two_photon": "3D deconvolution with two-photon PSF (squared intensity)",
    "ultrasonic_phased_array": "Total Focusing Method (TFM) from full matrix capture",
    "ultrasound": "Delay-and-sum beamforming / adaptive beamforming",
    "us_mri": "Non-uniform FFT (NUFFT) / iterative reconstruction for non-Cartesian k-space",
    "waxs": "Rietveld refinement / PDF analysis from wide-angle diffraction",
    "weather_radar": "Radar equation inversion + dual-pol hydrometeor classification",
    "widefield": "Deconvolution (Wiener / Richardson-Lucy)",
    "widefield_lowdose": "CARE (Content-Aware Restoration) deep learning denoising",
    "xfel_sfx": "Monte Carlo integration of partial reflections + molecular replacement",
    "xray_crystallography": "Molecular replacement / SAD/MAD phasing + density modification",
    "xray_ndt": "FBP / iterative reconstruction + defect segmentation",
    "xray_radiography": "Logarithmic subtraction + contrast enhancement",
    "xrf_imaging": "Spectral fitting (element peak deconvolution) + quantification",
}


def parse_references_md():
    """Parse the markdown reference table into a dict keyed by modality."""
    text = REF_MD.read_text(encoding="utf-8")
    refs = {}

    # Match table rows: | # | **modality** | ref | doi | forward | matrix |
    pattern = re.compile(
        r'\|\s*(\d+)\s*\|\s*\*\*(\w+)\*\*\s*\|'
        r'\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|'
    )

    for m in pattern.finditer(text):
        num = int(m.group(1))
        modality = m.group(2)
        canonical_ref = m.group(3).strip()
        doi_link = m.group(4).strip()
        forward_model = m.group(5).strip()
        measurement_matrix = m.group(6).strip()

        # Extract URL from markdown link [text](url)
        link_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', doi_link)
        if link_match:
            doi_text = link_match.group(1)
            doi_url = link_match.group(2)
        else:
            doi_text = doi_link
            doi_url = doi_link

        refs[modality] = {
            "number": num,
            "canonical_reference": canonical_ref,
            "doi": doi_text,
            "link": doi_url,
            "forward_model": forward_model,
            "measurement_matrix": measurement_matrix,
        }

    return refs


def main():
    print("=" * 70)
    print("BUILDING reference.json FOR ALL MODALITY STANDARD DIRECTORIES")
    print("=" * 70)

    refs = parse_references_md()
    print(f"Parsed {len(refs)} references from standard_references.md")

    # Find all standard dirs
    standard_dirs = sorted(BENCH_DIR.glob("*/standard"))
    print(f"Found {len(standard_dirs)} standard/ directories")

    created = 0
    skipped = 0
    missing_ref = []

    for sd in standard_dirs:
        modality = sd.parent.name
        if modality in refs:
            r = refs[modality]
            ref_json = {
                "modality": modality,
                "canonical_reference": r["canonical_reference"],
                "doi": r["doi"],
                "link": r["link"],
                "ground_truth": GROUND_TRUTHS.get(modality, f"Ground truth object/scene for {modality}"),
                "forward_model": r["forward_model"],
                "measurement_matrix": r["measurement_matrix"],
                "reconstruction": RECONSTRUCTIONS.get(modality, f"Standard reconstruction for {modality}"),
            }
            out_path = sd / "reference.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ref_json, f, indent=2, ensure_ascii=False)
            created += 1
        else:
            missing_ref.append(modality)
            skipped += 1

    print(f"\nCreated: {created} reference.json files")
    if missing_ref:
        print(f"Missing reference for {len(missing_ref)} modalities: {missing_ref}")
    print(f"Skipped: {skipped}")

    # Verify a few
    print("\n--- Sample reference.json files ---")
    for mod in ["ct", "mri", "cassi", "cup", "cryo_em"]:
        rpath = BENCH_DIR / mod / "standard" / "reference.json"
        if rpath.exists():
            with open(rpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"\n{mod}:")
            for k, v in data.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
