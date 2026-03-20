#!/usr/bin/env python3
"""PWM Spec.md Keyword Matcher — No-LLM path.

Matches a natural-language query to the best-fit modality and use case,
then displays the corresponding spec.md and optionally runs the solver.

Usage:
    python spec/keyword_match.py "CT reconstruction"
    python spec/keyword_match.py "MRI with mismatch correction"
    python spec/keyword_match.py "design lensless imaging system"
    python spec/keyword_match.py "simulate CT physics"
    python spec/keyword_match.py --list
    python spec/keyword_match.py "CT" --run

API Key (optional):
    python spec/keyword_match.py "sparse CT" --api-key YOUR_GEMINI_KEY
    python spec/keyword_match.py "sparse CT" --gateway-key YOUR_GATEWAY_KEY
    (LLM-guided finetuning not yet implemented; key is reserved for future use)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPEC_DIR = Path(__file__).parent
PUBLIC_DIR = SPEC_DIR.parent

# ---------------------------------------------------------------------------
# Modality keyword database
# Each entry: (modality_id, display_name, category, keywords, has_spec_file)
# ---------------------------------------------------------------------------
MODALITY_DB: List[Tuple[str, str, str, List[str], bool]] = [
    # Medical — Tomography
    ("ct", "X-ray Computed Tomography", "medical_tomo",
     ["ct", "computed tomography", "xray ct", "x-ray ct", "radon", "sinogram", "fbp",
      "filtered back projection", "lodopab", "fan beam", "cone beam", "sparse view",
      "low dose", "lodopab"], True),
    ("cbct", "Cone-Beam CT", "medical_tomo",
     ["cbct", "cone beam ct", "cone-beam", "dental ct", "conebeam", "3d ct",
      "kv imaging", "igrt"], True),
    ("pet", "Positron Emission Tomography", "medical_tomo",
     ["pet", "positron emission", "osem", "mlem pet", "fdg", "radiotracer",
      "nuclear medicine", "annihilation"], True),
    ("spect", "Single Photon Emission CT", "medical_tomo",
     ["spect", "single photon", "gamma camera", "collimator", "scintigraphy",
      "technetium", "nuclear imaging"], True),
    ("spect_ct", "SPECT/CT Fusion", "medical_tomo",
     ["spect ct", "spect/ct", "hybrid spect", "attenuation correction spect"], False),
    ("pet_ct", "PET/CT Fusion", "medical_tomo",
     ["pet ct", "pet/ct", "hybrid pet", "attenuation correction pet"], False),
    ("pet_mr", "PET/MR Fusion", "medical_tomo",
     ["pet mr", "pet/mr", "pet mri", "simultaneous pet mri"], False),
    ("spectral_ct", "Photon-Counting Spectral CT", "medical_tomo",
     ["spectral ct", "photon counting", "energy resolved", "dual energy ct",
      "k-edge", "material decomposition"], False),
    # Medical — MRI
    ("mri", "Magnetic Resonance Imaging", "medical_mri",
     ["mri", "magnetic resonance", "k-space", "kspace", "sense", "espirit",
      "compressed sensing mri", "parallel imaging", "m4raw", "fastmri",
      "undersampled mri", "reconstruction mri"], True),
    ("diffusion_mri", "Diffusion MRI", "medical_mri",
     ["diffusion mri", "dwi", "dti", "dki", "diffusion tensor", "tractography",
      "fiber tracking", "diffusion weighted"], False),
    ("fmri", "Functional MRI", "medical_mri",
     ["fmri", "functional mri", "bold", "brain activation", "hemodynamic"], False),
    ("asl_mri", "Arterial Spin Labeling MRI", "medical_mri",
     ["asl", "arterial spin", "perfusion mri", "cerebral blood flow"], False),
    ("cest_mri", "CEST MRI", "medical_mri",
     ["cest", "chemical exchange saturation", "amide proton transfer"], False),
    ("mr_elastography", "MR Elastography", "medical_mri",
     ["mr elastography", "mre", "stiffness mapping", "shear wave mri"], False),
    ("mr_fingerprinting", "MR Fingerprinting", "medical_mri",
     ["mr fingerprinting", "mrf", "tissue fingerprinting", "dictionary matching mri"], False),
    ("mra", "MR Angiography", "medical_mri",
     ["mra", "mr angiography", "magnetic resonance angiography", "vessel imaging mri"], False),
    ("mrs", "MR Spectroscopy", "medical_mri",
     ["mrs", "mr spectroscopy", "mrsi", "metabolite", "nmr spectroscopy"], False),
    ("swi", "Susceptibility Weighted Imaging", "medical_mri",
     ["swi", "susceptibility weighted", "venography mri", "hemosiderin", "iron mri"], False),
    ("us_mri", "Undersampled MRI", "medical_mri",
     ["undersampled mri", "accelerated mri", "rapid mri", "compressed mri"], False),
    # Medical — Ultrasound
    ("ultrasound", "Ultrasound Imaging", "medical_us",
     ["ultrasound", "us imaging", "das", "delay and sum", "beamforming", "bmode",
      "echocardiography", "abdominal ultrasound"], True),
    ("doppler_ultrasound", "Doppler Ultrasound", "medical_us",
     ["doppler", "doppler ultrasound", "flow velocity", "color doppler", "pulsed doppler"], False),
    ("elastography", "Ultrasound Elastography", "medical_us",
     ["elastography", "shear wave", "strain imaging", "tissue stiffness us"], False),
    ("ivus", "Intravascular Ultrasound", "medical_us",
     ["ivus", "intravascular ultrasound", "coronary imaging", "catheter us"], False),
    ("ultrasonic_phased_array", "Ultrasonic Phased Array NDT", "ndt",
     ["phased array", "ultrasonic ndt", "total focusing method", "tfm", "fmc"], False),
    # Medical — Optical
    ("oct", "Optical Coherence Tomography", "medical_opt",
     ["oct", "optical coherence tomography", "fft oct", "full field oct",
      "a-scan", "b-scan oct", "retinal oct", "angiography oct"], True),
    ("octa", "OCT Angiography", "medical_opt",
     ["octa", "oct angiography", "speckle variance", "doppler oct"], False),
    ("fundus", "Fundus Photography", "medical_opt",
     ["fundus", "retinal photography", "optic disc", "diabetic retinopathy"], False),
    ("endoscopy", "Endoscopy", "medical_opt",
     ["endoscopy", "endoscope", "colonoscopy", "bronchoscopy", "capsule endoscopy"], False),
    ("confocal_endomicroscopy", "Confocal Endomicroscopy", "medical_opt",
     ["confocal endomicroscopy", "pce", "probe-based confocal", "endo microscopy"], False),
    # Medical — X-ray
    ("mammography", "Mammography", "medical_xray",
     ["mammography", "breast xray", "digital mammography", "breast screening"], False),
    ("fluoroscopy", "Fluoroscopy", "medical_xray",
     ["fluoroscopy", "real-time xray", "fluoroscopic", "interventional xray"], False),
    ("digital_breast_tomo", "Digital Breast Tomosynthesis", "medical_xray",
     ["breast tomosynthesis", "dbt", "digital breast", "3d mammography"], False),
    ("xray_radiography", "X-ray Radiography", "medical_xray",
     ["xray radiography", "x-ray", "chest xray", "plain film", "digital radiography"], False),
    ("portal_imaging", "Megavoltage Portal Imaging", "medical_rt",
     ["portal imaging", "megavoltage", "epid", "treatment verification xray"], False),
    ("brachytherapy_img", "Brachytherapy Imaging", "medical_rt",
     ["brachytherapy", "seed implant imaging", "hdr brachytherapy"], False),
    ("proton_therapy_img", "Proton Therapy Imaging", "medical_rt",
     ["proton therapy imaging", "proton ct", "range verification", "pbct"], False),
    ("proton_radiography", "Proton Radiography", "medical_rt",
     ["proton radiography", "proton beam imaging", "hadron imaging"], False),
    ("angiography", "X-ray Angiography / DSA", "medical_xray",
     ["angiography", "dsa", "digital subtraction", "coronary angiography", "catheter xray"], False),
    ("dexa", "Dual-Energy X-ray Absorptiometry", "medical_xray",
     ["dexa", "dxa", "bone density", "absorptiometry", "osteoporosis imaging"], False),
    # Medical — Other
    ("photoacoustic", "Photoacoustic Imaging", "medical_opt",
     ["photoacoustic", "optoacoustic", "pai", "msot", "photoacoustic tomography",
      "laser ultrasound"], False),
    ("magnetic_particle", "Magnetic Particle Imaging", "medical_other",
     ["magnetic particle imaging", "mpi", "superparamagnetic", "tracer imaging"], False),
    ("bioluminescence_tomo", "Bioluminescence Tomography", "medical_opt",
     ["bioluminescence", "blt", "luciferin", "in vivo optical imaging"], False),
    ("ceus", "Contrast Enhanced Ultrasound", "medical_us",
     ["ceus", "contrast enhanced ultrasound", "microbubble", "perfusion ultrasound"], False),
    ("clem", "Correlative Light-Electron Microscopy", "microscopy",
     ["clem", "correlative", "light electron microscopy", "colocalization em"], False),
    # Electron Microscopy
    ("tem", "Transmission Electron Microscopy", "microscopy_em",
     ["tem", "transmission electron microscopy", "electron microscopy",
      "high resolution tem", "hrtem", "ctem"], True),
    ("stem", "Scanning TEM", "microscopy_em",
     ["stem", "scanning tem", "haadf", "annular dark field", "eels stem"], True),
    ("sem", "Scanning Electron Microscopy", "microscopy_em",
     ["sem", "scanning electron", "secondary electron", "backscattered electron"], False),
    ("cryo_em", "Cryo-Electron Microscopy", "microscopy_em",
     ["cryo em", "cryo-em", "cryoem", "single particle", "cryo electron",
      "relion", "ctf correction"], True),
    ("cryo_et", "Cryo-Electron Tomography", "microscopy_em",
     ["cryo et", "cryo-et", "cryoet", "subtomogram", "cryo tomography"], False),
    ("fib_sem", "FIB-SEM", "microscopy_em",
     ["fib sem", "fib-sem", "focused ion beam", "volume electron microscopy",
      "connectomics"], False),
    ("electron_tomography", "Electron Tomography", "microscopy_em",
     ["electron tomography", "et", "tilt series", "missing wedge electron"], False),
    ("electron_diffraction", "Electron Diffraction", "microscopy_em",
     ["electron diffraction", "ed", "microed", "serial crystallography electron"], False),
    ("electron_holography", "Electron Holography", "microscopy_em",
     ["electron holography", "lorentz microscopy", "phase shift electron"], False),
    ("ebsd", "EBSD", "microscopy_em",
     ["ebsd", "electron backscatter diffraction", "crystallography sem",
      "grain orientation", "kikuchi"], False),
    ("eels", "EELS", "microscopy_em",
     ["eels", "electron energy loss", "core-loss", "plasmon eels"], False),
    ("edx_mapping", "EDX Mapping", "microscopy_em",
     ["edx", "edx mapping", "eds", "x-ray microanalysis", "elemental mapping sem"], False),
    ("eht_imaging", "Extreme High-Temperature Imaging", "microscopy_em",
     ["eht", "in situ heating", "high temperature em", "operando em"], False),
    # Optical Microscopy
    ("widefield", "Widefield Fluorescence", "microscopy_opt",
     ["widefield", "wide field", "epifluorescence", "fluorescence microscopy",
      "deconvolution microscopy"], True),
    ("widefield_lowdose", "Low-Dose Widefield", "microscopy_opt",
     ["low dose widefield", "photon limited fluorescence", "low photon microscopy"], False),
    ("confocal_3d", "Confocal 3D Microscopy", "microscopy_opt",
     ["confocal 3d", "confocal microscopy", "laser scanning confocal", "lscm",
      "optical sectioning"], False),
    ("confocal_livecell", "Live-Cell Confocal", "microscopy_opt",
     ["live cell confocal", "time-lapse confocal", "live cell imaging"], False),
    ("two_photon", "Two-Photon Microscopy", "microscopy_opt",
     ["two photon", "2p microscopy", "multiphoton", "tpef", "2-photon"], False),
    ("three_photon", "Three-Photon Microscopy", "microscopy_opt",
     ["three photon", "3p microscopy", "3-photon", "deep brain imaging"], False),
    ("lattice_lightsheet", "Lattice Light Sheet", "microscopy_opt",
     ["lattice light sheet", "llsm", "light sheet microscopy", "spim", "disr"], False),
    ("lightsheet", "Light Sheet Microscopy", "microscopy_opt",
     ["light sheet", "selective plane", "spim", "dslm", "mesolightsheet"], False),
    ("tirf", "TIRF Microscopy", "microscopy_opt",
     ["tirf", "total internal reflection", "evanescent wave", "single molecule tirf"], False),
    ("spinning_disk", "Spinning Disk Confocal", "microscopy_opt",
     ["spinning disk", "nipkow disk", "yokogawa", "fast confocal"], False),
    # Super-resolution
    ("sted", "STED Microscopy", "super_res",
     ["sted", "stimulated emission depletion", "super resolution sted",
      "nanoscale fluorescence"], False),
    ("palm_storm", "PALM/STORM Microscopy", "super_res",
     ["palm", "storm", "smlm", "single molecule localization", "stochastic optical",
      "dstorm", "fpalm"], False),
    ("sim", "Structured Illumination Microscopy", "super_res",
     ["sim", "structured illumination", "sim reconstruction", "wiener sim",
      "super resolution sim"], True),
    ("minflux", "MINFLUX Nanoscopy", "super_res",
     ["minflux", "minimum photon flux", "nanoscopy", "1nm resolution"], False),
    ("dna_paint", "DNA-PAINT", "super_res",
     ["dna paint", "point accumulation", "dna based super resolution"], False),
    ("expansion", "Expansion Microscopy", "super_res",
     ["expansion microscopy", "exm", "expansion", "hydrogel expansion"], False),
    # Spectroscopy
    ("raman_imaging", "Raman Imaging", "spectroscopy",
     ["raman", "raman imaging", "raman spectroscopy", "spontaneous raman",
      "raman mapping"], True),
    ("cars", "CARS Microscopy", "spectroscopy",
     ["cars", "coherent anti-stokes raman", "coherent raman"], False),
    ("srs", "SRS Microscopy", "spectroscopy",
     ["srs", "stimulated raman scattering", "hyperspectral srs"], False),
    ("ftir_imaging", "FTIR Imaging", "spectroscopy",
     ["ftir", "fourier transform infrared", "infrared spectroscopy imaging",
      "mid-infrared", "ir spectroscopy"], False),
    ("flim", "FLIM", "spectroscopy",
     ["flim", "fluorescence lifetime", "time-correlated", "tcspc", "phasor flim"], True),
    ("cathodoluminescence", "Cathodoluminescence", "spectroscopy",
     ["cathodoluminescence", "cl microscopy", "electron beam luminescence"], False),
    ("brillouin", "Brillouin Microscopy", "spectroscopy",
     ["brillouin", "brillouin scattering", "mechanical spectroscopy", "viscoelastic"], False),
    ("maldi_msi", "MALDI Mass Spec Imaging", "spectroscopy",
     ["maldi", "maldi msi", "mass spectrometry imaging", "maldi imaging",
      "spatial metabolomics"], False),
    ("desi", "DESI Mass Spec Imaging", "spectroscopy",
     ["desi", "desorption electrospray", "ambient mass spec", "desi imaging"], False),
    ("libs", "LIBS", "spectroscopy",
     ["libs", "laser induced breakdown", "laser ablation spectroscopy"], False),
    ("sims", "SIMS", "spectroscopy",
     ["sims", "secondary ion mass", "nano sims", "isotope imaging sims"], False),
    ("afm", "Atomic Force Microscopy", "spm",
     ["afm", "atomic force", "scanning force", "afm imaging", "nanomechanics"], False),
    ("stm", "Scanning Tunneling Microscopy", "spm",
     ["stm", "scanning tunneling", "tunneling microscopy", "atomic resolution"], False),
    ("mfm", "Magnetic Force Microscopy", "spm",
     ["mfm", "magnetic force", "magnetic domain imaging afm"], False),
    ("nsom", "Near-Field Optical Microscopy", "spm",
     ["nsom", "snom", "near field optical", "sub-diffraction optical"], False),
    # X-ray / Scattering
    ("saxs", "SAXS", "xray_scattering",
     ["saxs", "small angle x-ray", "small angle scattering", "nanostructure xray"], False),
    ("waxs", "WAXS", "xray_scattering",
     ["waxs", "wide angle x-ray", "powder diffraction", "crystallography xray"], False),
    ("xrf_imaging", "XRF Imaging", "xray_scattering",
     ["xrf", "x-ray fluorescence", "elemental mapping xray", "synchrotron xrf"], False),
    ("xrf_tomo", "XRF Tomography", "xray_scattering",
     ["xrf tomo", "fluorescence tomography", "3d xrf", "synchrotron tomo xrf"], False),
    ("xfel_sfx", "XFEL Serial Femtosecond Crystallography", "xray_scattering",
     ["xfel", "sfx", "serial femtosecond", "free electron laser", "xfel crystallography"], False),
    ("xray_crystallography", "X-ray Crystallography", "xray_scattering",
     ["xray crystallography", "protein crystallography", "diffraction crystallography",
      "phase problem", "direct methods"], False),
    ("dark_field", "Dark-Field Imaging", "xray_scattering",
     ["dark field", "dark-field xray", "grating interferometry", "scattering contrast"], False),
    ("phase_contrast", "Phase Contrast Imaging", "xray_scattering",
     ["phase contrast", "propagation phase", "grating phase", "talbot"], False),
    ("talbot_lau", "Talbot-Lau Interferometry", "xray_scattering",
     ["talbot lau", "grating interferometry", "differential phase", "dark field grating"], False),
    ("industrial_ct", "Industrial CT (NDT)", "ndt",
     ["industrial ct", "ndt ct", "computed tomography ndt", "part inspection ct",
      "additive manufacturing ct"], False),
    ("xray_ndt", "X-ray NDT", "ndt",
     ["xray ndt", "radiographic testing", "rt ndt", "digital radiography ndt"], False),
    # Computational Imaging
    ("cassi", "Coded-Aperture Spectral Imaging", "comp_imaging",
     ["cassi", "coded aperture spectral", "snapshot spectral", "hyperspectral cassi",
      "gap tv", "tsa", "spectral snapshot"], True),
    ("cacti", "Compressed Video (CACTI)", "comp_imaging",
     ["cacti", "compressed video", "temporal compressive", "efficient sci",
      "video snapshot compressive", "sci"], True),
    ("spc", "Single Pixel Camera", "comp_imaging",
     ["spc", "single pixel", "hadamard", "random mask acquisition", "compressive sensing camera"], False),
    ("coded_exposure", "Coded Exposure Photography", "comp_imaging",
     ["coded exposure", "flutter shutter", "motion deblur coded", "temporal coding"], False),
    ("ghost_imaging", "Ghost Imaging", "comp_imaging",
     ["ghost imaging", "computational ghost", "quantum ghost", "intensity correlation",
      "speckle correlation"], True),
    ("lensless", "Lensless Imaging", "comp_imaging",
     ["lensless", "diffuser camera", "flat camera", "computational camera",
      "psf calibration lensless", "admm lensless"], True),
    ("phase_retrieval", "Phase Retrieval", "comp_imaging",
     ["phase retrieval", "hio", "er algorithm", "gerchberg saxton", "phase problem",
      "coherent diffraction imaging", "oversampling"], True),
    ("ptychography", "Ptychography", "comp_imaging",
     ["ptychography", "pie", "epie", "dm ptycho", "scanning coherent diffraction",
      "position diversity"], False),
    ("holography", "Digital Holography", "comp_imaging",
     ["holography", "digital holography", "hologram reconstruction", "fresnel hologram",
      "off-axis holography"], False),
    ("fpm", "Fourier Ptychographic Microscopy", "comp_imaging",
     ["fpm", "fourier ptychographic", "led array microscopy", "synthetic aperture microscopy"], True),
    ("light_field", "Light Field Imaging", "comp_imaging",
     ["light field", "plenoptic", "lytro", "refocusing", "depth from light field"], False),
    ("integral", "Integral Imaging", "comp_imaging",
     ["integral imaging", "3d display", "elemental image", "micro lens array 3d"], False),
    ("cup", "Compressed Ultrafast Photography", "comp_imaging",
     ["cup", "compressed ultrafast", "streak camera comp", "femtosecond video"], False),
    ("streak_camera", "Streak Camera", "comp_imaging",
     ["streak camera", "ultrafast temporal", "time resolved imaging streak"], False),
    ("event_camera", "Event Camera", "comp_imaging",
     ["event camera", "dynamic vision sensor", "dvs", "neuromorphic camera",
      "asynchronous events"], False),
    ("hdr_imaging", "HDR Imaging", "comp_imaging",
     ["hdr", "high dynamic range", "exposure fusion", "tonemapping"], False),
    ("photometric_stereo", "Photometric Stereo", "comp_imaging",
     ["photometric stereo", "surface normal estimation", "shape from shading",
      "reflectance map"], False),
    ("panorama", "Panoramic Imaging", "comp_imaging",
     ["panorama", "image stitching", "panoramic reconstruction", "360 imaging"], False),
    ("structured_light", "Structured Light 3D", "comp_imaging",
     ["structured light", "fringe projection", "3d scanning structured",
      "phase shifting profilometry", "moiré"], False),
    ("tof_camera", "Time-of-Flight Camera", "comp_imaging",
     ["tof", "time of flight camera", "depth camera", "rgbd", "indirect tof"], False),
    ("odt", "Optical Diffraction Tomography", "comp_imaging",
     ["odt", "optical diffraction tomography", "refractive index tomography",
      "digital holographic tomography"], False),
    # Phase / Wavefront
    ("adaptive_optics", "Adaptive Optics", "wavefront",
     ["adaptive optics", "ao", "wavefront sensing", "deformable mirror",
      "shack hartmann", "atmospheric correction"], False),
    ("dic", "DIC Microscopy", "microscopy_opt",
     ["dic", "differential interference contrast", "nomarski", "phase gradient"], False),
    ("shearography", "Shearography", "ndt",
     ["shearography", "speckle shearing", "optical shearing interferometry", "ndt shear"], False),
    # Remote Sensing
    ("sar", "SAR Imaging", "remote_sensing",
     ["sar", "synthetic aperture radar", "backscatter radar", "sar imaging",
      "radar imaging", "sentinel sar", "envisat"], True),
    ("insar", "InSAR", "remote_sensing",
     ["insar", "interferometric sar", "phase unwrapping", "deformation mapping radar",
      "dinsar", "sentinel insar"], False),
    ("polsar", "PolSAR", "remote_sensing",
     ["polsar", "polarimetric sar", "polarization radar", "pol decomposition"], False),
    ("hyperspectral_remote", "Hyperspectral Remote Sensing", "remote_sensing",
     ["hyperspectral", "hyperspectral remote sensing", "aviris", "hyperion",
      "hyperspectral unmixing", "endmember"], True),
    ("multispectral_sat", "Multispectral Satellite", "remote_sensing",
     ["multispectral", "landsat", "sentinel 2", "worldview", "pan sharpening",
      "multispectral fusion"], False),
    ("ocean_color", "Ocean Color Remote Sensing", "remote_sensing",
     ["ocean color", "modis ocean", "chlorophyll remote sensing", "water leaving radiance"], False),
    ("passive_microwave", "Passive Microwave Remote Sensing", "remote_sensing",
     ["passive microwave", "tb brightness temperature", "amsr", "smos"], False),
    ("weather_radar", "Weather Radar", "remote_sensing",
     ["weather radar", "doppler weather", "nexrad", "precipitation radar",
      "dual pol radar"], False),
    ("solar_imaging", "Solar Imaging", "astronomy",
     ["solar", "solar imaging", "aia", "hinode", "corona imaging",
      "solar telescope", "sunspot"], False),
    # LiDAR / Depth
    ("lidar", "LiDAR", "lidar",
     ["lidar", "laser ranging", "point cloud", "3d lidar", "autonomous driving lidar"], False),
    ("flash_lidar", "Flash LiDAR", "lidar",
     ["flash lidar", "3d flash", "single shot depth", "tof lidar"], False),
    ("sonar", "Sonar Imaging", "acoustic",
     ["sonar", "side scan sonar", "multibeam sonar", "bathymetry", "underwater imaging"], False),
    ("ocean_acoustic_tomo", "Ocean Acoustic Tomography", "acoustic",
     ["ocean acoustic", "acoustic tomography", "ocean sound speed", "hydrophone array"], False),
    # Geophysics
    ("seismic_tomo", "Seismic Tomography", "geophysics",
     ["seismic tomo", "seismic tomography", "travel time inversion", "mantle tomography",
      "earthquake tomography"], False),
    ("fwi", "Full Waveform Inversion", "geophysics",
     ["fwi", "full waveform inversion", "seismic inversion", "acoustic wave inversion"], False),
    ("gpr", "Ground Penetrating Radar", "geophysics",
     ["gpr", "ground penetrating radar", "subsurface radar", "georadar",
      "hyperbola gpr"], False),
    ("impedance_tomo", "Electrical Impedance Tomography", "geophysics",
     ["eit", "impedance tomography", "electrical impedance", "conductivity imaging"], False),
    ("muon_tomo", "Muon Tomography", "physics",
     ["muon", "muon tomography", "cosmic ray imaging", "muography"], False),
    ("neutron_tomo", "Neutron Tomography", "physics",
     ["neutron", "neutron tomography", "neutron ct", "reactor neutron imaging"], False),
    # Astronomy
    ("radio_astronomy", "Radio Astronomy Imaging", "astronomy",
     ["radio astronomy", "clean algorithm", "dirty map", "radio telescope",
      "aperture synthesis"], False),
    ("radio_interferometry", "Radio Interferometry", "astronomy",
     ["radio interferometry", "vlbi", "uv plane", "baseline coverage", "imaging vlbi"], False),
    ("eht_imaging", "Event Horizon Telescope", "astronomy",
     ["eht", "event horizon telescope", "black hole imaging", "m87 imaging"], False),
    ("coronagraphy", "Coronagraphy", "astronomy",
     ["coronagraph", "exoplanet imaging", "star coronagraphy", "hci", "direct imaging"], False),
    ("lucky_imaging", "Lucky Imaging", "astronomy",
     ["lucky imaging", "atmospheric speckle", "short exposure astronomy",
      "speckle imaging"], False),
    # Other Physics
    ("gravitational_wave", "Gravitational Wave Imaging", "physics",
     ["gravitational wave", "ligo", "lisa", "gw detector", "matched filter gw"], False),
    ("particle_calorimetry", "Particle Calorimetry Imaging", "physics",
     ["calorimetry", "particle detector", "em calorimeter", "pflow", "shower imaging"], False),
    ("quantum_illumination", "Quantum Illumination", "physics",
     ["quantum illumination", "entangled imaging", "quantum radar",
      "quantum sensing", "squeezed light"], False),
    ("entangled_photon", "Entangled Photon Imaging", "physics",
     ["entangled photon", "ghost imaging quantum", "coincidence imaging",
      "two photon entanglement"], False),
    ("pump_probe", "Pump-Probe Spectroscopy", "physics",
     ["pump probe", "transient absorption", "ultrafast spectroscopy", "time resolved pump"], False),
    ("shg", "Second Harmonic Generation Imaging", "physics",
     ["shg", "second harmonic generation", "nonlinear optical imaging",
      "collagen shg", "chi2 imaging"], False),
    # NDT / Industrial
    ("active_thermography", "Active Thermography", "ndt",
     ["thermography", "thermal ndt", "infrared thermography", "lock-in thermography",
      "pulsed thermography"], False),
    ("eddy_current", "Eddy Current Imaging", "ndt",
     ["eddy current", "electromagnetic ndt", "eddy current testing", "ec imaging"], False),
    # Machine Vision / Other
    ("machine_vision", "Machine Vision / Microscopy", "machine_vision",
     ["machine vision", "industrial inspection", "defect detection",
      "barcode", "industrial imaging"], False),
    ("atom_probe", "Atom Probe Tomography", "spectroscopy",
     ["atom probe", "apt", "atom probe tomography", "field ion microscopy"], False),
    ("acoustic_microscopy", "Acoustic Microscopy", "acoustic",
     ["acoustic microscopy", "scanning acoustic", "saw imaging", "sam microscopy"], False),
    ("acoustic_emission", "Acoustic Emission", "ndt",
     ["acoustic emission", "ae sensor", "stress wave", "ae testing ndt"], False),
    ("matrix", "Matrix Imaging (Ultrasound)", "acoustic",
     ["matrix imaging", "distortion matrix", "ultrasound matrix", "random medium"],False),
    # NeRF / Novel View
    ("nerf", "Neural Radiance Fields (NeRF)", "novel_view",
     ["nerf", "neural radiance", "novel view synthesis", "3d reconstruction nerf",
      "instant ngp", "mip nerf", "volume rendering neural"], False),
    ("gaussian_splatting", "Gaussian Splatting", "novel_view",
     ["gaussian splatting", "3dgs", "3d gaussian", "real-time novel view",
      "splat rendering"], False),
    # Optical Other
    ("polarization", "Polarization Imaging", "optical",
     ["polarization", "stokes parameters", "mueller matrix", "polarimetry"], False),
    ("terahertz", "Terahertz Imaging", "optical",
     ["terahertz", "thz imaging", "thz-tds", "terahertz spectroscopy"], False),
    ("dot", "Diffuse Optical Tomography", "medical_opt",
     ["dot", "diffuse optical", "nirs tomography", "near infrared diffuse", "doe"], False),
    ("ct_fluorescence", "CT-Fluorescence Fusion", "medical_opt",
     ["ct fluorescence", "hybrid ct optical", "fluorescence ct"], False),
    ("ism", "Image Scanning Microscopy", "microscopy_opt",
     ["ism", "image scanning", "rescan microscopy", "pixel reassignment"], False),
    ("mr_elastography", "MR Elastography", "medical_mri",
     ["mr elastography", "mre", "elasticity mri"], False),
]

# ---------------------------------------------------------------------------
# Use case keyword database
# ---------------------------------------------------------------------------
USE_CASE_DB = {
    "reconstruct": {
        "keywords": ["reconstruct", "reconstruction", "recover", "recover image",
                     "solve", "invert", "inverse problem", "deconvol", "algorithm",
                     "fbp", "tv admm", "compressed sensing", "deep learning recon",
                     "psnr", "ssim", "image quality"],
        "spec_dir": "01_reconstruct",
        "description": "Reconstruct measurement data using specific algorithms",
    },
    "mismatch": {
        "keywords": ["mismatch", "calibrat", "correct", "misalign", "wrong operator",
                     "imperfect operator", "phase error", "position error", "psf error",
                     "center of rotation", "coil mismatch", "scatter", "beam hardening",
                     "pixel shift", "rotation error", "cor", "dispersion error"],
        "spec_dir": "02_mismatch_reconstruct",
        "description": "Correct operator mismatch then reconstruct",
    },
    "system_design": {
        "keywords": ["design", "system design", "forward model", "simulate system",
                     "imaging system", "dag", "pipeline", "noise model", "detector",
                     "sensor design", "optimize system", "simulate acquisition"],
        "spec_dir": "03_system_design",
        "description": "Design and simulate an imaging system with mismatch",
    },
    "simulation": {
        "keywords": ["simulat", "physics simulation", "forward simulation",
                     "generate data", "phantom", "synthetic data", "numerical simulation",
                     "analytical", "ground truth generate"],
        "spec_dir": "04_simulation",
        "description": "Physics simulation examples",
    },
}

# ---------------------------------------------------------------------------
# Spec files that exist on disk
# ---------------------------------------------------------------------------
SPEC_FILES = {
    "reconstruct": {
        "ct": "01_reconstruct/ct.md",
        "mri": "01_reconstruct/mri.md",
        "cbct": "01_reconstruct/cbct.md",
        "pet": "01_reconstruct/pet.md",
        "spect": "01_reconstruct/spect.md",
        "ultrasound": "01_reconstruct/ultrasound.md",
        "oct": "01_reconstruct/oct.md",
        "cassi": "01_reconstruct/cassi.md",
        "cacti": "01_reconstruct/cacti.md",
        "lensless": "01_reconstruct/lensless.md",
        "phase_retrieval": "01_reconstruct/phase_retrieval.md",
        "sim": "01_reconstruct/sim.md",
    },
    "mismatch": {
        "ct": "02_mismatch_reconstruct/ct_mismatch.md",
        "mri": "02_mismatch_reconstruct/mri_mismatch.md",
        "cassi": "02_mismatch_reconstruct/cassi_mismatch.md",
        "lensless": "02_mismatch_reconstruct/lensless_mismatch.md",
        "widefield": "02_mismatch_reconstruct/microscopy_mismatch.md",
        "widefield_lowdose": "02_mismatch_reconstruct/microscopy_mismatch.md",
        "confocal_3d": "02_mismatch_reconstruct/microscopy_mismatch.md",
    },
    "system_design": {
        "ct": "03_system_design/ct_system.md",
        "mri": "03_system_design/mri_system.md",
        "lensless": "03_system_design/lensless_system.md",
        "sim": "03_system_design/sim_system.md",
    },
    "simulation": {
        "ct": "04_simulation/ct_simulation.md",
        "mri": "04_simulation/mri_simulation.md",
        "optics": "04_simulation/optics_simulation.md",
        "wave": "04_simulation/wave_simulation.md",
    },
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase and remove punctuation."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def score_modality(query_norm: str, keywords: List[str]) -> float:
    """Keyword overlap score with word-boundary protection for short keywords."""
    score = 0.0
    query_words = set(query_norm.split())
    for kw in keywords:
        kw_norm = normalize(kw)
        kw_words = kw_norm.split()
        if len(kw_words) == 1:
            # Single short keyword: require exact whole-word match to prevent
            # "tem" matching "system", "sim" matching "simulate", etc.
            if kw_norm in query_words:
                score += 1
        else:
            # Multi-word phrase: substring match in full query (safe, more specific)
            if kw_norm in query_norm:
                score += len(kw_words)
    return score


def score_use_case(query_norm: str, uc_keywords: List[str]) -> float:
    score = 0.0
    for kw in uc_keywords:
        if normalize(kw) in query_norm:
            score += 1.0
    return score


def match(query: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Returns (modality_id, use_case_key, confidence_score).
    """
    query_norm = normalize(query)

    # 1. Score modalities
    mod_scores: List[Tuple[float, str, str]] = []
    for mod_id, display, cat, keywords, _ in MODALITY_DB:
        s = score_modality(query_norm, keywords)
        if s > 0:
            mod_scores.append((s, mod_id, display))
    mod_scores.sort(reverse=True)

    best_mod = mod_scores[0][1] if mod_scores else None
    mod_conf = mod_scores[0][0] if mod_scores else 0.0

    # 2. Score use cases
    uc_scores: List[Tuple[float, str]] = []
    for uc_key, uc_info in USE_CASE_DB.items():
        s = score_use_case(query_norm, uc_info["keywords"])
        uc_scores.append((s, uc_key))
    uc_scores.sort(reverse=True)

    best_uc = uc_scores[0][1] if uc_scores[0][0] > 0 else "reconstruct"

    return best_mod, best_uc, mod_conf


def get_spec_path(modality: str, use_case: str) -> Optional[Path]:
    """Find spec file path. Falls back to template."""
    uc_files = SPEC_FILES.get(use_case, {})
    rel = uc_files.get(modality)
    if rel:
        p = SPEC_DIR / rel
        if p.exists():
            return p

    # Fall back to _template.md
    template_dir_map = {
        "reconstruct": "01_reconstruct",
        "mismatch": "02_mismatch_reconstruct",
        "system_design": "03_system_design",
        "simulation": "04_simulation",
    }
    template = SPEC_DIR / template_dir_map.get(use_case, "01_reconstruct") / "_template.md"
    if template.exists():
        return template
    return None


def generate_minimal_spec(modality: str, use_case: str) -> str:
    """Generate a minimal spec for modalities without a dedicated file."""
    # Find display name
    display = modality
    for mod_id, disp, cat, kws, _ in MODALITY_DB:
        if mod_id == modality:
            display = disp
            break

    return f"""# {display} — Auto-generated Spec

> **Generated by keyword matching** (no dedicated spec file for `{modality}`)
> For a custom spec, provide a Gemini API key: `--api-key YOUR_KEY`

## Modality
- **ID**: `{modality}`
- **Display Name**: {display}

## Quick Run (Use Case: {use_case.replace('_', ' ').title()})

```python
import sys, os
# Local: adjust path to your clone
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
# Colab: BASE = '/content/Physics_World_Model/pwm/public'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

from algorithm_base.{modality}.solvers import run_solver, list_solvers
import numpy as np

# List available solvers
print("Available solvers for {modality}:")
for key, info in list_solvers():
    gpu_tag = "[GPU]" if info.get("gpu") else "[CPU]"
    print(f"  {{key}}: {{info['name']}} {{gpu_tag}} — {{info.get('reference', '')}}")

# Run with sample data (replace with your measurement)
y = np.random.rand(64, 64).astype(np.float32)  # replace with real y
x_hat = run_solver('traditional_cpu', y, operator=None, cfg={{}})
print(f"Output shape: {{x_hat.shape}}, dtype: {{x_hat.dtype}}")
```

## Full Spec
See `spec/01_reconstruct/_template.md` for the complete spec template.
"""


def run_reconstruction(modality: str) -> None:
    """Actually run a solver on sample data."""
    sys.path.insert(0, str(PUBLIC_DIR))
    sys.path.insert(0, str(PUBLIC_DIR / "packages" / "pwm_core"))
    try:
        import importlib
        import numpy as np
        m = importlib.import_module(f"algorithm_base.{modality}.solvers")
        solvers = m.list_solvers()
        if not solvers:
            print(f"[ERROR] No solvers found for {modality}")
            return
        key = solvers[0][0]
        print(f"Running solver: {key} ({solvers[0][1]['name']})")
        y = np.random.rand(64, 64).astype(np.float32)
        x_hat = m.run_solver(key, y, None, {})
        import numpy as np_
        print(f"Output shape: {x_hat.shape}, dtype: {x_hat.dtype}")
        print(f"Value range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    except Exception as e:
        print(f"[ERROR] {e}")


def list_all_modalities() -> None:
    """Print a table of all modalities."""
    print(f"\n{'ID':<30} {'Display Name':<45} {'Category':<20} {'Spec'}")
    print("-" * 110)
    for mod_id, display, cat, _, has_spec in sorted(MODALITY_DB, key=lambda x: x[2] + x[0]):
        spec_mark = "Full spec" if has_spec else "Template"
        print(f"{mod_id:<30} {display:<45} {cat:<20} {spec_mark}")
    print(f"\nTotal: {len(MODALITY_DB)} modalities")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PWM Spec.md Keyword Matcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument("--run", action="store_true",
                        help="Run the first solver on sample data after matching")
    parser.add_argument("--list", action="store_true",
                        help="List all available modalities")
    parser.add_argument("--api-key", default=None,
                        help="Gemini 2.5 Flash API key (reserved for LLM finetuning)")
    parser.add_argument("--gateway-key", default=None,
                        help="comparegpt.io gateway key (reserved for LLM finetuning)")
    args = parser.parse_args()

    if args.list:
        list_all_modalities()
        return

    if not args.query:
        parser.print_help()
        return

    # LLM key check
    if args.api_key or args.gateway_key:
        print("[INFO] LLM-guided spec finetuning is not yet implemented.")
        print("[INFO] Falling back to keyword matching.\n")

    # Always announce keyword matching mode
    print("=" * 60)
    print("PWM Spec Matcher — Keyword Match Mode")
    print("(No LLM API key used. Results are based on keyword matching.)")
    print("=" * 60)

    modality, use_case, confidence = match(args.query)

    if not modality:
        print(f"\n[No match found for: '{args.query}']")
        print("Try one of:")
        list_all_modalities()
        return

    # Find display name
    display = modality
    for mod_id, disp, cat, _, _ in MODALITY_DB:
        if mod_id == modality:
            display = disp
            break

    uc_info = USE_CASE_DB.get(use_case, {})
    print(f"\nBest match:")
    print(f"  Modality  : {display} ({modality})")
    print(f"  Use Case  : {uc_info.get('description', use_case)}")
    print(f"  Confidence: {confidence:.1f}\n")

    # Load spec file
    spec_path = get_spec_path(modality, use_case)
    if spec_path and spec_path.exists():
        print(f"Spec file: {spec_path.relative_to(SPEC_DIR)}\n")
        print("-" * 60)
        print(spec_path.read_text())
    else:
        print("[No dedicated spec file found. Generating minimal spec...]\n")
        print(generate_minimal_spec(modality, use_case))

    # Optionally run
    if args.run:
        print("\n" + "=" * 60)
        print("Running solver (sample data)...")
        run_reconstruction(modality)


if __name__ == "__main__":
    main()
