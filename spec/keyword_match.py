#!/usr/bin/env python3
"""PWM keyword matcher — no LLM required.

Usage:
    python3 spec/keyword_match.py "CT reconstruction"
    python3 spec/keyword_match.py "MRI mismatch"
    python3 spec/keyword_match.py list
"""
from __future__ import annotations
import re, sys
from pathlib import Path

SPEC_DIR = Path(__file__).parent

# ── Modality keyword database (modality_id, display_name, keywords) ──────
MODALITIES = [
    ('acoustic_emission', 'Acoustic Emission', ['acoustic emission', 'ae testing', 'stress wave']),
    ('acoustic_microscopy', 'Acoustic Microscopy', ['acoustic microscopy', 'sam', 'ultrasound microscopy']),
    ('active_thermography', 'Active Thermography', ['thermography', 'infrared ndt', 'thermal imaging ndt']),
    ('adaptive_optics', 'Adaptive Optics', ['adaptive optics', 'wavefront', 'deformable mirror']),
    ('afm', 'Atomic Force Microscopy (AFM)', ['afm', 'atomic force']),
    ('angiography', 'Angiography', ['angiography', 'vessel imaging', 'dsa']),
    ('asl_mri', 'ASL MRI', ['asl', 'arterial spin labeling', 'perfusion mri']),
    ('atom_probe', 'Atom Probe Tomography', ['atom probe', 'field ion', 'apt']),
    ('bioluminescence_tomo', 'Bioluminescence Tomography', ['bioluminescence', 'blt', 'optical tomography']),
    ('brachytherapy_img', 'Brachytherapy Imaging', ['brachytherapy', 'seed implant imaging']),
    ('brillouin', 'Brillouin Microscopy', ['brillouin', 'stimulated brillouin', 'mechanical spectroscopy']),
    ('cacti', 'CACTI (Compressed Video)', ['cacti', 'compressed video', 'high speed video snapshot']),
    ('cars', 'CARS Microscopy', ['cars', 'coherent anti-stokes raman']),
    ('cassi', 'CASSI (Coded Aperture)', ['cassi', 'coded aperture', 'spectral snapshot', 'hyperspectral snapshot']),
    ('cathodoluminescence', 'Cathodoluminescence', ['cathodoluminescence', 'cl imaging']),
    ('cbct', 'Cone-Beam CT (CBCT)', ['cbct', 'cone beam ct', 'dental ct', 'cone-beam']),
    ('cest_mri', 'CEST MRI', ['cest', 'chemical exchange saturation transfer']),
    ('ceus', 'Contrast-Enhanced Ultrasound', ['ceus', 'contrast enhanced ultrasound', 'microbubble']),
    ('clem', 'CLEM', ['clem', 'correlative light electron', 'correlative microscopy']),
    ('coded_exposure', 'Coded Exposure', ['coded exposure', 'flutter shutter', 'coded aperture video']),
    ('confocal_3d', 'Confocal 3D', ['confocal 3d', 'confocal z-stack', 'optical sectioning']),
    ('confocal_endomicroscopy', 'Confocal Endomicroscopy', ['endomicroscopy', 'confocal endoscopy', 'cle']),
    ('confocal_livecell', 'Confocal Live-Cell', ['confocal live cell', 'live cell imaging', 'time-lapse confocal']),
    ('coronagraphy', 'Coronagraphy', ['coronagraphy', 'coronagraph', 'stellar coronagraphy']),
    ('cryo_em', 'Cryo-EM', ['cryo-em', 'cryo em', 'single particle', 'electron microscopy cryo']),
    ('cryo_et', 'Cryo-ET', ['cryo-et', 'cryo electron tomography', 'subtomogram']),
    ('ct', 'X-ray CT', ['ct', 'computed tomography', 'x-ray ct', 'fan beam', 'parallel beam', 'sinogram']),
    ('ct_fluorescence', 'CT Fluorescence', ['ct fluorescence', 'xrf ct', 'x-ray fluorescence ct']),
    ('cup', 'Compressed Ultrafast Photography', ['cup', 'compressed ultrafast', 'streak camera compressed']),
    ('dark_field', 'Dark-Field Imaging', ['dark field', 'dark-field', 'grating interferometry']),
    ('desi', 'DESI-MS Imaging', ['desi', 'desorption electrospray', 'ambient mass spec imaging']),
    ('dexa', 'Dual-Energy X-ray (DEXA)', ['dexa', 'dual energy x-ray', 'bone density']),
    ('dic', 'Differential Interference Contrast', ['dic', 'differential interference contrast', 'nomarski']),
    ('diffusion_mri', 'Diffusion MRI', ['diffusion mri', 'dwi', 'dti', 'diffusion tensor', 'tractography']),
    ('digital_breast_tomo', 'Digital Breast Tomosynthesis', ['breast tomosynthesis', 'dbt', 'mammography tomo']),
    ('dna_paint', 'DNA-PAINT', ['dna-paint', 'dna paint', 'exchange-paint', 'points accumulation']),
    ('doppler_ultrasound', 'Doppler Ultrasound', ['doppler ultrasound', 'color doppler', 'pulsed doppler']),
    ('dot', 'Diffuse Optical Tomography', ['dot', 'diffuse optical', 'optical tomography near infrared']),
    ('ebsd', 'EBSD', ['ebsd', 'electron backscatter diffraction', 'crystallographic orientation']),
    ('eddy_current', 'Eddy Current Testing', ['eddy current', 'electromagnetic ndt']),
    ('edx_mapping', 'EDX Mapping', ['edx', 'eds mapping', 'energy dispersive x-ray mapping']),
    ('eels', 'EELS', ['eels', 'electron energy loss spectroscopy']),
    ('eht_imaging', 'Event Horizon Telescope', ['eht', 'event horizon telescope', 'black hole imaging']),
    ('elastography', 'Elastography', ['elastography', 'tissue stiffness', 'shear wave']),
    ('electron_diffraction', 'Electron Diffraction', ['electron diffraction', '4d-stem', 'ued']),
    ('electron_holography', 'Electron Holography', ['electron holography', 'off-axis holography electron']),
    ('electron_tomography', 'Electron Tomography', ['electron tomography', 'et reconstruction']),
    ('endoscopy', 'Endoscopy', ['endoscopy', 'endoscope', 'gastrointestinal imaging']),
    ('entangled_photon', 'Entangled Photon Imaging', ['entangled photon', 'quantum imaging', 'ghost imaging quantum']),
    ('event_camera', 'Event Camera', ['event camera', 'dynamic vision sensor', 'dvs']),
    ('expansion', 'Expansion Microscopy', ['expansion microscopy', 'exm', 'expansion']),
    ('fib_sem', 'FIB-SEM', ['fib-sem', 'fib sem', 'focused ion beam sem']),
    ('flash_lidar', 'Flash LiDAR', ['flash lidar', 'single photon lidar', 'tof lidar']),
    ('flim', 'FLIM', ['flim', 'fluorescence lifetime', 'time-correlated single photon']),
    ('fluoroscopy', 'Fluoroscopy', ['fluoroscopy', 'real-time x-ray', 'interventional imaging']),
    ('fmri', 'fMRI', ['fmri', 'functional mri', 'bold', 'brain activation']),
    ('fpm', 'Fourier Ptychographic Microscopy', ['fpm', 'fourier ptychography', 'led array microscopy']),
    ('ftir_imaging', 'FTIR Imaging', ['ftir', 'fourier transform infrared imaging', 'infrared spectroscopy imaging']),
    ('fundus', 'Fundus Imaging', ['fundus', 'retinal imaging', 'ophthalmoscopy']),
    ('fwi', 'Full Waveform Inversion', ['fwi', 'full waveform inversion', 'seismic inversion']),
    ('gaussian_splatting', 'Gaussian Splatting', ['gaussian splatting', '3d gaussian', 'nerf gaussian']),
    ('ghost_imaging', 'Ghost Imaging', ['ghost imaging', 'computational ghost', 'single pixel ghost']),
    ('gpr', 'Ground Penetrating Radar', ['gpr', 'ground penetrating radar', 'subsurface radar']),
    ('gravitational_wave', 'Gravitational Wave Detection', ['gravitational wave', 'ligo', 'gw detection']),
    ('hdr_imaging', 'HDR Imaging', ['hdr', 'high dynamic range', 'tone mapping']),
    ('holography', 'Digital Holography', ['holography', 'digital holography', 'hologram reconstruction']),
    ('hyperspectral_remote', 'Hyperspectral Remote Sensing', ['hyperspectral remote', 'hyperspectral satellite', 'aviris']),
    ('impedance_tomo', 'Electrical Impedance Tomography', ['eit', 'electrical impedance', 'impedance tomography']),
    ('industrial_ct', 'Industrial CT', ['industrial ct', 'ndt ct', 'nondestructive ct']),
    ('insar', 'InSAR', ['insar', 'interferometric sar', 'surface deformation radar']),
    ('integral', 'Integral Imaging', ['integral imaging', 'light field display', 'plenoptic camera']),
    ('ism', 'Image Scanning Microscopy', ['ism', 'image scanning microscopy', 'airyscan']),
    ('ivus', 'IVUS', ['ivus', 'intravascular ultrasound', 'coronary ultrasound']),
    ('lattice_lightsheet', 'Lattice Light-Sheet', ['lattice light sheet', 'lattice lightsheet', 'llsm']),
    ('lensless', 'Lensless Imaging', ['lensless', 'diffusercam', 'computational camera', 'flatcam']),
    ('libs', 'LIBS', ['libs', 'laser induced breakdown spectroscopy']),
    ('lidar', 'LiDAR', ['lidar', '3d lidar', 'point cloud lidar', 'spinning lidar']),
    ('light_field', 'Light Field Imaging', ['light field', 'plenoptic', 'lytro', 'light-field']),
    ('lightsheet', 'Light-Sheet Microscopy', ['light sheet', 'lightsheet', 'spim', 'dslm']),
    ('lucky_imaging', 'Lucky Imaging', ['lucky imaging', 'speckle imaging', 'atmospheric speckle']),
    ('machine_vision', 'Machine Vision', ['machine vision', 'industrial vision', 'defect detection']),
    ('magnetic_particle', 'Magnetic Particle Imaging', ['mpi', 'magnetic particle imaging']),
    ('maldi_msi', 'MALDI Mass Spec Imaging', ['maldi', 'maldi msi', 'matrix assisted laser mass spec']),
    ('mammography', 'Mammography', ['mammography', 'breast x-ray', 'breast screening']),
    ('matrix', 'Matrix Completion', ['matrix completion', 'low rank matrix', 'collaborative filtering']),
    ('mfm', 'Magnetic Force Microscopy', ['mfm', 'magnetic force microscopy']),
    ('minflux', 'MINFLUX', ['minflux', 'minimal photon flux', 'nanoscale localization']),
    ('mr_elastography', 'MR Elastography', ['mr elastography', 'mre', 'magnetic resonance elastography']),
    ('mr_fingerprinting', 'MR Fingerprinting', ['mr fingerprinting', 'mrf', 'rapid quantitative mri']),
    ('mra', 'MR Angiography', ['mra', 'mr angiography', 'tof mra', 'contrast enhanced mra']),
    ('mri', 'MRI', ['mri', 'magnetic resonance imaging', 'k-space', 'undersampled mri', 'parallel imaging']),
    ('mrs', 'MR Spectroscopy', ['mrs', 'mr spectroscopy', 'proton spectroscopy', 'metabolite']),
    ('multispectral_sat', 'Multispectral Satellite', ['multispectral satellite', 'sentinel', 'landsat', 'multispectral remote']),
    ('muon_tomo', 'Muon Tomography', ['muon tomography', 'muon radiography', 'cosmic ray imaging']),
    ('nerf', 'NeRF', ['nerf', 'neural radiance field', 'novel view synthesis']),
    ('neutron_diffraction', 'Neutron Diffraction', ['neutron diffraction', 'neutron crystallography']),
    ('neutron_tomo', 'Neutron Tomography', ['neutron tomography', 'neutron ct', 'neutron radiography']),
    ('nirs_brain', 'fNIRS', ['fnirs', 'functional nirs', 'near infrared spectroscopy brain']),
    ('nsom', 'Near-Field Optical Microscopy', ['nsom', 'near field optical', 'aperture nsom']),
    ('ocean_acoustic_tomo', 'Ocean Acoustic Tomography', ['ocean acoustic tomography', 'ocean acoustic']),
    ('ocean_color', 'Ocean Color Remote Sensing', ['ocean color', 'coastal remote sensing', 'chlorophyll mapping']),
    ('oct', 'Optical Coherence Tomography (OCT)', ['oct', 'optical coherence tomography', 'retinal oct', 'a-scan']),
    ('octa', 'OCT Angiography', ['octa', 'oct angiography', 'speckle variance oct']),
    ('odt', 'Optical Diffraction Tomography', ['odt', 'optical diffraction tomography', 'holographic tomography']),
    ('palm_storm', 'PALM/STORM', ['palm', 'storm', 'smlm', 'single molecule localization', 'super resolution localization']),
    ('panorama', 'Panoramic Imaging', ['panorama', 'panoramic', 'image stitching', '360 image']),
    ('particle_calorimetry', 'Particle Calorimetry', ['calorimetry', 'particle detector', 'hep imaging']),
    ('passive_microwave', 'Passive Microwave Remote Sensing', ['passive microwave', 'radiometry satellite', 'amsr']),
    ('pet', 'PET', ['pet', 'positron emission tomography', 'pet scan', 'fdg pet']),
    ('pet_ct', 'PET-CT', ['pet-ct', 'pet ct', 'combined pet ct']),
    ('pet_mr', 'PET-MR', ['pet-mr', 'pet mr', 'combined pet mri']),
    ('phase_contrast', 'Phase-Contrast Imaging', ['phase contrast', 'propagation based', 'in-line holography']),
    ('phase_retrieval', 'Phase Retrieval', ['phase retrieval', 'hio', 'raar', 'ptychography', 'coherent diffractive']),
    ('photoacoustic', 'Photoacoustic Imaging', ['photoacoustic', 'optoacoustic', 'pact', 'pai']),
    ('photometric_stereo', 'Photometric Stereo', ['photometric stereo', 'surface normal', 'shape from shading']),
    ('polarization', 'Polarization Imaging', ['polarization imaging', 'stokes', 'mueller matrix']),
    ('polsar', 'PolSAR', ['polsar', 'polarimetric sar', 'quad-pol sar']),
    ('portal_imaging', 'Portal Imaging', ['portal imaging', 'epid', 'megavoltage imaging']),
    ('proton_radiography', 'Proton Radiography', ['proton radiography', 'proton imaging']),
    ('proton_therapy_img', 'Proton Therapy Imaging', ['proton therapy imaging', 'proton therapy']),
    ('ptychography', 'Ptychography', ['ptychography', 'ptychographic', 'ptycho']),
    ('pump_probe', 'Pump-Probe Spectroscopy', ['pump probe', 'transient absorption', 'ultrafast spectroscopy']),
    ('quantum_illumination', 'Quantum Illumination', ['quantum illumination', 'quantum radar', 'quantum sensing']),
    ('radio_astronomy', 'Radio Astronomy', ['radio astronomy', 'radio telescope', 'vlbi']),
    ('radio_interferometry', 'Radio Interferometry', ['radio interferometry', 'aperture synthesis', 'uvw plane']),
    ('raman_imaging', 'Raman Imaging', ['raman imaging', 'raman spectroscopy imaging', 'confocal raman']),
    ('sar', 'SAR', ['sar', 'synthetic aperture radar', 'sar imaging', 'sar reconstruction']),
    ('saxs', 'SAXS', ['saxs', 'small angle x-ray scattering', 'saxs reconstruction']),
    ('seismic_tomo', 'Seismic Tomography', ['seismic tomography', 'seismic imaging', 'earthquake tomography']),
    ('sem', 'SEM', ['sem', 'scanning electron microscopy']),
    ('shearography', 'Shearography', ['shearography', 'speckle shear', 'laser shearography']),
    ('shg', 'SHG Microscopy', ['shg', 'second harmonic generation', 'shg microscopy']),
    ('sim', 'Structured Illumination Microscopy', ['sim', 'structured illumination', 'sir', 'super resolution fluorescence']),
    ('sims', 'SIMS Imaging', ['sims', 'secondary ion mass spectrometry', 'nanoSIMS']),
    ('solar_imaging', 'Solar Imaging', ['solar imaging', 'solar euv', 'aia solar', 'sun imaging']),
    ('sonar', 'SONAR', ['sonar', 'underwater acoustics', 'side-scan sonar']),
    ('spc', 'Single Pixel Camera', ['spc', 'single pixel camera', 'single pixel', 'compressive sensing camera']),
    ('spect', 'SPECT', ['spect', 'single photon emission ct']),
    ('spect_ct', 'SPECT-CT', ['spect-ct', 'spect ct', 'combined spect ct']),
    ('spectral_ct', 'Spectral CT', ['spectral ct', 'photon counting ct', 'energy resolving ct', 'dual energy ct']),
    ('spinning_disk', 'Spinning Disk Confocal', ['spinning disk', 'nipkow disk', 'spinning disk confocal']),
    ('srs', 'SRS Microscopy', ['srs', 'stimulated raman scattering', 'srs microscopy']),
    ('sted', 'STED Microscopy', ['sted', 'stimulated emission depletion', 'sted microscopy']),
    ('stem', 'STEM', ['stem', 'scanning transmission electron', 'haadf stem']),
    ('stm', 'STM', ['stm', 'scanning tunneling microscopy', 'tunneling current']),
    ('streak_camera', 'Streak Camera', ['streak camera', 'ultrafast imaging', 'temporal imaging']),
    ('structured_light', 'Structured Light 3D', ['structured light', 'fringe projection', 'phase shifting 3d']),
    ('swi', 'Susceptibility Weighted Imaging', ['swi', 'susceptibility weighted', 'venography mri']),
    ('talbot_lau', 'Talbot-Lau Interferometry', ['talbot lau', 'talbot-lau', 'grating interferometry x-ray']),
    ('tem', 'TEM', ['tem', 'transmission electron microscopy']),
    ('terahertz', 'Terahertz Imaging', ['terahertz', 'thz', 'thz imaging', 't-ray']),
    ('three_photon', 'Three-Photon Microscopy', ['three photon', '3-photon', 'third harmonic']),
    ('tirf', 'TIRF Microscopy', ['tirf', 'total internal reflection', 'evanescent wave microscopy']),
    ('tof_camera', 'Time-of-Flight Camera', ['tof camera', 'time of flight camera', 'depth camera', 'rgbd']),
    ('two_photon', 'Two-Photon Microscopy', ['two photon', '2-photon', 'multiphoton microscopy']),
    ('ultrasonic_phased_array', 'Ultrasonic Phased Array', ['phased array ultrasound', 'ultrasonic phased array', 'fmc']),
    ('ultrasound', 'Ultrasound', ['ultrasound', 'das beamforming', 'b-mode ultrasound']),
    ('us_mri', 'Ultrasound-MRI', ['us mri', 'ultrasound mri', 'combined us mri']),
    ('waxs', 'WAXS', ['waxs', 'wide angle x-ray scattering']),
    ('weather_radar', 'Weather Radar', ['weather radar', 'doppler radar', 'nexrad', 'precipitation radar']),
    ('widefield', 'Widefield Fluorescence', ['widefield', 'wide field fluorescence', 'epifluorescence']),
    ('widefield_lowdose', 'Widefield Low-Dose', ['widefield low dose', 'low dose fluorescence', 'photon limited']),
    ('xfel_sfx', 'XFEL Serial Crystallography', ['xfel', 'serial femtosecond crystallography', 'sfx']),
    ('xray_crystallography', 'X-ray Crystallography', ['x-ray crystallography', 'protein crystallography', 'structure determination']),
    ('xray_ndt', 'X-ray NDT', ['x-ray ndt', 'x-ray nondestructive', 'industrial radiography']),
    ('xray_radiography', 'X-ray Radiography', ['x-ray radiography', 'chest x-ray', 'plain radiograph']),
    ('xrf_imaging', 'XRF Imaging', ['xrf imaging', 'x-ray fluorescence imaging', 'xrf mapping']),
    ('xrf_tomo', 'XRF Tomography', ['xrf tomography', 'fluorescence ct', 'xrf tomo']),
]

# ── Score ─────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]', ' ', s.lower())

def _match(query: str) -> tuple[str, str]:
    """Return (modality_id, display_name) of best match."""
    q = _norm(query)
    q_words = set(q.split())
    best_id, best_name, best_score = '', '', 0.0

    for mod_id, disp_name, keywords in MODALITIES:
        score = 0.0
        for kw in keywords:
            kw_n = _norm(kw)
            kw_words = kw_n.split()
            if len(kw_words) == 1:
                if kw_n in q_words:
                    score += 1
            else:
                if kw_n in q:
                    score += len(kw_words)
        if score > best_score:
            best_score, best_id, best_name = score, mod_id, disp_name

    return best_id, best_name


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 spec/keyword_match.py \"your query\"")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])

    if query.strip().lower() == 'list':
        print(f"{'ID':<30} {'Display Name'}")
        print('-' * 60)
        for mod_id, disp_name, _ in MODALITIES:
            print(f"{mod_id:<30} {disp_name}")
        return

    print("[Keyword Match Mode — no LLM API key used]")
    mod_id, disp_name = _match(query)

    if not mod_id:
        print("No match found. Try: python3 spec/keyword_match.py list")
        sys.exit(1)

    spec_path = SPEC_DIR / f"{mod_id}.md"
    if not spec_path.exists():
        print(f"Match: {disp_name} ({mod_id})")
        print(f"Spec not generated yet — run: python3 spec/generate_specs.py")
        sys.exit(1)

    print(f"Match: {disp_name}")
    print(f"Spec:  spec/{mod_id}.md\n")
    print(spec_path.read_text())


if __name__ == '__main__':
    main()
