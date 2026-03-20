"""Shared data for all 4 spec generators."""
import ast, re
from pathlib import Path

BASE = Path(__file__).parent.parent.parent   # pwm/public/
AB_DIR = BASE / 'algorithm_base'
SPEC_DIR = Path(__file__).parent.parent      # pwm/public/spec/

# ── Input formats ─────────────────────────────────────────────────────────
INPUT_FORMATS = {
    'acoustic_emission':    'waveform (samples, float32)',
    'acoustic_microscopy':  'RF data (H × W × T, float32)',
    'active_thermography':  'thermal sequence (T × H × W, float32)',
    'adaptive_optics':      'wavefront sensor (H × W, float32)',
    'afm':                  'force-distance map (H × W, float32)',
    'angiography':          'projection (H × W, float32)',
    'asl_mri':              'label-control pairs (2 × H × W, float32)',
    'atom_probe':           'hit positions (N × 3, float32)',
    'bioluminescence_tomo': 'surface flux (H × W × angles, float32)',
    'brachytherapy_img':    'dose map (H × W × D, float32)',
    'brillouin':            'spectral shift map (H × W, float32)',
    'cacti':                'coded frames (B × H × W, float32)',
    'cars':                 'CARS spectrum cube (H × W × λ, float32)',
    'cassi':                'coded snapshot (H × W, float32)',
    'cathodoluminescence':  'spectrum image (H × W × λ, float32)',
    'cbct':                 'projections (angles × H × W, float32)',
    'cest_mri':             'Z-spectrum (offsets × H × W, float32)',
    'ceus':                 'contrast frames (T × H × W, float32)',
    'clem':                 'EM + fluorescence (H × W, float32)',
    'coded_exposure':       'coded frames (N × H × W, float32)',
    'confocal_3d':          'Z-stack (Z × H × W, float32)',
    'confocal_endomicroscopy': 'confocal frame (H × W, float32)',
    'confocal_livecell':    'time-lapse (T × H × W, float32)',
    'coronagraphy':         'image (H × W, float32)',
    'cryo_em':              'particle images (N × H × W, float32)',
    'cryo_et':              'tilt series (angles × H × W, float32)',
    'ct':                   'sinogram (angles × detectors, float32)',
    'ct_fluorescence':      'XRF sinogram (angles × detectors × ch, float32)',
    'cup':                  'streak image (H × W_streak, float32)',
    'dark_field':           'grating images (2 × H × W, float32)',
    'desi':                 'mass image (H × W × m/z, float32)',
    'dexa':                 'dual-energy projections (2 × H × W, float32)',
    'dic':                  'DIC image (H × W, float32)',
    'diffusion_mri':        'DWI (N_dirs × H × W × D, float32)',
    'digital_breast_tomo':  'projections (angles × H × W, float32)',
    'dna_paint':            'localisation list (N × 2, float32)',
    'doppler_ultrasound':   'IQ data (H × W × T, float32)',
    'dot':                  'boundary flux (sources × detectors, float32)',
    'ebsd':                 'Kikuchi pattern (H × W × px × py, float32)',
    'eddy_current':         'induced voltage (coils × time, float32)',
    'edx_mapping':          'X-ray counts (H × W × channels, float32)',
    'eels':                 'energy-loss spectrum (H × W × E, float32)',
    'eht_imaging':          'hologram (H × W, complex64)',
    'elastography':         'displacement (H × W × 3, float32)',
    'electron_diffraction': 'diffraction pattern (H × W, float32)',
    'electron_holography':  'hologram (H × W, float32)',
    'electron_tomography':  'tilt series (angles × H × W, float32)',
    'endoscopy':            'image (H × W × 3, uint8)',
    'entangled_photon':     'coincidence counts (H × W, float32)',
    'event_camera':         'event stream (N × 4: t,x,y,p)',
    'expansion':            'confocal + expansion (H × W, float32)',
    'fib_sem':              'cross-sections (Z × H × W, uint8)',
    'flash_lidar':          'range + intensity (H × W × 2, float32)',
    'flim':                 'photon arrivals (H × W × T, float32)',
    'fluoroscopy':          'X-ray frames (T × H × W, float32)',
    'fmri':                 'BOLD volumes (T × H × W × D, float32)',
    'fpm':                  'LED images (N_leds × H × W, float32)',
    'ftir_imaging':         'interferogram (H × W × OPD, float32)',
    'fundus':               'photograph (H × W × 3, uint8)',
    'fwi':                  'seismic waveforms (receivers × time, float32)',
    'gaussian_splatting':   'posed images (N × H × W × 3, float32)',
    'ghost_imaging':        'bucket signal (N_patterns, float32)',
    'gpr':                  'B-scan (traces × samples, float32)',
    'gravitational_wave':   'strain (samples, float32)',
    'hdr_imaging':          'multi-exposure (K × H × W × 3, uint8)',
    'holography':           'hologram (H × W, float32)',
    'hyperspectral_remote': 'radiance cube (H × W × bands, float32)',
    'impedance_tomo':       'boundary voltages (M, float32)',
    'industrial_ct':        'sinogram (angles × detectors, float32)',
    'insar':                'interferometric phase (H × W, float32)',
    'integral':             'integral image (H × W, float32)',
    'ism':                  'raw stack (H_scan × W_scan × px × py, float32)',
    'ivus':                 'RF pullback (frames × elements × samples, float32)',
    'lattice_lightsheet':   'Z-stack (Z × H × W, float32)',
    'lensless':             'diffuser measurement (H × W, float32)',
    'libs':                 'emission spectrum (wavelengths, float32)',
    'lidar':                'point cloud (N × 3, float32)',
    'light_field':          'light field (u × v × s × t, float32)',
    'lightsheet':           'Z-stack (Z × H × W, float32)',
    'lucky_imaging':        'speckle frames (N × H × W, float32)',
    'machine_vision':       'image (H × W × 3, uint8)',
    'magnetic_particle':    'system function (freq × ch, complex64)',
    'maldi_msi':            'mass image (H × W × m/z, float32)',
    'mammography':          'projection pair (2 × H × W, float32)',
    'matrix':               'partial matrix (M × N, float32, NaN=missing)',
    'mfm':                  'magnetic force map (H × W, float32)',
    'minflux':              'photon records (N × 5: t,x,y,z,id)',
    'mr_elastography':      'wave images (slices × H × W, complex64)',
    'mr_fingerprinting':    'signal evolution (T × H × W, complex64)',
    'mra':                  'k-space (kx × ky × kz, complex64)',
    'mri':                  'k-space (H × W × 2: real+imag, float32)',
    'mrs':                  'FID (T, complex64)',
    'multispectral_sat':    'radiance (H × W × bands, float32)',
    'muon_tomo':            'muon tracks (N × 6, float32)',
    'nerf':                 'posed images (N × H × W × 3, float32)',
    'neutron_diffraction':  'pattern (2θ × intensity, float32)',
    'neutron_tomo':         'projections (angles × H × W, float32)',
    'nirs_brain':           'optical signal (channels × T, float32)',
    'nsom':                 'near-field signal (H × W, float32)',
    'ocean_acoustic_tomo':  'travel times (pairs, float32)',
    'ocean_color':          'radiance (H × W × bands, float32)',
    'oct':                  'spectrum (wavenumbers × A-scans, float32)',
    'octa':                 'B-scans (T × depth × A-scans, float32)',
    'odt':                  'holograms (angles × H × W, complex64)',
    'palm_storm':           'localisations (N × 4: x,y,σ,I)',
    'panorama':             'images (N × H × W × 3, uint8)',
    'particle_calorimetry': 'deposits (N × 5, float32)',
    'passive_microwave':    'brightness T (H × W × ch, float32)',
    'pet':                  'sinogram (angles × detectors, float32)',
    'pet_ct':               'PET sino + CT proj (both float32)',
    'pet_mr':               'PET sino + MRI k-space (both float32)',
    'phase_contrast':       'image (H × W, float32)',
    'phase_retrieval':      'diffraction intensities (H × W, float32)',
    'photoacoustic':        'time-series (elements × time, float32)',
    'photometric_stereo':   'images under N lights (N × H × W, float32)',
    'polarization':         'Stokes images (4 × H × W, float32)',
    'polsar':               'scattering matrix (H × W × 4, complex64)',
    'portal_imaging':       'EPID projection (H × W, float32)',
    'proton_radiography':   'fluence map (H × W, float32)',
    'proton_therapy_img':   'dose distribution (H × W × D, float32)',
    'ptychography':         'diffraction patterns (N_pos × H × W, float32)',
    'pump_probe':           'transient spectra (T × λ, float32)',
    'quantum_illumination': 'coincidence image (H × W, float32)',
    'radio_astronomy':      'visibilities (baselines × freq × T, complex64)',
    'radio_interferometry': 'UV-plane data (N_baselines, complex64)',
    'raman_imaging':        'Raman spectra (H × W × wn, float32)',
    'sar':                  'raw data (range × azimuth, complex64)',
    'saxs':                 'scattering pattern (H × W, float32)',
    'seismic_tomo':         'travel times (src-recv, float32)',
    'sem':                  'SEM image (H × W, uint16)',
    'shearography':         'shearograms (2 × H × W, float32)',
    'shg':                  'Z-stack (Z × H × W, float32)',
    'sim':                  'raw frames (9 × H × W: 3 angles × 3 phases)',
    'sims':                 'ion images (H × W × m/z, float32)',
    'solar_imaging':        'EUV image (H × W, float32)',
    'sonar':                'echo data (elements × samples, float32)',
    'spc':                  'photon counts (T × H × W, uint16)',
    'spect':                'projections (angles × detectors, float32)',
    'spect_ct':             'SPECT proj + CT sino (both float32)',
    'spectral_ct':          'energy-bin sinos (bins × angles × det, float32)',
    'spinning_disk':        'Z-stack (Z × H × W, float32)',
    'srs':                  'SRS image (H × W, float32)',
    'sted':                 'STED + confocal (2 × H × W, float32)',
    'stem':                 'HAADF image (H × W, float32)',
    'stm':                  'tunneling map (H × W, float32)',
    'streak_camera':        'streak image (time × space, float32)',
    'structured_light':     'pattern images (N × H × W, float32)',
    'swi':                  'phase image (H × W × slices, float32)',
    'talbot_lau':           'stepping images (N_steps × H × W, float32)',
    'tem':                  'TEM image (H × W, float32)',
    'terahertz':            'THz waveform (T × H × W, float32)',
    'three_photon':         'Z-stack (Z × H × W, float32)',
    'tirf':                 'TIRF frames (T × H × W, float32)',
    'tof_camera':           'depth + amplitude (H × W × 2, float32)',
    'two_photon':           'Z-stack (Z × H × W, float32)',
    'ultrasonic_phased_array': 'FMC data (elem × elem × time, float32)',
    'ultrasound':           'RF data (elements × samples, float32)',
    'us_mri':               'US + MRI combined data',
    'waxs':                 'wide-angle pattern (H × W, float32)',
    'weather_radar':        'reflectivity (H × W, float32)',
    'widefield':            'fluorescence image (H × W, float32)',
    'widefield_lowdose':    'photon-limited image (H × W, float32)',
    'xfel_sfx':             'diffraction patterns (N_shots × H × W, float32)',
    'xray_crystallography': 'structure factors (hkl × F, float32)',
    'xray_ndt':             'projection (H × W, float32)',
    'xray_radiography':     'attenuation image (H × W, float32)',
    'xrf_imaging':          'fluorescence map (H × W × elements, float32)',
    'xrf_tomo':             'XRF sinograms (elem × angles × det, float32)',
}

# ── 3D volumetric modalities (show orthogonal slices) ─────────────────────
VOLUMETRIC = {
    'cbct', 'confocal_3d', 'confocal_livecell', 'cryo_et',
    'diffusion_mri', 'electron_tomography', 'fib_sem', 'fmri',
    'lattice_lightsheet', 'lightsheet', 'mra', 'mr_fingerprinting',
    'neutron_tomo', 'proton_therapy_img', 'shg', 'spinning_disk',
    'three_photon', 'two_photon', 'xrf_tomo', 'brachytherapy_img',
}

# ── Hyperspectral modalities (show band slice) ────────────────────────────
HYPERSPECTRAL = {
    'cars', 'cassi', 'cathodoluminescence', 'desi', 'edx_mapping',
    'eels', 'ftir_imaging', 'hyperspectral_remote', 'maldi_msi',
    'multispectral_sat', 'ocean_color', 'raman_imaging', 'sims',
    'spectral_ct', 'xrf_tomo', 'ct_fluorescence',
}

# ── Mismatch parameters per modality ─────────────────────────────────────
MISMATCH = {
    'ct': {
        'param': 'center-of-rotation (CoR) offset',
        'range': '[-5, +5] px',
        'typical': '±1 px',
        'method': 'cross-correlation on 180°-opposed projections',
        'cfg_key': 'cor_offset',
    },
    'mri': {
        'param': 'coil sensitivity maps (B1 field error)',
        'range': '[0.9, 1.1] gain per coil',
        'typical': '±5% gain per coil',
        'method': 'ESPIRiT auto-calibration from ACS region',
        'cfg_key': 'sens_error',
    },
    'cassi': {
        'param': 'dispersion step (px)',
        'range': '[1, 5] px',
        'typical': '±0.5 px drift',
        'method': 'grid search on sparsity metric',
        'cfg_key': 'disp_step',
    },
    'lensless': {
        'param': 'PSF shift (px)',
        'range': '[-5, +5] px (x and y)',
        'typical': '±1 px thermal drift',
        'method': 'gradient calibration against nominal PSF',
        'cfg_key': 'psf_shift',
    },
    'cbct': {
        'param': 'source-detector distance (SAD/SDD)',
        'range': 'SAD ±5 mm, SDD ±10 mm',
        'typical': '±2 mm per session',
        'method': 'phantom-based geometric calibration',
        'cfg_key': 'geometry_error',
    },
    'ultrasound': {
        'param': 'sound speed (m/s)',
        'range': '[1400, 1600] m/s',
        'typical': '±30 m/s tissue variation',
        'method': 'coherence-based speed estimation',
        'cfg_key': 'c0',
    },
    'oct': {
        'param': 'dispersion coefficients (β₂, β₃)',
        'range': 'β₂ ∈ [-1e-27, 1e-27] s²/m',
        'typical': '±2e-28 s²/m',
        'method': 'numerical dispersion compensation (NDC)',
        'cfg_key': 'disp_coeff',
    },
    'pet': {
        'param': 'attenuation map (μ-map)',
        'range': 'μ ∈ [0, 0.3] cm⁻¹',
        'typical': '±10% scaling error',
        'method': 'CT-based attenuation correction',
        'cfg_key': 'mu_map',
    },
    'spect': {
        'param': 'scatter fraction',
        'range': '[0.1, 0.4]',
        'typical': '~0.25 in body imaging',
        'method': 'triple-energy window (TEW) subtraction',
        'cfg_key': 'scatter_frac',
    },
    'sim': {
        'param': 'illumination pattern phase offset',
        'range': '[0, 2π/3] rad',
        'typical': '±0.1 rad drift',
        'method': 'cross-power spectrum phase estimation',
        'cfg_key': 'phase_offset',
    },
    'photoacoustic': {
        'param': 'speed of sound (m/s)',
        'range': '[1480, 1560] m/s',
        'typical': '±20 m/s in tissue',
        'method': 'model-based estimation from time-of-flight',
        'cfg_key': 'c0',
    },
    'industrial_ct': {
        'param': 'center-of-rotation (CoR) offset',
        'range': '[-10, +10] px',
        'typical': '±2 px',
        'method': 'sinogram symmetry calibration',
        'cfg_key': 'cor_offset',
    },
    'fpm': {
        'param': 'LED position error (mm)',
        'range': '[-2, +2] mm (x, y)',
        'typical': '±0.5 mm',
        'method': 'embedded pupil function recovery (EPRY)',
        'cfg_key': 'led_pos_error',
    },
    'ptychography': {
        'param': 'probe position error (px)',
        'range': '[-3, +3] px',
        'typical': '±1 px',
        'method': 'annealing position refinement',
        'cfg_key': 'pos_error',
    },
    'phase_retrieval': {
        'param': 'detector-sample distance (mm)',
        'range': '[-5, +5] mm',
        'typical': '±2 mm',
        'method': 'Fresnel propagation distance calibration',
        'cfg_key': 'dist_error',
    },
}

# ── Best CPU solver per modality ──────────────────────────────────────────
BEST_CPU = {
    'ct': 'pnp_admm_nlm',
    'mri': 'espirit',
    'cassi': 'tv_admm',
    'lensless': 'tv_admm',
    'cbct': 'tv_admm',
    'pet': 'osem',
    'spect': 'os_em',
    'oct': 'bm3d_deconv',
    'sim': 'wienersim',
    'ultrasound': 'tv_admm',
    'photoacoustic': 'tr_reconstruction',
    'fpm': 'admm_ptf',
    'ptychography': 'epie',
    'phase_retrieval': 'raar',
}


def extract_modality_info(solvers_py: Path):
    """Parse MODALITY_ID, DISPLAY_NAME, docstring, SOLVERS dict."""
    src = solvers_py.read_text(encoding='utf-8', errors='replace')

    mod_id = re.search(r'^MODALITY_ID\s*=\s*["\'](.+?)["\']', src, re.MULTILINE)
    disp_name = re.search(r'^DISPLAY_NAME\s*=\s*["\'](.+?)["\']', src, re.MULTILINE)

    # Docstring (first triple-quoted string)
    doc = ''
    m = re.search(r'"""(.*?)"""', src, re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
        # Find "Dataset:" line
        dataset_line = next((l for l in lines if l.startswith('Dataset:')), '')
        doc = dataset_line

    # SOLVERS dict via AST
    solvers = {}
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'SOLVERS':
                        try:
                            solvers = ast.literal_eval(node.value)
                        except Exception:
                            pass
    except SyntaxError:
        pass

    return (
        mod_id.group(1) if mod_id else solvers_py.parent.name,
        disp_name.group(1) if disp_name else solvers_py.parent.name.replace('_', ' ').title(),
        doc,
        solvers,
    )


def extract_psnr(ref: str) -> str:
    m = re.search(r'(\d+\.?\d*)\s*dB', ref or '')
    return f"~{m.group(1)} dB" if m else ''


def extract_ssim(ref: str) -> str:
    m = re.search(r'SSIM[:\s]+(\d+\.?\d*)', ref or '', re.IGNORECASE)
    return f"~{m.group(1)}" if m else ''


def get_input(mod_id: str) -> str:
    return INPUT_FORMATS.get(mod_id, 'measurement (H × W, float32)')


def viz_code(mod_id: str, solver_key: str) -> str:
    """Return visualization code block for this modality."""
    if mod_id in VOLUMETRIC:
        return f"""\
# Visualize (3D: orthogonal slices)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
mid = [s // 2 for s in x.shape]
axes[0, 0].imshow(x[mid[0]], cmap='gray'); axes[0, 0].set_title('Recon axial')
axes[0, 1].imshow(x[:, mid[1], :], cmap='gray'); axes[0, 1].set_title('Recon coronal')
axes[0, 2].imshow(x[:, :, mid[2]], cmap='gray'); axes[0, 2].set_title('Recon sagittal')
if x_true is not None:
    axes[1, 0].imshow(x_true[mid[0]], cmap='gray'); axes[1, 0].set_title('GT axial')
    axes[1, 1].imshow(x_true[:, mid[1], :], cmap='gray'); axes[1, 1].set_title('GT coronal')
    axes[1, 2].imshow(x_true[:, :, mid[2]], cmap='gray'); axes[1, 2].set_title('GT sagittal')
plt.tight_layout(); plt.savefig('{mod_id}_{solver_key}.png'); plt.show()"""
    elif mod_id in HYPERSPECTRAL:
        return f"""\
# Visualize (hyperspectral: spatial slice + spectral profile)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
n_bands = x.shape[-1] if x.ndim == 3 else x.shape[0]
band = n_bands // 2
xb = x[..., band] if x.ndim == 3 else x[band]
axes[0].imshow(xb, cmap='gray'); axes[0].set_title(f'Recon band {{band}}')
axes[1].plot(x[x.shape[0]//2, x.shape[1]//2] if x.ndim==3 else x[:,x.shape[1]//2,x.shape[2]//2])
axes[1].set_title('Spectral profile (center pixel)')
if x_true is not None:
    xtb = x_true[..., band] if x_true.ndim == 3 else x_true[band]
    axes[2].imshow(xtb, cmap='gray'); axes[2].set_title('GT band')
plt.tight_layout(); plt.savefig('{mod_id}_{solver_key}.png'); plt.show()"""
    else:
        return f"""\
# Visualize
import matplotlib.pyplot as plt
ncols = 3 if x_true is not None else 2
fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
y_show = y if y.ndim == 2 else y[0] if y.ndim == 3 else y
axes[0].imshow(y_show, cmap='gray'); axes[0].set_title('Measurement')
axes[1].imshow(x, cmap='gray'); axes[1].set_title('Reconstruction')
if x_true is not None:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('{mod_id}_{solver_key}.png'); plt.show()"""
