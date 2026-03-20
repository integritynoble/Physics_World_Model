#!/usr/bin/env python3
"""Generate minimal per-modality spec files for all 167 modalities.

Each spec covers all 4 use cases (reconstruct, mismatch, system design, simulation)
in ~30 lines. Reads SOLVERS dicts via AST parsing — no imports needed.

Usage:
    python3 spec/generate_specs.py
"""

import ast, os, re, sys
from pathlib import Path

BASE = Path(__file__).parent.parent   # pwm/public/
AB_DIR = BASE / 'algorithm_base'
SPEC_DIR = Path(__file__).parent      # pwm/public/spec/

# ── Input format per modality ─────────────────────────────────────────────
INPUT_FORMATS = {
    'acoustic_emission':    'waveform (samples, float32)',
    'acoustic_microscopy':  'RF data (H × W × T, float32)',
    'active_thermography':  'thermal sequence (T × H × W, float32)',
    'adaptive_optics':      'wavefront sensor data (H × W, float32)',
    'afm':                  'force-distance map (H × W, float32)',
    'angiography':          'projection (H × W, float32)',
    'asl_mri':              'label-control pairs (2 × H × W, float32)',
    'atom_probe':           'hit positions (N × 3, float32)',
    'bioluminescence_tomo': 'surface photon flux (H × W × angles, float32)',
    'brachytherapy_img':    'dose map (H × W × D, float32)',
    'brillouin':            'spectral shift map (H × W, float32)',
    'cacti':                'coded frames (T/B × H × W, float32)',
    'cars':                 'CARS spectrum (H × W × λ, float32)',
    'cassi':                'coded snapshot (H × W, float32)',
    'cathodoluminescence':  'spectrum image (H × W × λ, float32)',
    'cbct':                 'projections (angles × H × W, float32)',
    'cest_mri':             'Z-spectrum (offsets × H × W, float32)',
    'ceus':                 'contrast echo frames (T × H × W, float32)',
    'clem':                 'EM + fluorescence pairs (H × W, float32)',
    'coded_exposure':       'coded frames (N × H × W, float32)',
    'confocal_3d':          'Z-stack (Z × H × W, float32)',
    'confocal_endomicroscopy': 'confocal frame (H × W, float32)',
    'confocal_livecell':    'time-lapse (T × H × W, float32)',
    'coronagraphy':         'coronagraph image (H × W, float32)',
    'cryo_em':              'particle images (N × H × W, float32)',
    'cryo_et':              'tilt series (angles × H × W, float32)',
    'ct':                   'sinogram (angles × detectors, float32)',
    'ct_fluorescence':      'XRF sinogram (angles × detectors × channels, float32)',
    'cup':                  'streaked image (H × W_streak, float32)',
    'dark_field':           'grating image pairs (2 × H × W, float32)',
    'desi':                 'mass spectrum image (H × W × m/z, float32)',
    'dexa':                 'dual-energy projections (2 × H × W, float32)',
    'dic':                  'DIC image pair (H × W, float32)',
    'diffusion_mri':        'DWI volumes (N_dirs × H × W × D, float32)',
    'digital_breast_tomo':  'projections (angles × H × W, float32)',
    'dna_paint':            'localisation list (N × 2, float32)',
    'doppler_ultrasound':   'IQ data (H × W × T, float32)',
    'dot':                  'boundary flux measurements (sources × detectors, float32)',
    'ebsd':                 'Kikuchi pattern (H × W × px × py, float32)',
    'eddy_current':         'induced voltage (coils × time, float32)',
    'edx_mapping':          'X-ray counts (H × W × channels, float32)',
    'eels':                 'energy-loss spectrum (H × W × E, float32)',
    'eht_imaging':          'hologram (H × W, complex64)',
    'elastography':         'displacement field (H × W × 3, float32)',
    'electron_diffraction': 'diffraction pattern (H × W, float32)',
    'electron_holography':  'hologram (H × W, float32)',
    'electron_tomography':  'tilt series (angles × H × W, float32)',
    'endoscopy':            'endoscope image (H × W × 3, uint8)',
    'entangled_photon':     'coincidence counts (H × W, float32)',
    'event_camera':         'event stream (N × 4: t,x,y,p)',
    'expansion':            'confocal + expansion pairs (H × W, float32)',
    'fib_sem':              'cross-section stack (Z × H × W, uint8)',
    'flash_lidar':          'range + intensity (H × W × 2, float32)',
    'flim':                 'photon arrival times (H × W × T, float32)',
    'fluoroscopy':          'X-ray frame sequence (T × H × W, float32)',
    'fmri':                 'BOLD volumes (T × H × W × D, float32)',
    'fpm':                  'LED array images (N_leds × H × W, float32)',
    'ftir_imaging':         'interferogram (H × W × OPD, float32)',
    'fundus':               'fundus photograph (H × W × 3, uint8)',
    'fwi':                  'seismic waveforms (receivers × time, float32)',
    'gaussian_splatting':   'multi-view images (N × H × W × 3, float32)',
    'ghost_imaging':        'bucket detector signal (N_patterns, float32)',
    'gpr':                  'B-scan (traces × samples, float32)',
    'gravitational_wave':   'strain time series (samples, float32)',
    'hdr_imaging':          'multi-exposure stack (K × H × W × 3, uint8)',
    'holography':           'hologram (H × W, float32)',
    'hyperspectral_remote':  'radiance cube (H × W × bands, float32)',
    'impedance_tomo':       'boundary voltages (M, float32)',
    'industrial_ct':        'sinogram (angles × detectors, float32)',
    'insar':                'interferometric phase (H × W, float32)',
    'integral':             'integral image (H × W, float32)',
    'ism':                  'ISM raw stack (H_scan × W_scan × px × py, float32)',
    'ivus':                 'RF pullback (frames × elements × samples, float32)',
    'lattice_lightsheet':   'Z-stack (Z × H × W, float32)',
    'lensless':             'diffuser measurement (H × W, float32)',
    'libs':                 'emission spectrum (wavelengths, float32)',
    'lidar':                'point cloud (N × 3, float32)',
    'light_field':          'light field (u × v × s × t, float32)',
    'lightsheet':           'Z-stack (Z × H × W, float32)',
    'lucky_imaging':        'speckle frame stack (N × H × W, float32)',
    'machine_vision':       'image (H × W × 3, uint8)',
    'magnetic_particle':    'system function (freq × channels, complex64)',
    'maldi_msi':            'mass spectrum image (H × W × m/z, float32)',
    'mammography':          'projection pair (2 × H × W, float32)',
    'matrix':               'matrix completion (M × N, float32, with NaNs)',
    'mfm':                  'magnetic force map (H × W, float32)',
    'minflux':              'photon records (N × 5: t,x,y,z,id)',
    'mr_elastography':      'wave images (slices × H × W, complex64)',
    'mr_fingerprinting':    'signal evolution (T × H × W, complex64)',
    'mra':                  'angiography k-space (kx × ky × kz, complex64)',
    'mri':                  'k-space (H × W × 2 real+imag, float32)',
    'mrs':                  'FID time series (T, complex64)',
    'multispectral_sat':    'satellite radiance (H × W × bands, float32)',
    'muon_tomo':            'muon tracks (N × 6, float32)',
    'nerf':                 'posed images (N × H × W × 3, float32)',
    'neutron_diffraction':  'diffraction pattern (2θ × intensity, float32)',
    'neutron_tomo':         'projections (angles × H × W, float32)',
    'nirs_brain':           'time-resolved optical (channels × T, float32)',
    'nsom':                 'near-field signal (H × W, float32)',
    'ocean_acoustic_tomo':  'travel times (source-receiver pairs, float32)',
    'ocean_color':          'radiance (H × W × bands, float32)',
    'oct':                  'spectral interferogram (wavenumbers × A-scans, float32)',
    'octa':                 'OCT angiography B-scans (T × depth × A-scans, float32)',
    'odt':                  'holographic projections (angles × H × W, complex64)',
    'palm_storm':           'localisation list (N × 4: x,y,σ,intensity)',
    'panorama':             'overlapping images (N × H × W × 3, uint8)',
    'particle_calorimetry': 'energy deposits (N × 5, float32)',
    'passive_microwave':    'brightness temperature (H × W × channels, float32)',
    'pet':                  'sinogram (angles × detectors, float32)',
    'pet_ct':               'PET sinogram + CT projections (both float32)',
    'pet_mr':               'PET sinogram + MRI k-space (both float32)',
    'phase_contrast':       'propagation-based image (H × W, float32)',
    'phase_retrieval':      'diffraction intensities (H × W, float32)',
    'photoacoustic':        'PA time-series (elements × time, float32)',
    'photometric_stereo':   'images under N lights (N × H × W, float32)',
    'polarization':         'Stokes images (4 × H × W, float32)',
    'polsar':               'scattering matrix (H × W × 4, complex64)',
    'portal_imaging':       'EPID projection (H × W, float32)',
    'proton_radiography':   'proton fluence map (H × W, float32)',
    'proton_therapy_img':   'dose distribution (H × W × D, float32)',
    'ptychography':         'diffraction patterns (N_pos × H × W, float32)',
    'pump_probe':           'transient spectra (T × wavelengths, float32)',
    'quantum_illumination': 'coincidence image (H × W, float32)',
    'radio_astronomy':      'visibilities (baselines × freq × T, complex64)',
    'radio_interferometry': 'UV-plane data (N_baselines, complex64)',
    'raman_imaging':        'Raman spectra (H × W × wavenumbers, float32)',
    'sar':                  'raw SAR data (range × azimuth, complex64)',
    'saxs':                 'small-angle pattern (H × W, float32)',
    'seismic_tomo':         'travel times (source-receiver, float32)',
    'sem':                  'SEM image (H × W, uint16)',
    'shearography':         'shearogram pairs (2 × H × W, float32)',
    'shg':                  'SHG image stack (Z × H × W, float32)',
    'sim':                  'raw SIM frames (9 × H × W: 3 angles × 3 phases)',
    'sims':                 'ion images (H × W × m/z, float32)',
    'solar_imaging':        'EUV/AIA image (H × W, float32)',
    'sonar':                'echo data (elements × samples, float32)',
    'spc':                  'photon count frames (T × H × W, uint16)',
    'spect':                'projections (angles × detectors, float32)',
    'spect_ct':             'SPECT projections + CT sinogram (both float32)',
    'spectral_ct':          'energy-bin sinograms (bins × angles × detectors, float32)',
    'spinning_disk':        'confocal Z-stack (Z × H × W, float32)',
    'srs':                  'SRS image (H × W, float32)',
    'sted':                 'STED + confocal pairs (2 × H × W, float32)',
    'stem':                 'STEM-HAADF image (H × W, float32)',
    'stm':                  'tunneling current map (H × W, float32)',
    'streak_camera':        'streak image (time × space, float32)',
    'structured_light':     'pattern images (N × H × W, float32)',
    'swi':                  'phase image (H × W × slices, float32)',
    'talbot_lau':           'grating stepping images (N_steps × H × W, float32)',
    'tem':                  'TEM image (H × W, float32)',
    'terahertz':            'THz waveform (T × H × W, float32)',
    'three_photon':         'Z-stack (Z × H × W, float32)',
    'tirf':                 'TIRF frames (T × H × W, float32)',
    'tof_camera':           'depth + amplitude (H × W × 2, float32)',
    'two_photon':           'Z-stack (Z × H × W, float32)',
    'ultrasonic_phased_array': 'FMC data (elements × elements × time, float32)',
    'ultrasound':           'RF data (elements × samples, float32)',
    'us_mri':               'combined US + MRI data',
    'waxs':                 'wide-angle pattern (H × W, float32)',
    'weather_radar':        'reflectivity (H × W, float32)',
    'widefield':            'fluorescence image (H × W, float32)',
    'widefield_lowdose':    'photon-limited image (H × W, float32)',
    'xfel_sfx':             'diffraction patterns (N_shots × H × W, float32)',
    'xray_crystallography': 'structure factor amplitudes (hkl × F, float32)',
    'xray_ndt':             'projection (H × W, float32)',
    'xray_radiography':     'attenuation image (H × W, float32)',
    'xrf_imaging':          'fluorescence map (H × W × elements, float32)',
    'xrf_tomo':             'XRF sinograms (elements × angles × detectors, float32)',
}

# ── Papers examples per modality ─────────────────────────────────────────
PAPER_LINKS = {
    'ct': [
        'papers/system_design/outputs/ct_forward_v1_iter1.md',
        'papers/system_design/outputs/ct_reconstruction_v1_iter1.md',
    ],
    'lensless': [
        'papers/system_design/outputs/lensless_forward_v1_iter1.md',
        'papers/system_design/outputs/lensless_reconstruction_v1_iter1.md',
        'papers/system_design/outputs/lensless_3d_forward_v1_iter1.md',
    ],
    'sim': [
        'papers/system_design/outputs/sim_forward_v1_iter1.md',
        'papers/system_design/outputs/sim_reconstruction_v1_iter1.md',
    ],
    'cassi': [
        'papers/system_design/outputs/spectral_lensless_forward_v1_iter1.md',
        'papers/inversenet/README_CASSI.md',
    ],
    'cacti': [
        'papers/inversenet/CACTI_IMPLEMENTATION_COMPLETE.md',
    ],
    'spc': [
        'papers/inversenet/SPC_IMPLEMENTATION_COMPLETE.md',
        'papers/inversenet/SPC_RESULTS.md',
    ],
}

# ── Physics simulation examples (universal_simulation benchmark) ──────────
SIMULATION_LINKS = {
    'ultrasound': 'papers/universal_simulation/benchmark/01_classical_mechanics/',
    'terahertz': 'papers/universal_simulation/benchmark/02_electromagnetics/',
    'photoacoustic': 'papers/universal_simulation/benchmark/02_electromagnetics/',
    'mri': 'papers/universal_simulation/benchmark/02_electromagnetics/',
    'phase_retrieval': 'papers/universal_simulation/benchmark/03_quantum_chemistry/',
    'oct': 'papers/universal_simulation/benchmark/09_optics/',
    'lensless': 'papers/universal_simulation/benchmark/09_optics/',
    'fwi': 'papers/universal_simulation/benchmark/11_seismic/',
    'seismic_tomo': 'papers/universal_simulation/benchmark/11_seismic/',
}

# ── Mismatch parameters per modality ─────────────────────────────────────
MISMATCH_PARAMS = {
    'ct':       'center-of-rotation offset (px)',
    'mri':      'coil sensitivity maps (B1 field)',
    'cassi':    'dispersion step (px)',
    'lensless': 'PSF shift (px)',
    'cbct':     'geometric calibration (source-detector distance)',
    'ultrasound': 'sound speed (m/s)',
    'oct':      'dispersion coefficients (β₂, β₃)',
    'pet':      'attenuation map (μ-map)',
    'spect':    'scatter fraction',
    'sim':      'illumination pattern phase',
}


def extract_modality_info(solvers_py: Path):
    """Parse MODALITY_ID, DISPLAY_NAME, and SOLVERS from solvers.py."""
    src = solvers_py.read_text(encoding='utf-8', errors='replace')

    mod_id = ''
    disp_name = ''
    m = re.search(r'^MODALITY_ID\s*=\s*["\'](.+?)["\']', src, re.MULTILINE)
    if m:
        mod_id = m.group(1)
    m = re.search(r'^DISPLAY_NAME\s*=\s*["\'](.+?)["\']', src, re.MULTILINE)
    if m:
        disp_name = m.group(1)

    # Extract SOLVERS dict via AST
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

    return mod_id, disp_name, solvers


def extract_psnr(ref: str) -> str:
    """Parse '— 29.5 dB' from reference string."""
    m = re.search(r'(\d+\.?\d*)\s*dB', ref or '')
    return f"~{m.group(1)} dB" if m else ''


def build_algo_table(solvers: dict, modality_id: str) -> str:
    """Build compact algorithm table showing best CPU/GPU + total count."""
    cpu_list = []
    gpu_list = []
    for key, v in solvers.items():
        if not isinstance(v, dict):
            continue
        name = v.get('name', key)
        is_gpu = bool(v.get('gpu', False))
        psnr = extract_psnr(v.get('reference', ''))
        (gpu_list if is_gpu else cpu_list).append((key, name, psnr))

    n_cpu = len(cpu_list)
    n_gpu = len(gpu_list)
    total = n_cpu + n_gpu
    if total == 0:
        return f"Use `list_solvers()` to see all algorithms.\n"

    lines = [f"| Key | Name | PSNR | Device |",
             f"|-----|------|------|--------|"]

    # Show all CPU solvers
    for key, name, psnr in cpu_list:
        lines.append(f"| `{key}` | {name} | {psnr} | CPU |")

    # Show all GPU solvers
    for key, name, psnr in gpu_list:
        lines.append(f"| `{key}` | {name} | {psnr} | GPU |")

    return '\n'.join(lines)


def generate_spec(mod_dir: Path) -> str:
    """Generate minimal spec markdown for one modality."""
    solvers_py = mod_dir / 'solvers.py'
    if not solvers_py.exists():
        return ''

    mod_id, disp_name, solvers = extract_modality_info(solvers_py)
    if not mod_id:
        mod_id = mod_dir.name
    if not disp_name:
        disp_name = mod_id.replace('_', ' ').title()

    input_fmt = INPUT_FORMATS.get(mod_id, 'measurement (H × W, float32)')
    total = sum(1 for v in solvers.values() if isinstance(v, dict))
    algo_table = build_algo_table(solvers, mod_id)
    gcs_path = f"gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/"

    # Run button
    run_code = f'''\
```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.{mod_id}.solvers import run_solver, list_solvers
list_solvers()                    # {total} algorithms
y = ...                           # {input_fmt}
x = run_solver('best_quality', y) # swap key to compare
```'''

    # Mismatch section
    mismatch_param = MISMATCH_PARAMS.get(mod_id, '')
    mismatch_section = ''
    if mismatch_param:
        mismatch_section = f"\n## Mismatch\n\nKey mismatch: **{mismatch_param}**  \nCorrection: `run_solver('best_quality', y, cfg={{'calibrate': True}})`\n"

    # Papers section
    paper_links = PAPER_LINKS.get(mod_id, [])
    sim_link = SIMULATION_LINKS.get(mod_id, '')
    papers_section = ''
    all_links = paper_links + ([sim_link] if sim_link else [])
    if all_links:
        papers_section = '\n## Papers\n\n' + '\n'.join(f'- `{lnk}`' for lnk in all_links) + '\n'

    spec = f"""\
# {disp_name}

**Input**: {input_fmt}
**Benchmark**: `{gcs_path}`

## Algorithms ({total} total)

{algo_table}

## Run

{run_code}
{mismatch_section}{papers_section}"""

    return spec


def main():
    if not AB_DIR.exists():
        print(f"ERROR: algorithm_base not found at {AB_DIR}")
        sys.exit(1)

    modality_dirs = sorted(d for d in AB_DIR.iterdir()
                           if d.is_dir() and d.name != 'shared')

    generated = 0
    skipped = 0
    for mod_dir in modality_dirs:
        spec_content = generate_spec(mod_dir)
        if not spec_content:
            skipped += 1
            continue

        out_path = SPEC_DIR / f"{mod_dir.name}.md"
        out_path.write_text(spec_content, encoding='utf-8')
        generated += 1

    print(f"Generated {generated} spec files in {SPEC_DIR}")
    if skipped:
        print(f"Skipped {skipped} directories (no solvers.py)")


if __name__ == '__main__':
    main()
