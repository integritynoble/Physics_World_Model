"""
Build standard synthetic datasets for all remaining modalities (141 of 168).
Generates HDF5 files with synthetic ground truth + ideal forward model.
Status: built-synthetic (needs real public data to be marked done).

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
import yaml
import os
from pathlib import Path
from scipy import ndimage

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
CFG_DIR = ROOT / "benchmarks" / "configs"
DATA_DIR = ROOT / "datasets" / "benchmark"

# ═══════════════════════════════════════════════════════════════════
# Ground truth generators
# ═══════════════════════════════════════════════════════════════════

def smooth_2d(H, W, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.randn(H, W).astype(np.float32)
    smooth = ndimage.gaussian_filter(noise, sigma=min(H, W) / 8)
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W / 2, H / 2
    for _ in range(rng.randint(3, 8)):
        r = rng.uniform(0.05, 0.3) * min(H, W)
        dx = rng.uniform(-0.3, 0.3) * W
        dy = rng.uniform(-0.3, 0.3) * H
        smooth += rng.uniform(0.3, 1.0) * np.exp(
            -((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2) / (2 * r ** 2)
        )
    smooth -= smooth.min()
    smooth /= smooth.max() + 1e-10
    return smooth.astype(np.float32)


def shepp_logan(H, W, seed=0):
    x = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[:H, :W]
    yy = (yy - H / 2) / (H / 2)
    xx = (xx - W / 2) / (W / 2)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 1.0),
        (0.0, 0.0, 0.6624, 0.874, 0.8),
        (0.22, 0.0, 0.11, 0.31, 0.2),
        (-0.22, 0.0, 0.16, 0.41, 0.3),
        (0.0, 0.35, 0.21, 0.25, 0.4),
        (0.0, 0.1, 0.046, 0.046, 0.6),
        (-0.08, -0.605, 0.046, 0.023, 0.5),
    ]
    for cx, cy, a, b, val in ellipses:
        mask = ((xx - cx) ** 2 / a ** 2 + (yy - cy) ** 2 / b ** 2) <= 1.0
        x[mask] = val
    rng = np.random.RandomState(seed)
    x += 0.05 * rng.randn(H, W).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def smooth_3d(D, H, W, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.randn(D, H, W).astype(np.float32)
    smooth = ndimage.gaussian_filter(noise, sigma=min(D, H, W) / 6)
    smooth -= smooth.min()
    smooth /= smooth.max() + 1e-10
    return smooth.astype(np.float32)


def spectral_cube(H, W, C, seed=0):
    rng = np.random.RandomState(seed)
    n_end = min(5, C)
    endmembers = np.zeros((n_end, C), dtype=np.float32)
    for i in range(n_end):
        center = rng.uniform(0.2, 0.8) * C
        width = rng.uniform(0.05, 0.2) * C
        endmembers[i] = np.exp(-0.5 * ((np.arange(C) - center) / max(width, 0.1)) ** 2)
    cube = np.zeros((H, W, C), dtype=np.float32)
    for i in range(n_end):
        ab = smooth_2d(H, W, seed=seed + i + 1)
        cube += ab[:, :, None] * endmembers[i][None, None, :]
    cube -= cube.min()
    cube /= cube.max() + 1e-10
    return cube.astype(np.float32)


def signal_1d(N, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, N)
    x = np.zeros(N, dtype=np.float32)
    for _ in range(rng.randint(3, 10)):
        c = rng.uniform(0.1, 0.9)
        w = rng.uniform(0.005, 0.05)
        a = rng.uniform(0.1, 1.0)
        x += a * np.exp(-0.5 * ((t - c) / w) ** 2)
    x = np.clip(x, 0, None)
    x /= x.max() + 1e-10
    return x.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Forward models
# ═══════════════════════════════════════════════════════════════════

def radon_fwd(x, n_angles=120):
    H, W = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino = np.zeros((n_angles, W), dtype=np.float32)
    for i, a in enumerate(angles):
        rot = ndimage.rotate(x, a, reshape=False, order=1)
        sino[i] = rot.sum(axis=0)
    sino /= sino.max() + 1e-10
    return sino


def fourier_fwd(x, accel=4):
    k = np.fft.fft2(x)
    mask = np.zeros(x.shape, dtype=bool)
    H = x.shape[0]
    nc = max(1, int(H * 0.08))
    mask[H // 2 - nc // 2 : H // 2 + nc // 2, :] = True
    rng = np.random.RandomState(42)
    extra = max(1, H // accel - nc)
    idx = rng.choice(H, extra, replace=False)
    mask[idx, :] = True
    y = k * mask
    return np.stack([y.real, y.imag], axis=-1).astype(np.float32)


def psf_fwd(x, sigma=2.0):
    if x.ndim <= 2:
        return ndimage.gaussian_filter(x, sigma=sigma).astype(np.float32)
    y = np.zeros_like(x)
    for c in range(x.shape[-1]):
        y[..., c] = ndimage.gaussian_filter(x[..., c], sigma=sigma)
    return y.astype(np.float32)


def downsample_fwd(x, factor=2):
    slc = tuple(slice(None, None, factor) for _ in range(min(x.ndim, 2)))
    if x.ndim > 2:
        slc = slc + (slice(None),) * (x.ndim - 2)
    return x[slc].copy().astype(np.float32)


def mask_fwd(x, density=0.3, seed=42):
    rng = np.random.RandomState(seed)
    m = (rng.rand(*x.shape) < density).astype(np.float32)
    return (x * m).astype(np.float32)


def identity_fwd(x):
    return x.copy()


def proj3d_fwd(vol, n_angles=60):
    D, H, W = vol.shape
    sino = np.zeros((n_angles, D, W), dtype=np.float32)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    for i, a in enumerate(angles):
        rot = ndimage.rotate(vol, a, axes=(1, 2), reshape=False, order=1)
        sino[i] = rot.sum(axis=2)
    sino /= sino.max() + 1e-10
    return sino


# ═══════════════════════════════════════════════════════════════════
# Modality definitions: (gt_type, fwd_type, n, params_dict)
# gt_type: shepp | 2d | 3d | spectral | 1d
# fwd_type: radon | fourier | psf | downsample | mask | identity | proj3d
# ═══════════════════════════════════════════════════════════════════

MODS = {
    # ── Medical CT-like ──
    "cbct":                 ("shepp","radon",10,{"src":"AAPM Low-Dose CT 2016","ref":"aapm.org/grandchallenge/lowdosect"}),
    "industrial_ct":        ("shepp","radon",10,{"src":"WoDT industrial CT","ref":"Simulation"}),
    "digital_breast_tomo":  ("shepp","radon",10,{"src":"VDM-100 DBT (TCIA)","ref":"cancerimagingarchive.net"}),
    "spectral_ct":          ("shepp","radon",10,{"src":"AAPM Spectral CT","ref":"aapm.org/grandchallenge"}),
    "mammography":          ("shepp","radon",10,{"src":"CBIS-DDSM","ref":"doi:10.1038/sdata.2017.177"}),
    "xray_radiography":     ("2d","psf",10,{"s":1.5,"src":"Chest X-ray14 (NIH)","ref":"doi:10.1109/CVPR.2017.369"}),
    "xray_ndt":             ("shepp","radon",10,{"src":"WoDT benchmark","ref":"Simulation"}),
    "dexa":                 ("shepp","radon",10,{"src":"OAI DXA","ref":"oai.ucsf.edu"}),
    "angiography":          ("2d","psf",10,{"s":1.0,"src":"XCAD coronary","ref":"github.com/XiaoweiXu/XCAD"}),
    "portal_imaging":       ("2d","psf",10,{"s":2.0,"src":"AAPM TG-58 EPID","ref":"aapm.org"}),
    "brachytherapy_img":    ("2d","psf",10,{"s":2.0,"src":"AAPM TG-43 phantom","ref":"aapm.org"}),
    "proton_radiography":   ("shepp","radon",10,{"src":"pCT simulation","ref":"Simulation"}),
    "proton_therapy_img":   ("shepp","radon",10,{"src":"TOPAS MC simulation","ref":"Simulation"}),
    # ── MRI variants ──
    "asl_mri":              ("shepp","fourier",10,{"src":"ISMRM-OSIPI ASL","ref":"doi:10.1002/mrm.29224"}),
    "cest_mri":             ("shepp","fourier",10,{"src":"ISMRM 2024 CEST","ref":"ismrm.org"}),
    "diffusion_mri":        ("shepp","fourier",10,{"src":"HCP dMRI","ref":"doi:10.1016/j.neuroimage.2013.05.041"}),
    "fmri":                 ("shepp","fourier",10,{"src":"HCP fMRI","ref":"hcp.nmr.wustl.edu"}),
    "mr_elastography":      ("shepp","fourier",10,{"src":"RSNA QIBA MRE","ref":"rsna.org/qiba"}),
    "mr_fingerprinting":    ("shepp","fourier",10,{"src":"MRF (Ma Nature 2013)","ref":"doi:10.1038/nature11971"}),
    "mra":                  ("shepp","fourier",10,{"src":"IXI TOF-MRA","ref":"brain-development.org"}),
    "mrs":                  ("1d","identity",10,{"src":"MRSHUB benchmark","ref":"doi:10.1002/mrm.29478"}),
    "swi":                  ("shepp","fourier",10,{"src":"OpenNeuro SWI","ref":"doi:10.18112/openneuro.ds002778"}),
    "us_mri":               ("shepp","fourier",10,{"src":"PETRA/ZTE simulation","ref":"Simulation"}),
    # ── Ultrasound variants ──
    "doppler_ultrasound":   ("2d","psf",10,{"s":3.0,"src":"EchoNet-Dynamic","ref":"doi:10.1038/s41586-020-2145-8"}),
    "ceus":                 ("2d","psf",10,{"s":3.0,"src":"CAMUS cardiac US","ref":"doi:10.1109/TMI.2019.2900516"}),
    "ivus":                 ("2d","psf",10,{"s":2.0,"src":"MICCAI 2011 IVUS","ref":"miccai.org"}),
    "elastography":         ("2d","psf",10,{"s":2.0,"src":"RSNA QIBA MRE phantom","ref":"rsna.org/qiba"}),
    "ultrasonic_phased_array":("2d","psf",10,{"s":2.0,"src":"PAUT simulation","ref":"Simulation"}),
    # ── Optical microscopy ──
    "confocal_3d":          ("3d","psf",10,{"s":1.5,"src":"OpenCell 3D confocal","ref":"doi:10.1126/science.abi6983"}),
    "confocal_endomicroscopy":("2d","psf",10,{"s":1.5,"src":"CellvizioNet","ref":"Simulation"}),
    "confocal_livecell":    ("2d","psf",10,{"s":1.0,"src":"LiveCell","ref":"doi:10.1038/s41592-021-01249-6"}),
    "spinning_disk":        ("2d","psf",10,{"s":1.5,"src":"Broad BBBC","ref":"broadinstitute.org/bbbc"}),
    "lattice_lightsheet":   ("3d","psf",10,{"s":1.5,"src":"Allen Cell LLS","ref":"allencell.org"}),
    "lightsheet":           ("3d","psf",10,{"s":2.0,"src":"Allen Brain SPIM","ref":"alleninstitute.org"}),
    "two_photon":           ("2d","psf",10,{"s":1.0,"src":"Allen Brain 2P","ref":"doi:10.1016/j.neuron.2019.10.020"}),
    "three_photon":         ("3d","psf",10,{"s":1.5,"src":"Kleinfeld 3PM (UCSD)","ref":"doi:10.1126/science.1261605"}),
    "sted":                 ("2d","psf",10,{"s":0.5,"src":"STED benchmark","ref":"doi:10.1038/s41592-018-0023-6"}),
    "sim":                  ("2d","psf",10,{"s":1.0,"src":"SIMbench","ref":"doi:10.1038/s41592-018-0046-z"}),
    "palm_storm":           ("2d","psf",10,{"s":1.0,"src":"SMLM Challenge 2016","ref":"doi:10.1038/nmeth.4291"}),
    "tirf":                 ("2d","psf",10,{"s":1.0,"src":"SMLM Challenge TIRF","ref":"smlmchallenge.net"}),
    "dna_paint":            ("2d","psf",10,{"s":0.5,"src":"SMLM Challenge 2016","ref":"doi:10.1038/nmeth.4291"}),
    "widefield":            ("2d","psf",10,{"s":2.0,"src":"Broad BBBC / MitoCheck","ref":"broadinstitute.org/bbbc"}),
    "widefield_lowdose":    ("2d","psf",10,{"s":2.5,"src":"CARE low-dose","ref":"doi:10.1038/s41592-018-0216-7"}),
    "flim":                 ("2d","psf",10,{"s":1.5,"src":"FLUTE FLIM","ref":"doi:10.1038/s41592-019-0349-6"}),
    "expansion":            ("3d","psf",10,{"s":1.0,"src":"Allen ExM","ref":"alleninstitute.org"}),
    "ism":                  ("2d","psf",10,{"s":0.8,"src":"ISM simulation","ref":"Simulation"}),
    "minflux":              ("2d","psf",10,{"s":0.3,"src":"MINFLUX simulation","ref":"Simulation"}),
    "shg":                  ("2d","psf",10,{"s":1.0,"src":"SHG collagen","ref":"Simulation"}),
    "srs":                  ("spectral","psf",10,{"s":1.0,"src":"SRS spectral","ref":"Simulation"}),
    "cars":                 ("spectral","psf",10,{"s":1.0,"src":"CARS/SRS simulation","ref":"Simulation"}),
    # ── Electron microscopy ──
    "sem":                  ("2d","psf",10,{"s":0.5,"src":"NIST SEM calibration","ref":"nist.gov"}),
    "tem":                  ("2d","psf",10,{"s":0.5,"src":"EMPIAR TEM","ref":"ebi.ac.uk/empiar"}),
    "stem":                 ("2d","psf",10,{"s":0.5,"src":"EMPIAR STEM","ref":"ebi.ac.uk/empiar"}),
    "cryo_et":              ("3d","proj3d",10,{"src":"SHREC 2021 cryo-ET","ref":"shrec.cs.uu.nl/2021"}),
    "electron_tomography":  ("3d","proj3d",10,{"src":"EMPIAR-10005","ref":"ebi.ac.uk/empiar"}),
    "fib_sem":              ("3d","psf",10,{"s":0.5,"src":"OpenOrganelle FIB-SEM","ref":"openorganelle.janelia.org"}),
    "electron_diffraction": ("2d","fourier",10,{"src":"RRUFF+ICSD CIF","ref":"rruff.info"}),
    "electron_holography":  ("2d","identity",10,{"src":"FZJ electron holography","ref":"Simulation"}),
    "cathodoluminescence":  ("spectral","psf",10,{"s":1.0,"src":"HyperSpy CL","ref":"doi:10.5281/zenodo.6513794"}),
    "eels":                 ("1d","identity",10,{"src":"EELS.info public DB","ref":"eels.info"}),
    "edx_mapping":          ("spectral","psf",10,{"s":1.0,"src":"HyperSpy EDX","ref":"doi:10.5281/zenodo.3257834"}),
    "ebsd":                 ("2d","identity",10,{"src":"DREAM.3D EBSD","ref":"dream3d.io"}),
    "sims":                 ("spectral","identity",10,{"src":"SIMS surface DB","ref":"Simulation"}),
    "clem":                 ("2d","psf",10,{"s":1.5,"src":"EMPIAR-10094 CLEM","ref":"ebi.ac.uk/empiar/EMPIAR-10094"}),
    # ── Scanning probe ──
    "afm":                  ("2d","psf",10,{"s":0.5,"src":"QUAM-AFM dataset","ref":"doi:10.1021/acs.jcim.1c01323"}),
    "stm":                  ("2d","psf",10,{"s":0.3,"src":"NIST surface SRM","ref":"doi:10.1088/0957-4484"}),
    "mfm":                  ("2d","psf",10,{"s":0.5,"src":"MFM simulation","ref":"Simulation"}),
    "nsom":                 ("2d","psf",10,{"s":0.3,"src":"NSOM simulation","ref":"Simulation"}),
    # ── Spectroscopy ──
    "raman_imaging":        ("spectral","psf",10,{"s":1.0,"src":"RRUFF Raman DB","ref":"doi:10.2138/am.2006.2168"}),
    "ftir_imaging":         ("spectral","psf",10,{"s":1.5,"src":"USGS Spectral Lib v7","ref":"doi:10.3133/ds1035"}),
    "libs":                 ("1d","identity",10,{"src":"NIST LIBS database","ref":"nist.gov"}),
    "brillouin":            ("1d","identity",10,{"src":"RRUFF Brillouin DB","ref":"Simulation"}),
    "terahertz":            ("1d","identity",10,{"src":"NIST THz spectroscopy","ref":"nist.gov"}),
    "maldi_msi":            ("spectral","identity",10,{"src":"MetaboLights MALDI MSI","ref":"ebi.ac.uk/metabolights"}),
    "desi":                 ("spectral","identity",10,{"src":"MetaboLights DESI-MSI","ref":"ebi.ac.uk/metabolights"}),
    # ── Remote sensing ──
    "sar":                  ("2d","fourier",10,{"src":"Sentinel-1 GRD","ref":"sentinel.esa.int"}),
    "insar":                ("2d","fourier",10,{"src":"Sentinel-1 SLC","ref":"esa.int/copernicus"}),
    "polsar":               ("2d","fourier",10,{"src":"UAVSAR (NASA JPL)","ref":"uavsar.jpl.nasa.gov"}),
    "multispectral_sat":    ("spectral","downsample",10,{"src":"Sentinel-2 L2A","ref":"sentinel.esa.int"}),
    "ocean_color":          ("spectral","downsample",10,{"src":"NASA MODIS L3","ref":"oceancolor.gsfc.nasa.gov"}),
    "weather_radar":        ("2d","psf",10,{"s":2.0,"src":"NEXRAD WSR-88D","ref":"doi:10.1175/BAMS-88-3-313"}),
    "passive_microwave":    ("spectral","downsample",10,{"src":"AMSR2 L3 Tb","ref":"nsidc.org"}),
    # ── Optical / computational ──
    "hdr_imaging":          ("2d","identity",10,{"src":"Fairchild HDR-DB","ref":"doi:10.2352/issn.2169-2629"}),
    "light_field":          ("2d","identity",10,{"src":"Stanford LF Archive","ref":"lightfield.stanford.edu"}),
    "integral":             ("2d","identity",10,{"src":"Stanford LF Archive","ref":"lightfield.stanford.edu"}),
    "event_camera":         ("2d","identity",10,{"src":"DAVIS 240C / MVSEC","ref":"doi:10.1109/LRA.2018.2800793"}),
    "flash_lidar":          ("2d","identity",10,{"src":"KITTI LiDAR","ref":"doi:10.1109/CVPR.2012.6248074"}),
    "lidar":                ("2d","identity",10,{"src":"KITTI LiDAR","ref":"doi:10.1109/CVPR.2012.6248074"}),
    "tof_camera":           ("2d","identity",10,{"src":"ETH3D ToF","ref":"doi:10.1109/CVPR.2017.272"}),
    "structured_light":     ("2d","identity",10,{"src":"CAVE structured light","ref":"doi:10.1109/CVPR.2012.6248026"}),
    "photometric_stereo":   ("2d","psf",10,{"s":1.0,"src":"DiLiGenT benchmark","ref":"doi:10.1109/TPAMI.2015.2457918"}),
    "polarization":         ("2d","identity",10,{"src":"AOLP/DAVIS polarization","ref":"Simulation"}),
    "lucky_imaging":        ("2d","psf",10,{"s":3.0,"src":"Palomar speckle","ref":"Simulation"}),
    "gaussian_splatting":   ("2d","identity",10,{"src":"Tanks & Temples","ref":"doi:10.1145/3072959.3073599"}),
    "ghost_imaging":        ("2d","mask",10,{"src":"Ghost imaging sim","ref":"Simulation"}),
    "fpm":                  ("2d","fourier",10,{"src":"UCB FPM benchmark","ref":"doi:10.1038/lsa.2015.140"}),
    "coronagraphy":         ("2d","psf",10,{"s":2.0,"src":"HST MAST archive","ref":"mast.stsci.edu"}),
    "adaptive_optics":      ("2d","psf",10,{"s":3.0,"src":"ESO VLT SPHERE AO","ref":"doi:10.1051/0004-6361/201730834"}),
    "dark_field":           ("shepp","radon",10,{"src":"Munich Talbot-Lau","ref":"Simulation"}),
    "talbot_lau":           ("shepp","radon",10,{"src":"TU Munich Talbot-Lau","ref":"Simulation"}),
    "dic":                  ("2d","psf",10,{"s":1.0,"src":"DIC Challenge","ref":"dic-challenge.epfl.ch"}),
    "phase_contrast":       ("2d","fourier",10,{"src":"APS Argonne","ref":"doi:10.1107/S2059798320008918"}),
    "phase_retrieval":      ("2d","fourier",10,{"src":"CDI ptychography (Zenodo)","ref":"doi:10.5281/zenodo.7671177"}),
    "ptychography":         ("2d","fourier",10,{"src":"CDI ptychography (Zenodo)","ref":"doi:10.5281/zenodo.7671177"}),
    # ── Wave / acoustic ──
    "acoustic_emission":    ("1d","identity",10,{"src":"EWGAE AE benchmark","ref":"Simulation"}),
    "acoustic_microscopy":  ("2d","psf",10,{"s":1.5,"src":"SAM synthetic","ref":"Simulation"}),
    "sonar":                ("2d","psf",10,{"s":3.0,"src":"NOAA multibeam sonar","ref":"Simulation"}),
    "ocean_acoustic_tomo":  ("2d","radon",10,{"src":"Ocean acoustic sim","ref":"Simulation"}),
    "gpr":                  ("2d","radon",10,{"src":"GPR simulation","ref":"Simulation"}),
    # ── Nuclear / particle / X-ray ──
    "spect":                ("shepp","radon",10,{"src":"SIMIND MC SPECT","ref":"simind.com"}),
    "spect_ct":             ("shepp","radon",10,{"src":"TCIA SPECT-CT","ref":"cancerimagingarchive.net"}),
    "pet_ct":               ("shepp","radon",10,{"src":"TCIA PET-CT","ref":"doi:10.1016/j.radonc.2020.01.033"}),
    "pet_mr":               ("shepp","radon",10,{"src":"ADNI PET-MRI","ref":"adni.loni.usc.edu"}),
    "muon_tomo":            ("shepp","radon",10,{"src":"CERN muon tomo sim","ref":"Simulation"}),
    "neutron_diffraction":  ("1d","fourier",10,{"src":"ILL neutron archive","ref":"ill.eu"}),
    "neutron_tomo":         ("shepp","radon",10,{"src":"PSI NEUTRA dataset","ref":"psi.ch/en/num/neutra"}),
    "xfel_sfx":             ("2d","fourier",10,{"src":"LCLS SFX archive","ref":"lcls.slac.stanford.edu"}),
    "xray_crystallography": ("2d","fourier",10,{"src":"PDB","ref":"doi:10.1093/nar/gky1049"}),
    "saxs":                 ("2d","fourier",10,{"src":"cSAXS synchrotron (PSI)","ref":"psi.ch/en/sls/csaxs"}),
    "waxs":                 ("2d","fourier",10,{"src":"ESRF WAXS archive","ref":"doi:10.1107/S1600576714015283"}),
    "xrf_imaging":          ("spectral","identity",10,{"src":"ESRF XRF imaging","ref":"esrf.eu"}),
    "xrf_tomo":             ("spectral","radon",10,{"src":"ESRF XRF tomo","ref":"esrf.eu"}),
    "ct_fluorescence":      ("3d","radon",10,{"src":"CT-FMT simulation","ref":"Simulation"}),
    # ── Other ──
    "dot":                  ("3d","radon",10,{"src":"UCL DOT simulation","ref":"Simulation"}),
    "impedance_tomo":       ("2d","identity",10,{"src":"EIDORS simulation","ref":"doi:10.1088/0967-3334/27/5/S02"}),
    "active_thermography":  ("2d","psf",10,{"s":3.0,"src":"PVC-Infrared Dataset","ref":"doi:10.3390/app13052901"}),
    "eddy_current":         ("2d","psf",10,{"s":2.0,"src":"EEDB NDT benchmark","ref":"Simulation"}),
    "shearography":         ("2d","psf",10,{"s":2.0,"src":"Shearography sim","ref":"Simulation"}),
    "bioluminescence_tomo": ("3d","radon",10,{"src":"BLT simulation","ref":"Simulation"}),
    "cup":                  ("2d","mask",10,{"src":"CUP simulation","ref":"Simulation"}),
    "streak_camera":        ("2d","mask",10,{"src":"Streak camera sim","ref":"Simulation"}),
    "pump_probe":           ("2d","identity",10,{"src":"SLAC LCLS","ref":"Simulation"}),
    "entangled_photon":     ("2d","mask",10,{"src":"Quantum imaging sim","ref":"Simulation"}),
    "quantum_illumination": ("2d","mask",10,{"src":"Quantum illumination sim","ref":"Simulation"}),
    "magnetic_particle":    ("3d","fourier",10,{"src":"OpenMPIData","ref":"doi:10.1002/mrm.26596"}),
    "spc":                  ("2d","mask",10,{"src":"SPC random matrix","ref":"Simulation"}),
    "odt":                  ("2d","fourier",10,{"src":"Toulouse ODT/TORCH","ref":"Simulation"}),
    "octa":                 ("2d","psf",10,{"s":1.0,"src":"ROSE OCTA","ref":"doi:10.1109/TPAMI.2021.3093584"}),
    "nirs_brain":           ("1d","identity",10,{"src":"fNIRS-BIDS","ref":"fnirs-bids.readthedocs.io"}),
    "radio_interferometry": ("2d","fourier",10,{"src":"VLBI imaging challenge 2022","ref":"radiointerferometrychallenge.github.io"}),
    "atom_probe":           ("3d","identity",10,{"src":"APT simulation","ref":"Simulation"}),
}


def get_shape(modality, gt_type):
    """Get shape from YAML config, falling back to sensible defaults."""
    cfg_path = CFG_DIR / f"{modality}.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        xs = cfg.get("x_shape")
        if isinstance(xs, list):
            # Replace string tokens with ints
            clean = []
            for v in xs:
                if isinstance(v, int):
                    clean.append(min(v, 256))
                else:
                    clean.append(16)  # default for T, C, L, S, E tokens
            return tuple(clean)
    defaults = {
        "2d": (256, 256), "shepp": (256, 256),
        "3d": (32, 64, 64), "spectral": (64, 64, 16), "1d": (1024,),
    }
    return defaults.get(gt_type, (256, 256))


def gen_gt(gt_type, shape, seed):
    if gt_type == "shepp":
        return shepp_logan(shape[0], shape[1] if len(shape) > 1 else shape[0], seed)
    elif gt_type == "2d":
        return smooth_2d(shape[0], shape[1] if len(shape) > 1 else shape[0], seed)
    elif gt_type == "3d":
        return smooth_3d(*shape[:3], seed=seed) if len(shape) >= 3 else smooth_2d(shape[0], shape[1], seed)
    elif gt_type == "spectral":
        if len(shape) >= 3:
            return spectral_cube(shape[0], shape[1], shape[2], seed)
        return smooth_2d(shape[0], shape[1] if len(shape) > 1 else shape[0], seed)
    elif gt_type == "1d":
        return signal_1d(shape[0], seed)
    return smooth_2d(shape[0], shape[1] if len(shape) > 1 else shape[0], seed)


def apply_fwd(fwd_type, x, params, seed):
    sigma = params.get("s", 2.0)
    if fwd_type == "radon":
        if x.ndim == 2:
            return radon_fwd(x, 120)
        elif x.ndim == 3:
            out = []
            for s in range(x.shape[0]):
                out.append(radon_fwd(x[s], 60))
            return np.stack(out, axis=0)
        return radon_fwd(x.reshape(x.shape[0], -1), 120)
    elif fwd_type == "fourier":
        if x.ndim >= 2:
            img = x if x.ndim == 2 else x[:, :, 0]
            return fourier_fwd(img)
        k = np.fft.fft(x)
        return np.stack([k.real, k.imag], axis=-1).astype(np.float32)
    elif fwd_type == "psf":
        return psf_fwd(x, sigma)
    elif fwd_type == "downsample":
        return downsample_fwd(x)
    elif fwd_type == "mask":
        return mask_fwd(x, 0.3, seed)
    elif fwd_type == "proj3d":
        if x.ndim == 3:
            return proj3d_fwd(x, 60)
        return identity_fwd(x)
    return identity_fwd(x)


def build_one(modality, gt_type, fwd_type, n, params):
    out_dir = DATA_DIR / modality / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in out_dir.iterdir() if f.suffix == ".h5"]
    if len(existing) >= n:
        return "skip"

    shape = get_shape(modality, gt_type)
    src = params.get("src", "Simulation")
    ref = params.get("ref", "Simulation")

    for i in range(n):
        seed = 42 + i * 137
        x = gen_gt(gt_type, shape, seed)
        try:
            y = apply_fwd(fwd_type, x, params, seed)
        except Exception as e:
            print(f"    fwd error {modality}[{i}]: {e}")
            y = identity_fwd(x)

        h5 = out_dir / f"standard_{modality}_{i:02d}.h5"
        with h5py.File(str(h5), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip", compression_opts=4)
            f.create_dataset("y_ideal", data=y, compression="gzip", compression_opts=4)
            hp = f.create_group("H_params")
            hp.attrs["forward_type"] = fwd_type
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"
            md = f.create_group("metadata")
            md.attrs["source"] = src
            md.attrs["reference"] = ref
            md.attrs["sample_index"] = i
            md.attrs["date_built"] = "2026-03-13"
            md.attrs["synthetic"] = True

    meta = {
        "modality": modality,
        "canonical_dataset": src,
        "source_type": "synthetic",
        "reference": ref,
        "n_samples": n,
        "x_true_shape": list(x.shape),
        "y_ideal_shape": list(y.shape),
        "forward_type": fwd_type,
        "noise": "none", "mismatch": "none",
        "date_built": "2026-03-13",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/",
        "status": "built-synthetic",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({"modality": modality, "forward_type": fwd_type,
                    "noise": "none", "mismatch": "none", "split": "standard"}, f, indent=2)
    return "ok"


def main():
    total = len(MODS)
    print(f"Building standard datasets for {total} remaining modalities...\n")
    ok = skip = fail = 0
    for idx, (mod, (gt, fwd, n, params)) in enumerate(sorted(MODS.items()), 1):
        try:
            status = build_one(mod, gt, fwd, n, params)
            if status == "skip":
                skip += 1
                print(f"  [{idx}/{total}] {mod}: already built, skipped")
            else:
                ok += 1
                print(f"  [{idx}/{total}] {mod}: built {n} samples")
        except Exception as e:
            fail += 1
            print(f"  [{idx}/{total}] {mod}: FAILED — {e}")
    print(f"\n{'='*60}")
    print(f"New: {ok} | Skipped: {skip} | Failed: {fail} | Total: {ok+skip+29}/168")


if __name__ == "__main__":
    main()
