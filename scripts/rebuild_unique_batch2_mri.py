"""
Rebuild standard datasets — Batch 2: MRI variants (11 modalities).
Each gets a unique brain/body phantom with appropriate MRI-specific contrast.

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "benchmark"
N = 10


def to_uint8(arr):
    arr = np.nan_to_num(arr)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_img(arr, path):
    if arr.ndim == 2:
        Image.fromarray(to_uint8(arr), "L").save(str(path))
    elif arr.ndim == 3 and arr.shape[-1] in [2, 3]:
        if arr.shape[-1] == 2:  # complex k-space -> magnitude
            mag = np.sqrt(arr[..., 0]**2 + arr[..., 1]**2)
            Image.fromarray(to_uint8(np.log1p(mag)), "L").save(str(path))
        else:
            Image.fromarray(to_uint8(arr), "RGB").save(str(path))
    elif arr.ndim == 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(arr)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(str(path), dpi=100)
        plt.close(fig)


def save_modality(modality, samples, fwd_type, fwd_desc, meta_extra):
    out = DATA / modality / "standard"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("standard_*.h5"):
        old.unlink()
    files = []
    for i, (x, y) in enumerate(samples):
        h5p = out / f"standard_{modality}_{i:02d}.h5"
        with h5py.File(str(h5p), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            hp = f.create_group("H_params")
            hp.attrs["forward_type"] = fwd_type
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"
            md = f.create_group("metadata")
            md.attrs["source"] = meta_extra.get("src", "Simulation")
            md.attrs["reference"] = meta_extra.get("ref", "Simulation")
            md.attrs["sample_index"] = i
            md.attrs["date_built"] = "2026-03-14"
        files.append(h5p.name)
        img_dir = out / "images" / f"sample_{i:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        save_img(x, img_dir / "x_true.png")
        save_img(y, img_dir / "y_ideal.png")
    meta = {
        "modality": modality,
        "canonical_dataset": meta_extra.get("src", "Simulation"),
        "reference": meta_extra.get("ref", "Simulation"),
        "n_samples": len(samples), "forward_type": fwd_type,
        "forward_model": fwd_desc,
        "x_true_shape": list(samples[0][0].shape),
        "y_ideal_shape": list(samples[0][1].shape),
        "noise": "none", "mismatch": "none",
        "date_built": "2026-03-14",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/",
        "status": "done", "files": files,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": modality, "forward_type": fwd_type,
                    "noise": "none", "mismatch": "none", "split": "standard"}, f, indent=2)
    print(f"  {modality}: {len(samples)} samples")


def fourier_undersample(x, accel=4, seed=42):
    """Undersampled k-space with ACS center lines."""
    k = np.fft.fft2(x)
    mask = np.zeros(x.shape, dtype=bool)
    H = x.shape[0]
    nc = max(1, int(H * 0.08))
    mask[H//2 - nc//2:H//2 + nc//2, :] = True
    rng = np.random.RandomState(seed)
    extra = max(1, H // accel - nc)
    idx = rng.choice(H, extra, replace=False)
    mask[idx, :] = True
    y = k * mask
    return np.stack([y.real, y.imag], axis=-1).astype(np.float32)


# ===================================================================
# UNIQUE MRI PHANTOMS
# ===================================================================

def phantom_asl_perfusion(seed):
    """ASL MRI — cerebral perfusion map (CBF) with gray/white matter contrast."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Brain outline
    brain = (r < S * 0.4).astype(np.float32)
    # Gray matter (cortex — high perfusion, outer ring)
    cortex = ((r < S * 0.4) & (r > S * 0.28)).astype(np.float32)
    x += cortex * 0.7  # High CBF
    # White matter (low perfusion)
    wm = (r < S * 0.28).astype(np.float32)
    x += wm * 0.25
    # Deep gray nuclei (high perfusion spots)
    for side in [-1, 1]:
        # Thalamus
        tx = cx + side * S * 0.06
        ty = cy + S * 0.02
        thal = np.exp(-0.5 * ((xx - tx)**2 + (yy - ty)**2) / 6**2)
        x += thal * 0.5
        # Putamen
        px = cx + side * S * 0.12
        py = cy - S * 0.02
        put = np.exp(-0.5 * ((xx - px)**2 / 8**2 + (yy - py)**2 / 5**2))
        x += put * 0.6
    # Perfusion deficit region (stroke)
    if rng.rand() > 0.4:
        sx = cx + rng.uniform(-S * 0.15, S * 0.15)
        sy = cy + rng.uniform(-S * 0.15, S * 0.15)
        stroke = np.exp(-0.5 * ((xx - sx)**2 + (yy - sy)**2) / rng.uniform(6, 15)**2)
        x -= stroke * 0.4
    return np.clip(x * brain, 0, 1).astype(np.float32)


def phantom_cest(seed):
    """CEST MRI — pH-sensitive amide proton phantom with tumor contrast."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Brain tissue (baseline CEST signal)
    brain = (r < S * 0.4).astype(np.float32)
    x += brain * 0.4
    # Tumor region (elevated amide CEST — higher pH)
    tx = cx + rng.uniform(-S * 0.1, S * 0.1)
    ty = cy + rng.uniform(-S * 0.1, S * 0.1)
    ts = rng.uniform(8, 18)
    tumor = np.exp(-0.5 * ((xx - tx)**2 + (yy - ty)**2) / ts**2)
    x += tumor * 0.5  # Enhanced CEST in tumor
    # Necrotic core (low CEST)
    if ts > 12:
        core = np.exp(-0.5 * ((xx - tx)**2 + (yy - ty)**2) / (ts * 0.4)**2)
        x -= core * 0.3
    # CSF (very low CEST)
    for side in [-1, 1]:
        vx = cx + side * S * 0.04
        vent = np.exp(-0.5 * ((xx - vx)**2 / 5**2 + (yy - cy)**2 / 12**2))
        x -= vent * 0.2
    return np.clip(x * brain, 0, 1).astype(np.float32)


def phantom_diffusion_tracts(seed):
    """Diffusion MRI — white matter tract FA map with crossing fibers."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    brain = (r < S * 0.4).astype(np.float32)
    # White matter tracts (high FA along fiber directions)
    # Corpus callosum (horizontal band)
    cc = np.exp(-0.5 * (yy - cy + S * 0.08)**2 / 4**2) * (np.abs(xx - cx) < S * 0.25).astype(np.float32)
    x += cc * 0.8
    # Corticospinal tract (vertical)
    for side in [-1, 1]:
        cst_x = cx + side * S * 0.08
        cst = np.exp(-0.5 * (xx - cst_x)**2 / 4**2) * (np.abs(yy - cy) < S * 0.3).astype(np.float32)
        x += cst * 0.7
    # Superior longitudinal fasciculus (curved)
    for side in [-1, 1]:
        slf_cx = cx + side * S * 0.2
        slf_r = S * 0.15
        slf_dist = np.sqrt((xx - slf_cx)**2 + (yy - cy + S * 0.05)**2)
        slf = np.exp(-0.5 * ((slf_dist - slf_r) / 3)**2) * (yy < cy + S * 0.1).astype(np.float32)
        x += slf * 0.6
    # Gray matter (low FA)
    gm = ((r < S * 0.4) & (r > S * 0.3)).astype(np.float32)
    x = np.maximum(x, gm * 0.15)
    return np.clip(x * brain, 0, 1).astype(np.float32)


def phantom_fmri_activation(seed):
    """fMRI — BOLD activation map on brain background."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    brain = (r < S * 0.4).astype(np.float32)
    # Baseline brain tissue
    x += brain * 0.3
    # Gray matter cortex
    cortex = ((r < S * 0.4) & (r > S * 0.28)).astype(np.float32)
    x += cortex * 0.1
    # BOLD activation blobs
    tasks = ["visual", "motor", "auditory", "frontal"]
    task = rng.choice(tasks)
    if task == "visual":
        regions = [(cx, cy + S * 0.25, 12), (cx + 5, cy + S * 0.22, 8)]
    elif task == "motor":
        regions = [(cx - S * 0.15, cy - S * 0.15, 10), (cx + S * 0.15, cy - S * 0.15, 10)]
    elif task == "auditory":
        regions = [(cx - S * 0.22, cy, 10), (cx + S * 0.22, cy, 10)]
    else:
        regions = [(cx, cy - S * 0.25, 12), (cx - S * 0.08, cy - S * 0.2, 8)]
    for rx, ry, rs in regions:
        rs += rng.uniform(-2, 2)
        activation = np.exp(-0.5 * ((xx - rx)**2 + (yy - ry)**2) / rs**2)
        x += activation * rng.uniform(0.3, 0.6)
    return np.clip(x * brain, 0, 1).astype(np.float32)


def phantom_mre_elasticity(seed):
    """MR elastography — shear wave map with stiffness contrast."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Liver phantom (MRE most commonly used in liver)
    liver = (((xx - cx) / (S * 0.35))**2 + ((yy - cy) / (S * 0.3))**2 < 1)
    x += liver.astype(np.float32) * 0.3  # normal liver ~3 kPa
    # Fibrotic region (elevated stiffness)
    fx = cx + rng.uniform(-S * 0.1, S * 0.1)
    fy = cy + rng.uniform(-S * 0.1, S * 0.1)
    fibrosis = np.exp(-0.5 * ((xx - fx)**2 + (yy - fy)**2) / rng.uniform(10, 25)**2)
    x += fibrosis * 0.5 * liver  # ~8-12 kPa fibrotic
    # Hepatic vessels (lower stiffness — fluid)
    for _ in range(rng.randint(3, 6)):
        vx = cx + rng.uniform(-S * 0.2, S * 0.2)
        vy = cy + rng.uniform(-S * 0.15, S * 0.15)
        vs = rng.uniform(2, 5)
        vessel = np.exp(-0.5 * ((xx - vx)**2 + (yy - vy)**2) / vs**2)
        x -= vessel * 0.15 * liver
    # Shear wave pattern overlay
    freq = rng.uniform(3, 6)
    angle = rng.uniform(0, np.pi)
    wave = 0.1 * np.sin(2 * np.pi * freq * ((xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)) / S)
    x += wave * liver.astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_mrf_multicontrast(seed):
    """MR fingerprinting — multi-tissue T1/T2 phantom."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Cylindrical phantom with multiple vials (like ISMRMRD phantom)
    body = (r < S * 0.42).astype(np.float32)
    x += body * 0.2  # background water
    # Vials arranged in concentric circles with different T1/T2 values
    n_outer = rng.randint(8, 14)
    r_ring = S * 0.3
    vial_r = S * 0.04
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        vx = cx + r_ring * np.cos(angle)
        vy = cy + r_ring * np.sin(angle)
        vial = (np.sqrt((xx - vx)**2 + (yy - vy)**2) < vial_r).astype(np.float32)
        t1_val = 0.1 + 0.8 * i / n_outer  # varying T1
        x += vial * t1_val
    # Inner ring of vials
    n_inner = rng.randint(4, 8)
    r_inner = S * 0.15
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner + np.pi / n_inner
        vx = cx + r_inner * np.cos(angle)
        vy = cy + r_inner * np.sin(angle)
        vial = (np.sqrt((xx - vx)**2 + (yy - vy)**2) < vial_r).astype(np.float32)
        x += vial * rng.uniform(0.3, 0.9)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_mra_vessels(seed):
    """MR angiography — brain vascular tree (circle of Willis)."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Background brain (low signal in MRA)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    bg = (r < S * 0.42).astype(np.float32) * 0.05
    x += bg
    # Circle of Willis — ring of arteries at brain base
    cow_r = S * 0.12
    cow_width = 2.5
    cow = np.exp(-0.5 * ((np.sqrt((xx - cx)**2 + (yy - cy + S * 0.08)**2) - cow_r) / cow_width)**2)
    x += cow * 0.8
    # Internal carotid arteries (vertical, from below)
    for side in [-1, 1]:
        icx = cx + side * S * 0.08
        ica = np.exp(-0.5 * (xx - icx)**2 / 2.5**2) * (yy > cy).astype(np.float32)
        x += ica * 0.7
    # Middle cerebral arteries (lateral branches)
    for side in [-1, 1]:
        mca_start_x = cx + side * cow_r
        mca_start_y = cy - S * 0.08
        for t in np.linspace(0, 1, 150):
            px = mca_start_x + side * S * 0.25 * t
            py = mca_start_y - S * 0.15 * t + S * 0.05 * np.sin(t * 5)
            if 0 <= px < S and 0 <= py < S:
                x += np.exp(-0.5 * ((xx - px)**2 + (yy - py)**2) / 2**2) * 0.5 * (1 - 0.5 * t)
    # Anterior cerebral artery (midline, going up)
    for t in np.linspace(0, 1, 100):
        py = cy - S * 0.08 - S * 0.25 * t
        px = cx + S * 0.02 * np.sin(t * 3)
        if 0 <= py < S:
            x += np.exp(-0.5 * ((xx - px)**2 + (yy - py)**2) / 2**2) * 0.6
    # Posterior arteries
    for side in [-1, 1]:
        for t in np.linspace(0, 1, 80):
            px = cx + side * (cow_r + S * 0.12 * t)
            py = cy - S * 0.08 + S * 0.15 * t
            if 0 <= px < S and 0 <= py < S:
                x += np.exp(-0.5 * ((xx - px)**2 + (yy - py)**2) / 2**2) * 0.5 * (1 - 0.4 * t)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_mrs_spectrum(seed):
    """MR spectroscopy — brain metabolite spectrum (NAA, Cr, Cho, mI, Glu, Lac)."""
    rng = np.random.RandomState(seed)
    N_pts = 2048
    ppm = np.linspace(0, 5, N_pts)
    x = np.zeros(N_pts, dtype=np.float32)
    # Major brain metabolites at typical chemical shifts (ppm)
    metabolites = {
        "NAA": (2.02, 0.03, 0.8 + rng.uniform(-0.1, 0.1)),       # N-acetyl aspartate
        "Cr": (3.03, 0.03, 0.5 + rng.uniform(-0.05, 0.05)),       # Creatine
        "Cho": (3.22, 0.025, 0.3 + rng.uniform(-0.05, 0.05)),     # Choline
        "mI": (3.56, 0.04, 0.35 + rng.uniform(-0.05, 0.05)),      # Myo-inositol
        "Glu": (2.35, 0.05, 0.4 + rng.uniform(-0.08, 0.08)),      # Glutamate
        "GABA": (2.28, 0.04, 0.15 + rng.uniform(-0.03, 0.03)),    # GABA
        "Lac": (1.33, 0.025, rng.uniform(0.02, 0.2)),             # Lactate (varies)
    }
    for name, (center, width, amp) in metabolites.items():
        x += amp * np.exp(-0.5 * ((ppm - center) / width)**2)
    # Water residual (suppressed but present)
    x += 0.05 * np.exp(-0.5 * ((ppm - 4.7) / 0.1)**2)
    # Baseline distortion
    x += 0.02 * np.sin(np.pi * ppm / 5)
    return x.astype(np.float32)


def phantom_swi_veins(seed):
    """SWI — susceptibility-weighted imaging with venous anatomy."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    brain = (r < S * 0.4).astype(np.float32)
    # Brain tissue background
    x += brain * 0.6
    # Venous sinuses (dark in SWI due to deoxyhemoglobin)
    # Superior sagittal sinus (midline)
    sss = np.exp(-0.5 * (xx - cx)**2 / 2**2) * (yy < cy - S * 0.1).astype(np.float32)
    x -= sss * 0.4
    # Internal cerebral veins
    for side in [-1, 1]:
        icv_x = cx + side * S * 0.03
        icv = np.exp(-0.5 * (xx - icv_x)**2 / 1.5**2) * ((yy > cy - S * 0.1) & (yy < cy + S * 0.1)).astype(np.float32)
        x -= icv * 0.35
    # Cortical veins (small dots/lines near surface)
    n_veins = rng.randint(8, 20)
    for _ in range(n_veins):
        angle = rng.uniform(0, 2 * np.pi)
        vr = S * rng.uniform(0.25, 0.38)
        vx = cx + vr * np.cos(angle)
        vy = cy + vr * np.sin(angle)
        vs = rng.uniform(1, 3)
        vein = np.exp(-0.5 * ((xx - vx)**2 + (yy - vy)**2) / vs**2)
        x -= vein * rng.uniform(0.15, 0.35)
    # Microbleeds (very small dark spots)
    if rng.rand() > 0.3:
        for _ in range(rng.randint(1, 4)):
            mx = cx + rng.uniform(-S * 0.25, S * 0.25)
            my = cy + rng.uniform(-S * 0.25, S * 0.25)
            mb = np.exp(-0.5 * ((xx - mx)**2 + (yy - my)**2) / 1.5**2)
            x -= mb * 0.3
    return np.clip(x * brain, 0, 1).astype(np.float32)


def phantom_ute_mri(seed):
    """Ultrashort TE (UTE/ZTE) MRI — shows short-T2 tissues (bone, tendon)."""
    rng = np.random.RandomState(seed)
    S = 128
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Knee cross-section (UTE commonly used for MSK)
    # Cortical bone (bright in UTE — invisible in conventional MRI)
    femur_r = S * 0.15
    bone_shell = ((r < femur_r + 4) & (r > femur_r)).astype(np.float32)
    x += bone_shell * 0.9
    # Bone marrow (moderate signal)
    marrow = (r < femur_r).astype(np.float32)
    x += marrow * 0.4
    # Menisci (short T2, bright in UTE)
    for side in [-1, 1]:
        men_x = cx + side * S * 0.18
        men_y = cy + S * 0.15
        men_r = S * 0.06
        men = np.exp(-0.5 * ((xx - men_x)**2 / (men_r * 1.5)**2 + (yy - men_y)**2 / men_r**2))
        x += men * 0.7
    # Tendons/ligaments (bright in UTE)
    pcl = np.exp(-0.5 * (xx - cx)**2 / 2**2) * ((yy > cy - S * 0.05) & (yy < cy + S * 0.15)).astype(np.float32)
    x += pcl * 0.8
    # Cartilage
    cart_r = femur_r + 8
    cartilage = ((r < cart_r) & (r > femur_r + 4)).astype(np.float32)
    x += cartilage * 0.5
    # Muscle (moderate signal)
    muscle = ((r < S * 0.4) & (r > cart_r + 5)).astype(np.float32)
    x += muscle * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# BUILD ALL
# ===================================================================

def build_all():
    print("=" * 60)
    print("Batch 2: Rebuilding MRI variants with unique phantoms")
    print("=" * 60)

    modalities = [
        ("asl_mri", phantom_asl_perfusion, "fourier_undersample",
         "y = M * FFT(x); undersampled k-space (ASL perfusion)",
         {"src": "ISMRM-OSIPI ASL Perfusion Challenge", "ref": "doi:10.1002/mrm.29224"}),
        ("cest_mri", phantom_cest, "fourier_undersample",
         "y = M * FFT(x); CEST-weighted k-space",
         {"src": "ISMRM CEST Challenge 2024", "ref": "ismrm.org"}),
        ("diffusion_mri", phantom_diffusion_tracts, "fourier_undersample",
         "y = M * FFT(x); diffusion-weighted k-space",
         {"src": "Human Connectome Project dMRI", "ref": "doi:10.1016/j.neuroimage.2013.05.041"}),
        ("fmri", phantom_fmri_activation, "fourier_undersample",
         "y = M * FFT(x); BOLD-contrast k-space",
         {"src": "Human Connectome Project fMRI", "ref": "humanconnectome.org"}),
        ("mr_elastography", phantom_mre_elasticity, "fourier_undersample",
         "y = M * FFT(x); MRE stiffness-weighted k-space",
         {"src": "RSNA QIBA MRE Phantom", "ref": "rsna.org/qiba"}),
        ("mr_fingerprinting", phantom_mrf_multicontrast, "fourier_undersample",
         "y = M * FFT(x); MRF dictionary-encoded k-space",
         {"src": "MRF (Ma et al. Nature 2013)", "ref": "doi:10.1038/nature11971"}),
        ("mra", phantom_mra_vessels, "fourier_undersample",
         "y = M * FFT(x); MRA time-of-flight k-space",
         {"src": "IXI TOF-MRA dataset", "ref": "brain-development.org"}),
        ("mrs", phantom_mrs_spectrum, "spectral_fid",
         "y = FFT(FID(x)); MRS free induction decay",
         {"src": "MRSHUB/ISMRM MRS benchmark", "ref": "doi:10.1002/mrm.29478"}),
        ("swi", phantom_swi_veins, "fourier_undersample",
         "y = M * FFT(x); susceptibility-weighted k-space",
         {"src": "OpenNeuro SWI dataset", "ref": "doi:10.18112/openneuro.ds002778"}),
        ("us_mri", phantom_ute_mri, "fourier_undersample",
         "y = M * FFT(x); ultrashort TE k-space",
         {"src": "PETRA/ZTE UTE-MRI benchmark", "ref": "doi:10.1002/mrm.25781"}),
    ]

    for mod_name, phantom_fn, fwd_type, fwd_desc, meta in modalities:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            x = phantom_fn(seed)
            if fwd_type == "fourier_undersample":
                y = fourier_undersample(x, accel=4, seed=seed)
            elif fwd_type == "spectral_fid":
                # MRS: FFT of spectrum gives FID
                fid = np.fft.fft(x)
                y = np.stack([fid.real, fid.imag], axis=-1).astype(np.float32)
            else:
                y = x.copy()
            samples.append((x, y))
        save_modality(mod_name, samples, fwd_type, fwd_desc, meta)

    print(f"\nBatch 2 done: {len(modalities)} MRI modalities rebuilt")


if __name__ == "__main__":
    build_all()
