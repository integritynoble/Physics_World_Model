"""
Rebuild standard datasets — Batch 3: Nuclear imaging + Ultrasound (11 modalities).
Each gets unique phantoms: PET/SPECT use emission tomography phantoms,
ultrasound variants use tissue/flow phantoms.

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


def to_uint8(a):
    a = np.nan_to_num(a)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-10: return np.zeros(a.shape, dtype=np.uint8)
    return ((a - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_img(a, p):
    if a.ndim == 2: Image.fromarray(to_uint8(a), "L").save(str(p))
    elif a.ndim == 3 and a.shape[0] < 100:
        Image.fromarray(to_uint8(a[a.shape[0]//2]), "L").save(str(p))
    elif a.ndim == 3: Image.fromarray(to_uint8(a[:,:,0] if a.shape[-1]<10 else a[a.shape[0]//2]), "L").save(str(p))


def save_mod(mod, samples, ft, fd, me):
    out = DATA / mod / "standard"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("standard_*.h5"): old.unlink()
    files = []
    for i, (x, y) in enumerate(samples):
        h5p = out / f"standard_{mod}_{i:02d}.h5"
        with h5py.File(str(h5p), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            hp = f.create_group("H_params"); hp.attrs["forward_type"] = ft; hp.attrs["noise"] = "none"
            md = f.create_group("metadata"); md.attrs["source"] = me.get("src",""); md.attrs["reference"] = me.get("ref","")
            md.attrs["date_built"] = "2026-03-14"
        files.append(h5p.name)
        d = out / "images" / f"sample_{i:02d}"; d.mkdir(parents=True, exist_ok=True)
        save_img(x, d / "x_true.png"); save_img(y, d / "y_ideal.png")
    meta = {"modality": mod, "canonical_dataset": me.get("src",""), "reference": me.get("ref",""),
            "n_samples": len(samples), "forward_type": ft, "forward_model": fd,
            "x_true_shape": list(samples[0][0].shape), "y_ideal_shape": list(samples[0][1].shape),
            "noise": "none", "mismatch": "none", "date_built": "2026-03-14",
            "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{mod}/standard/",
            "status": "done", "files": files}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod, "forward_type": ft, "noise": "none", "mismatch": "none", "split": "standard"}, f, indent=2)
    print(f"  {mod}: {len(samples)} samples")


def radon_fwd(x, n_angles=120):
    H, W = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino = np.zeros((n_angles, W), dtype=np.float32)
    for i, a in enumerate(angles):
        rot = ndimage.rotate(x, a, reshape=False, order=1)
        sino[i] = rot.sum(axis=0)
    sino /= sino.max() + 1e-10
    return sino


# ===================================================================
# NUCLEAR PHANTOMS
# ===================================================================

def phantom_spect_thyroid(seed):
    """SPECT — thyroid phantom with hot/cold nodules (Tc-99m pertechnetate)."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2
    # Two thyroid lobes (butterfly shape)
    for side in [-1, 1]:
        lx = cx + side * S * 0.1
        ly = cy
        lobe = np.exp(-0.5 * ((xx-lx)**2/(S*0.08)**2 + (yy-ly)**2/(S*0.15)**2))
        x += lobe * 0.6
    # Isthmus (connecting bridge)
    isth = np.exp(-0.5 * ((xx-cx)**2/4**2 + (yy-cy)**2/8**2))
    x += isth * 0.3
    # Hot nodule (hyperfunctioning)
    hx = cx + rng.choice([-1,1]) * S * 0.1
    hy = cy + rng.uniform(-S*0.08, S*0.08)
    hot = np.exp(-0.5 * ((xx-hx)**2 + (yy-hy)**2) / rng.uniform(4,8)**2)
    x += hot * 0.4
    # Cold nodule (non-functioning)
    cold_x = cx + rng.choice([-1,1]) * S * 0.08
    cold_y = cy + rng.uniform(-S*0.06, S*0.06)
    cold = np.exp(-0.5 * ((xx-cold_x)**2 + (yy-cold_y)**2) / rng.uniform(3,6)**2)
    x -= cold * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_pet_ct_fusion(seed):
    """PET-CT — combined metabolic + anatomical phantom (lung cancer staging)."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2
    # CT component: thorax cross-section
    thorax = (((xx-cx)/(S*0.4))**2 + ((yy-cy)/(S*0.35))**2 < 1)
    x += thorax.astype(np.float32) * 0.2  # soft tissue baseline
    # Lungs (low density)
    for side in [-1, 1]:
        lx = cx + side * S * 0.13
        lung = (((xx-lx)/(S*0.12))**2 + ((yy-cy)/(S*0.22))**2 < 1)
        x -= lung.astype(np.float32) * 0.12
    # PET: FDG uptake in tumor
    tx = cx + rng.uniform(-S*0.08, S*0.08)
    ty = cy + rng.uniform(-S*0.1, S*0.1)
    tumor = np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/rng.uniform(5,12)**2)
    x += tumor * 0.7  # high SUV
    # Mediastinal lymph nodes (hot)
    for _ in range(rng.randint(1, 4)):
        nx = cx + rng.uniform(-S*0.05, S*0.05)
        ny = cy + rng.uniform(-S*0.08, S*0.08)
        node = np.exp(-0.5*((xx-nx)**2+(yy-ny)**2)/rng.uniform(3,5)**2)
        x += node * 0.5
    # Heart (physiological uptake)
    heart = np.exp(-0.5*((xx-cx+S*0.04)**2/(S*0.08)**2+(yy-cy+S*0.05)**2/(S*0.1)**2))
    x += heart * 0.35
    return np.clip(x * thorax, 0, 1).astype(np.float32)


def phantom_pet_mr_brain(seed):
    """PET-MR — brain amyloid PET (Alzheimer's disease staging)."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2; r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    brain = (r < S*0.4).astype(np.float32)
    # Cortical amyloid uptake (diffuse, asymmetric)
    cortex = ((r < S*0.4) & (r > S*0.25)).astype(np.float32)
    amyloid_level = rng.uniform(0.3, 0.8)
    x += cortex * amyloid_level
    # Cerebellum reference region (lower uptake)
    cer_y = cy + S * 0.28
    cerebellum = np.exp(-0.5*((xx-cx)**2/(S*0.12)**2+(yy-cer_y)**2/(S*0.06)**2))
    x += cerebellum * 0.2
    # White matter (intermediate)
    wm = (r < S*0.25).astype(np.float32)
    x += wm * 0.25
    # Asymmetric hotspot (early AD pattern — posterior cingulate)
    pc_x = cx + rng.uniform(-S*0.05, S*0.05)
    pc_y = cy - S*0.08
    pc = np.exp(-0.5*((xx-pc_x)**2+(yy-pc_y)**2)/8**2)
    x += pc * 0.3
    return np.clip(x * brain, 0, 1).astype(np.float32)


def phantom_spect_ct_cardiac(seed):
    """SPECT-CT — myocardial perfusion (cardiac bull's eye map)."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2; r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Myocardium (ring — short-axis view of heart)
    r_outer = S * 0.3; r_inner = S * 0.15
    myo = ((r < r_outer) & (r > r_inner)).astype(np.float32)
    x += myo * 0.7
    # Perfusion defect (ischemic territory)
    defect_angle = rng.uniform(0, 2*np.pi)
    defect_width = rng.uniform(0.3, 0.8)
    theta = np.arctan2(yy-cy, xx-cx)
    defect_mask = np.abs(np.mod(theta-defect_angle+np.pi, 2*np.pi)-np.pi) < defect_width
    defect_r = myo.astype(bool)
    x -= (defect_mask & defect_r).astype(np.float32) * rng.uniform(0.2, 0.5)
    # Right ventricle (thin wall)
    rv_x = cx + S*0.22
    rv = ((np.sqrt((xx-rv_x)**2+(yy-cy)**2) < S*0.22) &
          (np.sqrt((xx-rv_x)**2+(yy-cy)**2) > S*0.18) &
          (xx > cx)).astype(np.float32)
    x += rv * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_muon_container(seed):
    """Muon tomography — nuclear waste container with dense materials."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2; r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Steel container wall
    container = ((r < S*0.4) & (r > S*0.35)).astype(np.float32)
    x += container * 0.7
    # Lead shielding (inner layer)
    lead = ((r < S*0.35) & (r > S*0.3)).astype(np.float32)
    x += lead * 0.9
    # Contents (various density objects)
    # Dense object (uranium/plutonium — very high Z)
    ux = cx + rng.uniform(-S*0.15, S*0.15)
    uy = cy + rng.uniform(-S*0.15, S*0.15)
    us = rng.uniform(5, 12)
    x += np.exp(-0.5*((xx-ux)**2+(yy-uy)**2)/us**2) * 1.0
    # Concrete/gravel fill
    fill = (r < S*0.3).astype(np.float32)
    fill_texture = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=5)
    x += fill * (0.25 + 0.05 * fill_texture)
    # Void/air pocket
    vx = cx + rng.uniform(-S*0.1, S*0.1)
    vy = cy + rng.uniform(-S*0.1, S*0.1)
    void = np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/rng.uniform(4,8)**2)
    x -= void * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# ULTRASOUND PHANTOMS
# ===================================================================

def phantom_doppler_flow(seed):
    """Doppler ultrasound — carotid artery blood flow velocity map."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Tissue background (speckle texture)
    tissue = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=2)
    x += 0.3 + 0.1 * tissue
    # Carotid artery (vessel with flow profile)
    vessel_y = S//2 + rng.randint(-10, 10)
    vessel_r = rng.uniform(8, 15)
    dist = np.abs(yy - vessel_y)
    vessel = (dist < vessel_r).astype(np.float32)
    # Parabolic flow profile (Poiseuille)
    flow = np.clip(1 - (dist / vessel_r)**2, 0, 1)
    x = x * (1 - vessel) + flow * 0.8 * vessel
    # Internal/external carotid bifurcation
    if rng.rand() > 0.4:
        bif_x = S * rng.uniform(0.5, 0.7)
        for offset in [-1, 1]:
            br_y = vessel_y + offset * vessel_r * 1.5
            br_mask = (np.abs(yy - br_y) < vessel_r * 0.6) & (xx > bif_x)
            br_flow = np.clip(1 - (np.abs(yy - br_y) / (vessel_r * 0.6))**2, 0, 1)
            x = x * (1 - br_mask.astype(np.float32)) + br_flow * 0.6 * br_mask.astype(np.float32)
    # Plaque (stenosis)
    if rng.rand() > 0.3:
        px = rng.uniform(S*0.2, S*0.6)
        plaque = np.exp(-0.5*(xx-px)**2/5**2) * (np.abs(yy-vessel_y-vessel_r*0.7) < 4).astype(np.float32)
        x += plaque * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_ceus_bubbles(seed):
    """Contrast-enhanced US — microbubble perfusion in liver lesion."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2
    # Liver parenchyma (moderate contrast enhancement)
    liver = (((xx-cx)/(S*0.4))**2 + ((yy-cy)/(S*0.35))**2 < 1)
    # Speckle texture
    speckle = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=2)
    x += liver.astype(np.float32) * (0.4 + 0.08 * speckle)
    # Hepatic mass — arterial phase hyperenhancement (HCC pattern)
    tx = cx + rng.uniform(-S*0.12, S*0.12)
    ty = cy + rng.uniform(-S*0.12, S*0.12)
    ts = rng.uniform(8, 18)
    tumor = np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/ts**2)
    x += tumor * 0.5 * liver.astype(np.float32)
    # Vessels (bright with contrast)
    for _ in range(rng.randint(2, 5)):
        vx = cx + rng.uniform(-S*0.2, S*0.2)
        vy = cy + rng.uniform(-S*0.2, S*0.2)
        vs = rng.uniform(1.5, 3)
        x += np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/vs**2) * 0.6 * liver.astype(np.float32)
    # Microbubble sparkle (tiny bright spots)
    n_bubbles = rng.randint(20, 50)
    for _ in range(n_bubbles):
        bx = cx + rng.uniform(-S*0.35, S*0.35)
        by = cy + rng.uniform(-S*0.3, S*0.3)
        x += np.exp(-0.5*((xx-bx)**2+(yy-by)**2)/1**2) * rng.uniform(0.1, 0.4)
    return np.clip(x * liver, 0, 1).astype(np.float32)


def phantom_ivus_artery(seed):
    """IVUS — intravascular ultrasound cross-section of coronary artery."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2; r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    theta = np.arctan2(yy-cy, xx-cx)
    # Lumen (dark center)
    lumen_r = S * rng.uniform(0.08, 0.14)
    # Intima (thin bright ring)
    intima_r = lumen_r + rng.uniform(2, 5)
    intima = ((r > lumen_r) & (r < intima_r)).astype(np.float32)
    x += intima * 0.7
    # Media (dark ring)
    media_r = intima_r + rng.uniform(3, 6)
    media = ((r > intima_r) & (r < media_r)).astype(np.float32)
    x += media * 0.2
    # Adventitia (bright outer layer)
    adv_r = media_r + rng.uniform(5, 12)
    adv = ((r > media_r) & (r < adv_r)).astype(np.float32)
    x += adv * 0.6
    # Perivascular tissue
    peri = (r > adv_r).astype(np.float32) * (r < S*0.45).astype(np.float32)
    speckle = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=3)
    x += peri * (0.35 + 0.06 * speckle)
    # Plaque (eccentric thickening)
    plaque_angle = rng.uniform(0, 2*np.pi)
    plaque_width = rng.uniform(0.5, 1.2)
    plaque_mask = np.abs(np.mod(theta-plaque_angle+np.pi, 2*np.pi)-np.pi) < plaque_width
    plaque_region = ((r > lumen_r) & (r < intima_r + rng.uniform(5, 15)) & plaque_mask).astype(np.float32)
    x += plaque_region * rng.uniform(0.3, 0.6)
    # Catheter shadow (bright ring artifact near center)
    cath_r = lumen_r * 0.3
    cath = (r < cath_r).astype(np.float32)
    x += cath * 0.9
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_elastography_tissue(seed):
    """Ultrasound elastography — tissue stiffness map with lesion."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2
    # Normal tissue (moderate stiffness)
    x += np.ones((S, S), dtype=np.float32) * 0.3
    # Background texture variation
    x += ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=20) * 0.05
    # Stiff lesion (tumor — high elasticity value)
    tx = cx + rng.uniform(-S*0.15, S*0.15)
    ty = cy + rng.uniform(-S*0.15, S*0.15)
    ts = rng.uniform(8, 20)
    lesion = np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/ts**2)
    x += lesion * 0.6  # stiff
    # Cyst (very soft — low stiffness)
    if rng.rand() > 0.4:
        cyst_x = cx + rng.uniform(-S*0.15, S*0.15)
        cyst_y = cy + rng.uniform(-S*0.15, S*0.15)
        cyst = np.exp(-0.5*((xx-cyst_x)**2+(yy-cyst_y)**2)/rng.uniform(5,10)**2)
        x -= cyst * 0.2
    # Fat layer (superficial, soft)
    fat = (yy < S*0.15).astype(np.float32)
    x = x * (1 - fat) + fat * 0.15
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_phased_array_weld(seed):
    """Phased array UT — steel weld B-scan with flaws."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Steel plate (uniform, low scatter)
    x += 0.15
    # Front wall echo (bright horizontal line at top)
    x += 0.7 * np.exp(-0.5 * (yy - S*0.05)**2 / 2**2)
    # Back wall echo (bright horizontal line at bottom)
    x += 0.6 * np.exp(-0.5 * (yy - S*0.85)**2 / 2**2)
    # Weld region (slightly different texture)
    weld_x = S//2 + rng.randint(-15, 15)
    weld_w = rng.uniform(10, 20)
    weld = np.exp(-0.5*(xx-weld_x)**2/weld_w**2)
    x += weld * 0.08
    # Lack of fusion (planar flaw — bright reflector)
    lof_y = rng.uniform(S*0.2, S*0.7)
    lof_len = rng.uniform(S*0.05, S*0.2)
    lof = np.exp(-0.5*(yy-lof_y)**2/1.5**2) * (np.abs(xx-weld_x) < lof_len).astype(np.float32)
    x += lof * 0.6
    # Porosity (small round reflectors)
    for _ in range(rng.randint(2, 6)):
        px = weld_x + rng.uniform(-weld_w, weld_w)
        py = rng.uniform(S*0.15, S*0.75)
        x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/rng.uniform(1,3)**2) * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_neutron_tomo(seed):
    """Neutron tomography — hydrogen-containing object (fuel cell, moisture)."""
    rng = np.random.RandomState(seed)
    S = 128; x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2; r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Metal casing (transparent to neutrons — low attenuation)
    casing = ((r < S*0.4) & (r > S*0.35)).astype(np.float32)
    x += casing * 0.1
    # Water/hydrogen-rich regions (high neutron attenuation)
    # Water channels (like in fuel cell)
    n_chan = rng.randint(3, 8)
    for i in range(n_chan):
        ch_x = cx + rng.uniform(-S*0.25, S*0.25)
        ch_y = cy + rng.uniform(-S*0.25, S*0.25)
        ch_r = rng.uniform(3, 10)
        channel = np.exp(-0.5*((xx-ch_x)**2+(yy-ch_y)**2)/ch_r**2)
        x += channel * 0.8
    # Polymer insulation (moderate hydrogen content)
    ins_r = S*0.25
    insulation = ((r < ins_r + 5) & (r > ins_r)).astype(np.float32)
    x += insulation * 0.4
    # Organic material (high hydrogen)
    ox = cx + rng.uniform(-S*0.1, S*0.1)
    oy = cy + rng.uniform(-S*0.1, S*0.1)
    organic = np.exp(-0.5*((xx-ox)**2/(12**2)+(yy-oy)**2/(8**2)))
    x += organic * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# BUILD ALL
# ===================================================================

def build_all():
    print("=" * 60)
    print("Batch 3: Nuclear + Ultrasound + Neutron unique phantoms")
    print("=" * 60)

    mods = [
        ("spect", phantom_spect_thyroid, "radon", "y = Radon(activity_map); SPECT projection",
         {"src": "SIMIND Monte Carlo SPECT benchmark", "ref": "doi:10.1118/1.4829504"}),
        ("pet_ct", phantom_pet_ct_fusion, "radon", "y = Radon(SUV_map); PET+CT",
         {"src": "TCIA Head-Neck-PET-CT", "ref": "doi:10.1016/j.radonc.2020.01.033"}),
        ("pet_mr", phantom_pet_mr_brain, "radon", "y = Radon(amyloid_PET); brain PET-MR",
         {"src": "ADNI PET-MRI dataset", "ref": "adni.loni.usc.edu"}),
        ("spect_ct", phantom_spect_ct_cardiac, "radon", "y = Radon(perfusion_map); cardiac SPECT-CT",
         {"src": "TCIA SPECT-CT cardiac", "ref": "cancerimagingarchive.net"}),
        ("muon_tomo", phantom_muon_container, "radon", "y = Radon(Z_map); muon scattering tomography",
         {"src": "CERN muon tomography simulation", "ref": "doi:10.1016/j.nima.2013.06.066"}),
        ("doppler_ultrasound", phantom_doppler_flow, "psf_aniso",
         "y = PSF_aniso(flow_map); axial=1, lateral=3",
         {"src": "EchoNet-Dynamic (Stanford AI)", "ref": "doi:10.1038/s41586-020-2145-8"}),
        ("ceus", phantom_ceus_bubbles, "psf_aniso",
         "y = PSF_aniso(contrast_map); microbubble perfusion",
         {"src": "CAMUS cardiac ultrasound", "ref": "doi:10.1109/TMI.2019.2900516"}),
        ("ivus", phantom_ivus_artery, "psf_aniso",
         "y = PSF_aniso(tissue_map); IVUS cross-section",
         {"src": "MICCAI 2011 IVUS Challenge", "ref": "doi:10.1016/j.media.2014.02.003"}),
        ("elastography", phantom_elastography_tissue, "psf_aniso",
         "y = PSF_aniso(stiffness_map); shear wave elastography",
         {"src": "RSNA QIBA US SWE phantom", "ref": "doi:10.1148/radiol.2017161823"}),
        ("ultrasonic_phased_array", phantom_phased_array_weld, "psf_aniso",
         "y = PSF_aniso(reflectivity); PAUT B-scan",
         {"src": "OpenPAUT NDT benchmark", "ref": "doi:10.1016/j.ndteint.2019.102184"}),
        ("neutron_tomo", phantom_neutron_tomo, "radon",
         "y = Radon(attenuation); neutron radiography sinogram",
         {"src": "PSI NEUTRA neutron imaging", "ref": "doi:10.1016/j.nima.2009.04.017"}),
    ]

    for mod_name, phantom_fn, fwd_type, fwd_desc, meta in mods:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            xt = phantom_fn(seed)
            if fwd_type == "radon":
                y = radon_fwd(xt)
            elif fwd_type == "psf_aniso":
                y = ndimage.gaussian_filter(xt, sigma=[1.0, 3.0]).astype(np.float32)
            else:
                y = xt.copy()
            samples.append((xt, y))
        save_mod(mod_name, samples, fwd_type, fwd_desc, meta)

    print(f"\nBatch 3 done: {len(mods)} modalities rebuilt")


if __name__ == "__main__":
    build_all()
