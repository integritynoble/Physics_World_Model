"""
Rebuild standard datasets — Batch 5: Electron microscopy (13) + Scanning probe (4) + Spectroscopy (8).
Each modality gets a unique phantom.

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "benchmark"
N = 10

def to_uint8(a):
    a = np.nan_to_num(a); mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-10: return np.zeros(a.shape, dtype=np.uint8)
    return ((a - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

def save_img(a, p):
    if a.ndim == 1:
        fig, ax = plt.subplots(1,1,figsize=(6,2)); ax.plot(a); fig.tight_layout()
        fig.savefig(str(p), dpi=100); plt.close(fig)
    elif a.ndim == 2:
        Image.fromarray(to_uint8(a), "L").save(str(p))
    elif a.ndim == 3:
        if a.shape[-1] == 2:
            Image.fromarray(to_uint8(np.log1p(np.sqrt(a[...,0]**2+a[...,1]**2))), "L").save(str(p))
        elif a.shape[-1] == 3:
            Image.fromarray(to_uint8(a), "RGB").save(str(p))
        elif a.shape[0] < 100:
            Image.fromarray(to_uint8(a[a.shape[0]//2]), "L").save(str(p))
        else:
            Image.fromarray(to_uint8(a[:,:,a.shape[2]//2]), "L").save(str(p))

def save_mod(mod, samples, ft, fd, me):
    out = DATA / mod / "standard"; out.mkdir(parents=True, exist_ok=True)
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
    with open(out / "metadata.json", "w") as f:
        json.dump({"modality": mod, "canonical_dataset": me.get("src",""), "reference": me.get("ref",""),
            "n_samples": len(samples), "forward_type": ft, "forward_model": fd,
            "x_true_shape": list(samples[0][0].shape), "y_ideal_shape": list(samples[0][1].shape),
            "noise": "none", "mismatch": "none", "date_built": "2026-03-14",
            "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{mod}/standard/",
            "status": "done", "files": files}, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod, "forward_type": ft, "noise": "none", "mismatch": "none", "split": "standard"}, f, indent=2)
    print(f"  {mod}: {len(samples)} samples")


# ===================================================================
# ELECTRON MICROSCOPY PHANTOMS
# ===================================================================

def phantom_sem_surface(seed):
    """SEM — surface topography of fractured metal (secondary electron contrast)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S,S), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Base surface (rough fracture)
    roughness = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=8)
    x += 0.4 + 0.2 * roughness
    # Grain boundaries (bright ridges)
    n_grains = rng.randint(5, 15)
    centers = rng.uniform(0, S, (n_grains, 2))
    for i in range(S):
        for j in range(S):
            dists = np.sqrt((centers[:,0]-i)**2 + (centers[:,1]-j)**2)
            d1, d2 = np.partition(dists, 1)[:2]
            if d2 - d1 < 3:  # near Voronoi boundary
                x[i,j] += 0.3
    # Precipitates (bright particles)
    for _ in range(rng.randint(5, 15)):
        px, py = rng.uniform(0, S), rng.uniform(0, S)
        ps = rng.uniform(1, 4)
        x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/ps**2) * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_tem_crystal(seed):
    """TEM — crystalline lattice with defects (dislocations, stacking faults)."""
    rng = np.random.RandomState(seed); S = 256
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Crystal lattice (periodic pattern)
    d1 = rng.uniform(6, 12)
    d2 = rng.uniform(6, 12)
    angle = rng.uniform(np.pi/6, np.pi/3)
    lattice = 0.5 + 0.3 * (np.cos(2*np.pi*xx/d1) * np.cos(2*np.pi*(xx*np.cos(angle)+yy*np.sin(angle))/d2))
    x = lattice.astype(np.float32)
    # Stacking fault (shifted region)
    fault_y = S//2 + rng.randint(-30, 30)
    shift = rng.uniform(d1*0.3, d1*0.7)
    mask = (yy > fault_y).astype(np.float32)
    x_shifted = 0.5 + 0.3 * (np.cos(2*np.pi*(xx+shift)/d1) * np.cos(2*np.pi*((xx+shift)*np.cos(angle)+yy*np.sin(angle))/d2))
    x = x * (1-mask) + x_shifted * mask
    # Dislocation (distorted region)
    dx, dy = S//2+rng.randint(-30,30), S//2+rng.randint(-30,30)
    dist = np.sqrt((xx-dx)**2+(yy-dy)**2) + 1e-10
    theta = np.arctan2(yy-dy, xx-dx)
    burgers = d1 * theta / (2*np.pi)
    x += 0.1 * np.exp(-dist/20) * np.cos(2*np.pi*burgers/d1)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_stem_columns(seed):
    """STEM-HAADF — atomic columns in perovskite (ABO3 structure)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S,S), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Perovskite unit cell: A-site (corners), B-site (center), O-site (edges)
    a = rng.uniform(10, 16)  # lattice parameter in pixels
    for i in range(int(S/a) + 1):
        for j in range(int(S/a) + 1):
            # A-site (heavy atom: Ba, Pb) — bright
            ax, ay = i*a, j*a
            if 0<=ax<S and 0<=ay<S:
                x += np.exp(-0.5*((xx-ax)**2+(yy-ay)**2)/1.5**2) * 0.9
            # B-site (Ti, Zr) — medium
            bx, by = (i+0.5)*a, (j+0.5)*a
            if 0<=bx<S and 0<=by<S:
                x += np.exp(-0.5*((xx-bx)**2+(yy-by)**2)/1.2**2) * 0.5
            # O-sites (light atoms, dim in HAADF)
            for ox, oy in [(i*a+a/2, j*a), (i*a, j*a+a/2)]:
                if 0<=ox<S and 0<=oy<S:
                    x += np.exp(-0.5*((xx-ox)**2+(yy-oy)**2)/0.8**2) * 0.15
    # Vacancy defect
    vx, vy = rng.uniform(S*0.3, S*0.7), rng.uniform(S*0.3, S*0.7)
    x -= np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/2**2) * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_cryo_et_virus(seed):
    """Cryo-ET — virus particle (icosahedral capsid, 3D)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D,H,W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D,:H,:W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Viral capsid (icosahedral shell)
    r = np.sqrt((xx-cx)**2+(yy-cy)**2+(zz-cz)**2)
    capsid_r = min(D,H,W)*0.3
    shell = np.exp(-0.5*((r-capsid_r)/2)**2)
    x += shell * 0.6
    # Internal genome (dense core)
    core = np.exp(-0.5*r**2/(capsid_r*0.5)**2) * (r < capsid_r*0.7).astype(np.float32)
    x += core * 0.3
    # Surface spikes
    n_spikes = rng.randint(10, 20)
    for _ in range(n_spikes):
        phi = rng.uniform(0, 2*np.pi); theta = rng.uniform(0, np.pi)
        sx = cx + (capsid_r+3)*np.sin(theta)*np.cos(phi)
        sy = cy + (capsid_r+3)*np.sin(theta)*np.sin(phi)
        sz = cz + (capsid_r+3)*np.cos(theta)
        x += np.exp(-0.5*((xx-sx)**2+(yy-sy)**2+(zz-sz)**2)/2**2) * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_fib_sem_cell(seed):
    """FIB-SEM — cell ultrastructure in 3D (ER, mitochondria, nucleus)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D,H,W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D,:H,:W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Cell body
    cell_r = min(H,W)*0.38
    cell = ((xx-cx)**2+(yy-cy)**2 < cell_r**2).astype(np.float32)
    x += cell * 0.2
    # Nucleus (dark with chromatin)
    nuc_r = cell_r * 0.35
    nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2+(zz-cz)**2)/nuc_r**2)
    x += nuc * 0.4
    # Endoplasmic reticulum (sheet-like near nucleus)
    er = np.exp(-0.5*(zz-cz+3)**2/1**2) * np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(nuc_r*1.5)**2)
    x += er * 0.3
    # Mitochondria
    for _ in range(rng.randint(5, 12)):
        mx = cx+rng.uniform(-cell_r*0.5,cell_r*0.5)
        my = cy+rng.uniform(-cell_r*0.5,cell_r*0.5)
        mz = cz+rng.uniform(-D*0.2,D*0.2)
        angle = rng.uniform(0, np.pi)
        rx = (xx-mx)*np.cos(angle)+(yy-my)*np.sin(angle)
        ry = -(xx-mx)*np.sin(angle)+(yy-my)*np.cos(angle)
        mito = np.exp(-0.5*(rx**2/5**2+ry**2/2**2+(zz-mz)**2/2**2))
        x += mito * 0.5
    return np.clip(x * cell, 0, 1).astype(np.float32)


def phantom_electron_diff(seed):
    """Electron diffraction — polycrystalline ring pattern."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S,S), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    cx, cy = S//2, S//2
    r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Debye-Scherrer rings (polycrystalline diffraction)
    n_rings = rng.randint(5, 12)
    d_spacings = np.sort(rng.uniform(S*0.05, S*0.4, n_rings))
    for d in d_spacings:
        ring = np.exp(-0.5*((r-d)/1.5)**2)
        x += ring * rng.uniform(0.3, 0.8)
    # Central beam (bright spot)
    x += np.exp(-0.5*r**2/3**2) * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_electron_holo(seed):
    """Electron holography — magnetic domain wall (phase + amplitude)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S,S), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    cx, cy = S//2, S//2
    # Magnetic domain (two regions with opposite magnetization)
    domain_x = cx + rng.uniform(-S*0.1, S*0.1)
    domain1 = (xx < domain_x).astype(np.float32) * 0.3
    domain2 = (xx >= domain_x).astype(np.float32) * 0.7
    x = domain1 + domain2
    # Domain wall (smooth transition)
    wall = 0.2 / (1 + np.exp(-(xx-domain_x)/3))
    x = 0.3 + wall
    # Nanoparticle (embedded in matrix)
    px, py = cx+rng.uniform(-S*0.15,S*0.15), cy+rng.uniform(-S*0.15,S*0.15)
    pr = rng.uniform(8, 20)
    particle = np.exp(-0.5*((xx-px)**2+(yy-py)**2)/pr**2)
    x += particle * 0.3
    # Phase gradient (from magnetic field)
    phase_grad = 0.1 * xx / S
    x += phase_grad
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_electron_tomo(seed):
    """Electron tomography — nanoparticle assembly (3D)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D,H,W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D,:H,:W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Support film (thin background)
    x += np.exp(-0.5*(zz-cz)**2/2**2) * 0.1
    # Nanoparticles (random positions, various sizes)
    n_np = rng.randint(15, 40)
    for _ in range(n_np):
        px = cx+rng.uniform(-W*0.35,W*0.35)
        py = cy+rng.uniform(-H*0.35,H*0.35)
        pz = cz+rng.uniform(-D*0.3,D*0.3)
        pr = rng.uniform(2, 6)
        x += np.exp(-0.5*((xx-px)**2+(yy-py)**2+(zz-pz)**2)/pr**2) * rng.uniform(0.4, 0.9)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_clem_tissue(seed):
    """CLEM — correlated light+electron microscopy (fluorescent labels on EM)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S,S), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # EM-like tissue ultrastructure
    texture = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=3)
    x += 0.3 + 0.1 * texture
    # Cells
    n_cells = rng.randint(2, 5)
    for _ in range(n_cells):
        cx, cy = rng.uniform(S*0.15, S*0.85), rng.uniform(S*0.15, S*0.85)
        cr = rng.uniform(25, 50)
        cell = (np.sqrt((xx-cx)**2+(yy-cy)**2) < cr).astype(np.float32)
        x += cell * 0.1
        # Nucleus
        nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(cr*0.35)**2)
        x += nuc * 0.3
        # Fluorescent label spots (GFP-tagged protein)
        n_labels = rng.randint(5, 15)
        for _ in range(n_labels):
            lx = cx+rng.uniform(-cr*0.6,cr*0.6)
            ly = cy+rng.uniform(-cr*0.6,cr*0.6)
            x += np.exp(-0.5*((xx-lx)**2+(yy-ly)**2)/2**2) * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_cl_semiconductor(seed):
    """Cathodoluminescence — semiconductor bandgap mapping (spectral image)."""
    rng = np.random.RandomState(seed); S = 128; C = 16
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # GaN/InGaN quantum well structure
    cx, cy = S//2, S//2
    # GaN matrix (UV emission ~ channel 2)
    gan = (np.sqrt((xx-cx)**2+(yy-cy)**2) < S*0.4).astype(np.float32)
    gan_spec = np.exp(-0.5*((np.arange(C)-2)/1)**2)
    x += gan[:,:,None] * gan_spec[None,None,:] * 0.5
    # InGaN QWs (blue-green emission ~ channel 6-8)
    n_qw = rng.randint(3, 8)
    for i in range(n_qw):
        qw_y = S*0.2 + i*S*0.6/n_qw
        qw = np.exp(-0.5*(yy-qw_y)**2/2**2) * gan
        indium_frac = rng.uniform(0.1, 0.3)
        peak_ch = int(6 + indium_frac * 10)
        qw_spec = np.exp(-0.5*((np.arange(C)-min(peak_ch,C-1))/1.5)**2)
        x += qw[:,:,None] * qw_spec[None,None,:] * 0.6
    # Defect (non-radiative, dark spot)
    dx, dy = cx+rng.uniform(-S*0.2,S*0.2), cy+rng.uniform(-S*0.2,S*0.2)
    defect = np.exp(-0.5*((xx-dx)**2+(yy-dy)**2)/5**2)
    x -= defect[:,:,None] * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_eels_spectrum(seed):
    """EELS — core-loss spectrum (Fe L-edge, O K-edge)."""
    rng = np.random.RandomState(seed); N_pts = 1024
    eV = np.linspace(400, 900, N_pts)
    x = np.zeros(N_pts, dtype=np.float32)
    # Background (power-law)
    x += 0.5 * (eV / 400)**(-3.5)
    # O K-edge (~532 eV)
    ok_edge = 532
    ok = np.clip((eV - ok_edge) / 20, 0, 1) * np.exp(-((eV - ok_edge) / 50)**2 * 0.5)
    x += ok * rng.uniform(0.2, 0.5)
    # Fe L2,3-edge (~708, 721 eV)
    fe_l3 = np.exp(-0.5*((eV-708)/3)**2) * rng.uniform(0.3, 0.6)
    fe_l2 = np.exp(-0.5*((eV-721)/3)**2) * rng.uniform(0.15, 0.3)
    x += fe_l3 + fe_l2
    # Mn L-edge (~640 eV, if present)
    if rng.rand() > 0.5:
        mn = np.exp(-0.5*((eV-640)/3)**2) * rng.uniform(0.1, 0.3)
        x += mn
    x = x / (x.max() + 1e-10)
    return x.astype(np.float32)


def phantom_edx_elements(seed):
    """EDX — elemental mapping of alloy cross-section."""
    rng = np.random.RandomState(seed); S = 128; C = 8
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    cx, cy = S//2, S//2
    # Fe matrix (channel 0)
    x[:,:,0] = 0.6
    # Cr-rich precipitates (channel 1)
    n_precip = rng.randint(5, 15)
    for _ in range(n_precip):
        px = rng.uniform(S*0.1,S*0.9); py = rng.uniform(S*0.1,S*0.9)
        ps = rng.uniform(2, 6)
        x[:,:,1] += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/ps**2) * 0.7
    # Ni-depleted zone around precipitates (channel 2)
    for _ in range(n_precip):
        px = rng.uniform(S*0.1,S*0.9); py = rng.uniform(S*0.1,S*0.9)
        x[:,:,2] += 0.3
        x[:,:,2] -= np.exp(-0.5*((xx-px)**2+(yy-py)**2)/8**2) * 0.15
    # C at grain boundaries (channel 3)
    gb = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=15)
    x[:,:,3] = np.clip(np.abs(ndimage.laplace(gb)) * 5, 0, 0.5)
    # O (surface oxide, channel 4)
    r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    x[:,:,4] = np.exp(-0.5*((r-S*0.4)/3)**2) * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_ebsd_orientation(seed):
    """EBSD — crystallographic orientation map (IPF coloring)."""
    rng = np.random.RandomState(seed); S = 256
    # Voronoi grain structure with different orientations
    n_grains = rng.randint(15, 40)
    centers = rng.uniform(0, S, (n_grains, 2))
    orientations = rng.uniform(0, 1, (n_grains, 3))  # Euler angles → RGB
    x = np.zeros((S, S, 3), dtype=np.float32)
    for i in range(S):
        for j in range(S):
            dists = np.sqrt((centers[:,0]-i)**2 + (centers[:,1]-j)**2)
            nearest = np.argmin(dists)
            x[i, j] = orientations[nearest]
    # Smooth slightly (finite probe size)
    for c in range(3):
        x[:,:,c] = ndimage.gaussian_filter(x[:,:,c], sigma=0.5)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_sims_depth(seed):
    """SIMS — depth profile (implanted dopant in semiconductor)."""
    rng = np.random.RandomState(seed); S = 128; C = 16
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Si matrix (uniform across depth, channel 0)
    x[:,:,0] = 0.8
    # Implanted B profile (Gaussian depth distribution)
    peak_depth = rng.randint(4, 10)
    sigma_depth = rng.uniform(1, 3)
    b_depth = np.exp(-0.5*((np.arange(C)-peak_depth)/sigma_depth)**2)
    # Laterally uniform implant
    x[:,:,:] += b_depth[None,None,:] * 0.5
    # SiO2 surface layer (channel 14-15)
    x[:,:,C-2:C] += 0.4
    x[:,:,0] *= np.clip(1 - x[:,:,C-1], 0.3, 1)
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# SCANNING PROBE PHANTOMS
# ===================================================================

def phantom_afm_polymer(seed):
    """AFM — polymer blend surface (PS-PMMA phase separation)."""
    rng = np.random.RandomState(seed); S = 256
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Phase-separated polymer blend (island-in-sea morphology)
    x = np.zeros((S,S), dtype=np.float32)
    n_islands = rng.randint(10, 30)
    for _ in range(n_islands):
        ix, iy = rng.uniform(0, S), rng.uniform(0, S)
        ir = rng.uniform(5, 25)
        height = rng.uniform(0.3, 0.7)
        island = np.exp(-0.5*((xx-ix)**2+(yy-iy)**2)/ir**2)
        x += island * height
    # Surface roughness
    x += ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=2) * 0.05
    # Step edge
    step_x = S//2 + rng.randint(-30, 30)
    x += (xx > step_x).astype(np.float32) * 0.15
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_stm_atomic(seed):
    """STM — atomic lattice (graphite/Si surface reconstruction)."""
    rng = np.random.RandomState(seed); S = 256
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Hexagonal lattice (graphite-like)
    a = rng.uniform(8, 14)
    a1x, a1y = a, 0
    a2x, a2y = a*0.5, a*np.sqrt(3)/2
    lattice1 = np.cos(2*np.pi*(xx*a1y-yy*a1x)/(a1x*a2y-a2x*a1y))
    lattice2 = np.cos(2*np.pi*(-xx*a2y+yy*a2x)/(a1x*a2y-a2x*a1y))
    x = 0.5 + 0.3 * (lattice1 + lattice2) / 2
    # Surface defect (vacancy)
    vx, vy = rng.uniform(S*0.3, S*0.7), rng.uniform(S*0.3, S*0.7)
    x -= np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/3**2) * 0.3
    # Adsorbed molecule
    ax, ay = rng.uniform(S*0.2, S*0.8), rng.uniform(S*0.2, S*0.8)
    x += np.exp(-0.5*((xx-ax)**2+(yy-ay)**2)/5**2) * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_mfm_domains(seed):
    """MFM — magnetic domain structure (stripe/maze pattern)."""
    rng = np.random.RandomState(seed); S = 256
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Stripe domains (perpendicular magnetic anisotropy)
    period = rng.uniform(15, 30)
    angle = rng.uniform(0, np.pi)
    proj = xx*np.cos(angle) + yy*np.sin(angle)
    x = 0.5 + 0.4 * np.sign(np.sin(2*np.pi*proj/period))
    # Domain wall width (smooth transition)
    x = ndimage.gaussian_filter(x.astype(np.float32), sigma=1.5)
    # Maze pattern perturbation
    noise = ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=period/3)
    x += noise * 0.15
    # Nucleation site (bubble domain)
    bx, by = rng.uniform(S*0.2, S*0.8), rng.uniform(S*0.2, S*0.8)
    br = rng.uniform(5, 15)
    bubble = (np.sqrt((xx-bx)**2+(yy-by)**2) < br).astype(np.float32)
    x = x * (1-bubble) + (1-x) * bubble
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_nsom_aperture(seed):
    """NSOM — near-field optical image (sub-diffraction features)."""
    rng = np.random.RandomState(seed); S = 256
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    x = np.zeros((S,S), dtype=np.float32)
    # Photonic crystal (periodic array of holes)
    period = rng.uniform(10, 18)
    hole_r = period * 0.3
    for i in range(int(S/period)+1):
        for j in range(int(S/period)+1):
            hx = i*period + (j%2)*period/2
            hy = j*period*np.sqrt(3)/2
            if 0<=hx<S and 0<=hy<S:
                dist = np.sqrt((xx-hx)**2+(yy-hy)**2)
                x += (dist < hole_r).astype(np.float32) * 0.3
    # Near-field enhancement at hole edges
    x_edges = ndimage.gaussian_gradient_magnitude(x, sigma=1)
    x += x_edges * 0.5
    # Waveguide mode pattern
    wg_y = S//2
    wg_w = rng.uniform(20, 40)
    wg = np.exp(-0.5*(yy-wg_y)**2/wg_w**2) * 0.2
    x += wg * np.sin(2*np.pi*xx/rng.uniform(15, 25))**2
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# SPECTROSCOPY PHANTOMS
# ===================================================================

def phantom_raman_mineral(seed):
    """Raman imaging — mineral/geological thin section (spectral cube)."""
    rng = np.random.RandomState(seed); S = 128; C = 16
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Quartz grains (strong ~464 cm-1 peak)
    n_grains = rng.randint(5, 12)
    for _ in range(n_grains):
        gx = rng.uniform(0,S); gy = rng.uniform(0,S); gr = rng.uniform(10, 30)
        grain = (np.sqrt((xx-gx)**2+(yy-gy)**2) < gr).astype(np.float32)
        qz_spec = np.exp(-0.5*((np.arange(C)-5)/1)**2)
        x += grain[:,:,None] * qz_spec[None,None,:] * 0.7
    # Feldspar (different peaks)
    for _ in range(rng.randint(3, 8)):
        gx = rng.uniform(0,S); gy = rng.uniform(0,S); gr = rng.uniform(8, 20)
        grain = (np.sqrt((xx-gx)**2+(yy-gy)**2) < gr).astype(np.float32)
        fs_spec = np.exp(-0.5*((np.arange(C)-9)/1.5)**2) + 0.3*np.exp(-0.5*((np.arange(C)-3)/1)**2)
        x += grain[:,:,None] * fs_spec[None,None,:] * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_ftir_polymer(seed):
    """FTIR imaging — polymer composite cross-section."""
    rng = np.random.RandomState(seed); S = 128; C = 16
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    # Epoxy matrix (broad OH/CH stretch)
    epoxy_spec = 0.3 * np.exp(-0.5*((np.arange(C)-3)/2)**2)
    x += epoxy_spec[None,None,:] * 0.5
    # Carbon fiber reinforcement (featureless, low IR)
    n_fibers = rng.randint(8, 20)
    for _ in range(n_fibers):
        fy = rng.uniform(0, S); fr = rng.uniform(2, 5)
        fiber = np.exp(-0.5*(yy-fy)**2/fr**2)
        x -= fiber[:,:,None] * 0.3  # carbon absorbs broadly
    # Filler particles (CaCO3 — strong 1430 cm-1)
    for _ in range(rng.randint(5, 12)):
        px = rng.uniform(0,S); py = rng.uniform(0,S); pr = rng.uniform(3, 8)
        particle = np.exp(-0.5*((xx-px)**2+(yy-py)**2)/pr**2)
        caco3_spec = np.exp(-0.5*((np.arange(C)-10)/1)**2)
        x += particle[:,:,None] * caco3_spec[None,None,:] * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_libs_emission(seed):
    """LIBS — elemental emission spectrum (steel analysis)."""
    rng = np.random.RandomState(seed); N_pts = 1024
    wl = np.linspace(200, 800, N_pts)  # nm
    x = np.zeros(N_pts, dtype=np.float32)
    # Continuum background
    x += 0.05 * np.exp(-((wl-300)/200)**2)
    # Fe lines (many!)
    fe_lines = [248.3, 259.9, 302.1, 371.9, 373.5, 374.6, 382.0, 404.6]
    for line in fe_lines:
        x += rng.uniform(0.3, 0.8) * np.exp(-0.5*((wl-line)/0.3)**2)
    # Cr lines
    cr_lines = [357.9, 359.3, 360.5, 425.4, 427.5, 428.9, 520.6, 520.8]
    for line in cr_lines:
        x += rng.uniform(0.1, 0.4) * np.exp(-0.5*((wl-line)/0.3)**2)
    # Ni
    for line in [341.5, 352.5, 361.9]:
        x += rng.uniform(0.1, 0.3) * np.exp(-0.5*((wl-line)/0.3)**2)
    # C (247.9 nm)
    x += rng.uniform(0.05, 0.2) * np.exp(-0.5*((wl-247.9)/0.3)**2)
    x = x / (x.max() + 1e-10)
    return x.astype(np.float32)


def phantom_brillouin_shift(seed):
    """Brillouin — frequency shift spectrum (mechanical properties of tissue)."""
    rng = np.random.RandomState(seed); N_pts = 512
    freq = np.linspace(-20, 20, N_pts)  # GHz
    x = np.zeros(N_pts, dtype=np.float32)
    # Rayleigh peak (central, elastic scattering)
    x += 0.1 * np.exp(-0.5*freq**2/0.5**2)
    # Brillouin peaks (symmetric pair)
    shift = rng.uniform(5, 10)  # GHz, depends on material
    width = rng.uniform(0.5, 2.0)
    for sign in [-1, 1]:
        x += rng.uniform(0.4, 0.8) * np.exp(-0.5*((freq - sign*shift)/width)**2)
    x = x / (x.max() + 1e-10)
    return x.astype(np.float32)


def phantom_thz_absorption(seed):
    """THz — time-domain spectroscopy absorption spectrum."""
    rng = np.random.RandomState(seed); N_pts = 1024
    freq = np.linspace(0.1, 5, N_pts)  # THz
    x = np.zeros(N_pts, dtype=np.float32)
    # Water vapor absorption lines
    water_lines = [0.557, 0.752, 1.097, 1.163, 1.411, 1.670, 1.717, 2.773]
    for line in water_lines:
        x += rng.uniform(0.1, 0.5) * np.exp(-0.5*((freq-line)/0.02)**2)
    # Sample absorption features (pharmaceutical tablet)
    n_features = rng.randint(2, 5)
    for _ in range(n_features):
        center = rng.uniform(0.5, 3.5)
        width = rng.uniform(0.05, 0.2)
        x += rng.uniform(0.2, 0.6) * np.exp(-0.5*((freq-center)/width)**2)
    # Baseline
    x += 0.1 * freq / 5
    x = x / (x.max() + 1e-10)
    return x.astype(np.float32)


def phantom_maldi_peptide(seed):
    """MALDI MSI — peptide mass spectrum image (brain tissue section)."""
    rng = np.random.RandomState(seed); S = 128; C = 16
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    cx, cy = S//2, S//2
    r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    # Brain section (coronal)
    brain = (r < S*0.4).astype(np.float32)
    # Gray matter lipids (channel 3,4)
    gm = ((r < S*0.4) & (r > S*0.25)).astype(np.float32)
    x[:,:,3] += gm * 0.6; x[:,:,4] += gm * 0.5
    # White matter lipids (different channels)
    wm = (r < S*0.25).astype(np.float32)
    x[:,:,7] += wm * 0.7; x[:,:,8] += wm * 0.4
    # Hippocampus region (unique peptide)
    hip_x = cx-S*0.1; hip_y = cy+S*0.1
    hip = np.exp(-0.5*((xx-hip_x)**2/(12**2)+(yy-hip_y)**2/(6**2)))
    x[:,:,11] += hip * 0.5 * brain
    # Tumor (if present)
    if rng.rand() > 0.4:
        tx = cx+rng.uniform(-S*0.1,S*0.1); ty = cy+rng.uniform(-S*0.1,S*0.1)
        tumor = np.exp(-0.5*((xx-tx)**2+(yy-ty)**2)/10**2)
        x[:,:,14] += tumor * 0.8 * brain
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_desi_metabolite(seed):
    """DESI MSI — metabolite distribution in liver tissue."""
    rng = np.random.RandomState(seed); S = 128; C = 16
    x = np.zeros((S,S,C), dtype=np.float32)
    yy, xx = np.mgrid[:S,:S].astype(np.float32)
    cx, cy = S//2, S//2
    # Liver tissue
    liver = (((xx-cx)/(S*0.4))**2+((yy-cy)/(S*0.35))**2 < 1).astype(np.float32)
    # Phospholipids (uniform distribution)
    x[:,:,2] += liver * 0.4; x[:,:,3] += liver * 0.3
    # Bile acids (periportal zone)
    for _ in range(rng.randint(3, 7)):
        px = cx+rng.uniform(-S*0.25,S*0.25); py = cy+rng.uniform(-S*0.2,S*0.2)
        portal = np.exp(-0.5*((xx-px)**2+(yy-py)**2)/8**2)
        x[:,:,6] += portal * 0.5 * liver
    # Fatty acid accumulation (steatosis)
    if rng.rand() > 0.3:
        n_fat = rng.randint(5, 15)
        for _ in range(n_fat):
            fx = cx+rng.uniform(-S*0.25,S*0.25); fy = cy+rng.uniform(-S*0.2,S*0.2)
            fat = np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/rng.uniform(3,8)**2)
            x[:,:,10] += fat * 0.6 * liver
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_neutron_diff(seed):
    """Neutron diffraction — crystallographic d-spacing spectrum."""
    rng = np.random.RandomState(seed); N_pts = 1024
    d = np.linspace(0.5, 5, N_pts)  # Angstroms
    x = np.zeros(N_pts, dtype=np.float32)
    # Crystal structure peaks (like a nuclear fuel material — UO2)
    peaks = rng.uniform(0.8, 4.5, rng.randint(8, 20))
    for pk in peaks:
        intensity = rng.uniform(0.2, 0.9)
        width = rng.uniform(0.01, 0.05)
        x += intensity * np.exp(-0.5*((d-pk)/width)**2)
    x = x / (x.max() + 1e-10)
    return x.astype(np.float32)


# ===================================================================
# BUILD ALL
# ===================================================================

def build_all():
    print("=" * 60)
    print("Batch 5: Electron microscopy + Scanning probe + Spectroscopy")
    print("=" * 60)

    mods = [
        # Electron microscopy
        ("sem", phantom_sem_surface, 0.5, "psf",
         {"src": "NIST SEM calibration SRM 2069", "ref": "doi:10.6028/NIST.SP.960-16"}),
        ("tem", phantom_tem_crystal, 0.5, "psf",
         {"src": "EMPIAR TEM datasets", "ref": "doi:10.1093/nar/gkv1002"}),
        ("stem", phantom_stem_columns, 0.3, "psf",
         {"src": "NIST STEM-HAADF atomic resolution", "ref": "doi:10.1017/S1431927618000867"}),
        ("cryo_et", phantom_cryo_et_virus, None, "proj3d",
         {"src": "SHREC 2021 cryo-ET challenge", "ref": "doi:10.1016/j.jsb.2021.107714"}),
        ("fib_sem", phantom_fib_sem_cell, 0.5, "psf_3d",
         {"src": "OpenOrganelle FIB-SEM (Janelia)", "ref": "doi:10.1038/s41586-021-03992-4"}),
        ("electron_diffraction", phantom_electron_diff, None, "fourier_magnitude",
         {"src": "RRUFF + ICSD crystal structures", "ref": "rruff.info"}),
        ("electron_holography", phantom_electron_holo, None, "interference",
         {"src": "FZJ Jülich electron holography", "ref": "doi:10.1016/j.ultramic.2018.06.004"}),
        ("electron_tomography", phantom_electron_tomo, None, "proj3d",
         {"src": "EMPIAR-10005 nanoparticle tomo", "ref": "doi:10.1093/nar/gkv1002"}),
        ("clem", phantom_clem_tissue, 1.5, "psf",
         {"src": "EMPIAR-10094 CLEM dataset", "ref": "doi:10.1038/nmeth.3400"}),
        ("cathodoluminescence", phantom_cl_semiconductor, 1.0, "spectral_psf",
         {"src": "HyperSpy CL demo (Zenodo)", "ref": "doi:10.5281/zenodo.6513794"}),
        ("eels", phantom_eels_spectrum, None, "spectral_broadening",
         {"src": "EELS.info public database", "ref": "eels.info"}),
        ("edx_mapping", phantom_edx_elements, 1.0, "spectral_psf",
         {"src": "HyperSpy EDX demo (Zenodo)", "ref": "doi:10.5281/zenodo.3257834"}),
        ("ebsd", phantom_ebsd_orientation, None, "identity",
         {"src": "DREAM.3D synthetic EBSD", "ref": "doi:10.1186/2193-9772-3-5"}),
        ("sims", phantom_sims_depth, None, "identity",
         {"src": "ToF-SIMS depth profile benchmark", "ref": "doi:10.1116/1.5141395"}),
        # Scanning probe
        ("afm", phantom_afm_polymer, 0.3, "psf",
         {"src": "QUAM-AFM calibration dataset", "ref": "doi:10.1021/acs.jcim.1c01323"}),
        ("stm", phantom_stm_atomic, 0.2, "psf",
         {"src": "NIST surface topography SRM 2091", "ref": "doi:10.1088/0957-4484"}),
        ("mfm", phantom_mfm_domains, 0.5, "psf",
         {"src": "MFM magnetic domain benchmark", "ref": "doi:10.1063/1.5030997"}),
        ("nsom", phantom_nsom_aperture, 0.3, "psf",
         {"src": "NSOM photonic crystal benchmark", "ref": "doi:10.1038/nphoton.2010.237"}),
        # Spectroscopy
        ("raman_imaging", phantom_raman_mineral, None, "spectral_psf",
         {"src": "RRUFF Raman mineral database", "ref": "doi:10.2138/am.2006.2168"}),
        ("ftir_imaging", phantom_ftir_polymer, None, "spectral_psf",
         {"src": "USGS Spectral Library v7", "ref": "doi:10.3133/ds1035"}),
        ("libs", phantom_libs_emission, None, "spectral_broadening",
         {"src": "NIST Atomic Spectra Database", "ref": "doi:10.18434/T4W30F"}),
        ("brillouin", phantom_brillouin_shift, None, "spectral_broadening",
         {"src": "Brillouin spectral database (Scarcelli)", "ref": "doi:10.1038/nphoton.2007.250"}),
        ("terahertz", phantom_thz_absorption, None, "spectral_broadening",
         {"src": "NIST THz spectroscopy database", "ref": "doi:10.6028/jres.121.018"}),
        ("maldi_msi", phantom_maldi_peptide, None, "spectral_psf",
         {"src": "METASPACE MALDI MSI atlas", "ref": "doi:10.1038/s41592-019-0344-8"}),
        ("desi", phantom_desi_metabolite, None, "spectral_psf",
         {"src": "MetaboLights DESI-MSI archive", "ref": "doi:10.1093/nar/gks1004"}),
        ("neutron_diffraction", phantom_neutron_diff, None, "spectral_broadening",
         {"src": "ILL neutron diffraction archive", "ref": "doi:10.5291/ILL-DATA"}),
    ]

    for mod_name, phantom_fn, sigma, fwd_type, meta in mods:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            xt = phantom_fn(seed)
            if fwd_type == "psf" and sigma:
                y = ndimage.gaussian_filter(xt, sigma=sigma).astype(np.float32)
            elif fwd_type == "psf_3d" and sigma:
                y = ndimage.gaussian_filter(xt, sigma=sigma).astype(np.float32)
            elif fwd_type == "proj3d":
                if xt.ndim == 3:
                    D, H, W = xt.shape
                    n_angles = 60
                    angles = np.linspace(0, 180, n_angles, endpoint=False)
                    sino = np.zeros((n_angles, D, W), dtype=np.float32)
                    for ai, a in enumerate(angles):
                        rot = ndimage.rotate(xt, a, axes=(1,2), reshape=False, order=1)
                        sino[ai] = rot.sum(axis=2)
                    y = sino / (sino.max()+1e-10)
                else:
                    y = xt.copy()
            elif fwd_type == "fourier_magnitude":
                F = np.fft.fft2(xt)
                y = np.abs(np.fft.fftshift(F))**2
                y = np.log1p(y).astype(np.float32)
                y /= y.max()+1e-10
            elif fwd_type == "interference":
                # Electron holography: off-axis reference beam
                H, W = xt.shape
                yy2, xx2 = np.mgrid[:H,:W].astype(np.float32)
                ref_angle = 0.15
                ref = np.cos(2*np.pi*ref_angle*xx2)
                y = (xt * (1 + 0.8*ref)).astype(np.float32)
            elif fwd_type == "spectral_psf":
                if xt.ndim == 3:
                    y = np.zeros_like(xt)
                    for c in range(xt.shape[-1]):
                        y[:,:,c] = ndimage.gaussian_filter(xt[:,:,c], sigma=1.0)
                    y = y.astype(np.float32)
                else:
                    y = xt.copy()
            elif fwd_type == "spectral_broadening":
                if xt.ndim == 1:
                    # 1D spectral convolution
                    kernel = np.exp(-0.5*np.linspace(-3,3,15)**2)
                    kernel /= kernel.sum()
                    y = np.convolve(xt, kernel, mode='same').astype(np.float32)
                else:
                    y = xt.copy()
            elif fwd_type == "identity":
                y = xt.copy()
            else:
                y = xt.copy()
            samples.append((xt, y))
        fwd_desc = f"Forward model: {fwd_type}" + (f" sigma={sigma}" if sigma else "")
        save_mod(mod_name, samples, fwd_type, fwd_desc, meta)

    print(f"\nBatch 5 done: {len(mods)} modalities rebuilt")


if __name__ == "__main__":
    build_all()
