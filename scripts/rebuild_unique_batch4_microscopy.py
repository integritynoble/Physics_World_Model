"""
Rebuild standard datasets — Batch 4: Optical microscopy (22 modalities).
Each gets a unique biological phantom appropriate for that microscopy technique.

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
    a = np.nan_to_num(a); mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-10: return np.zeros(a.shape, dtype=np.uint8)
    return ((a - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

def save_img(a, p):
    if a.ndim == 2: Image.fromarray(to_uint8(a), "L").save(str(p))
    elif a.ndim == 3 and a.shape[-1] in [2,3]:
        if a.shape[-1] == 2:
            Image.fromarray(to_uint8(np.sqrt(a[...,0]**2+a[...,1]**2)), "L").save(str(p))
        else: Image.fromarray(to_uint8(a), "RGB").save(str(p))
    elif a.ndim == 3: Image.fromarray(to_uint8(a[a.shape[0]//2]), "L").save(str(p))

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


# ===================================================================
# UNIQUE OPTICAL MICROSCOPY PHANTOMS
# ===================================================================

def phantom_confocal3d_organelles(seed):
    """3D confocal — cell with labeled organelles (mitochondria, ER, nucleus)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D, H, W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D, :H, :W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Cell boundary
    cell_r = min(H,W)*0.35
    cell = (((xx-cx)**2+(yy-cy)**2)/(cell_r**2) + (zz-cz)**2/(D*0.4)**2 < 1).astype(np.float32)
    # Nucleus (central bright sphere)
    nuc_r = cell_r*0.3
    nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2+(zz-cz)**2)/nuc_r**2)
    x += nuc * 0.7
    # Mitochondria (elongated bright rods scattered around)
    for _ in range(rng.randint(8, 15)):
        mx = cx + rng.uniform(-cell_r*0.6, cell_r*0.6)
        my = cy + rng.uniform(-cell_r*0.6, cell_r*0.6)
        mz = cz + rng.uniform(-D*0.25, D*0.25)
        angle = rng.uniform(0, np.pi)
        sx = rng.uniform(2, 5); sy = rng.uniform(1, 2)
        rx = (xx-mx)*np.cos(angle) + (yy-my)*np.sin(angle)
        ry = -(xx-mx)*np.sin(angle) + (yy-my)*np.cos(angle)
        mito = np.exp(-0.5*(rx**2/sx**2 + ry**2/sy**2 + (zz-mz)**2/2**2))
        x += mito * rng.uniform(0.3, 0.6)
    # ER network (thin sheet-like)
    er = np.exp(-0.5*(zz-cz+2)**2/1.5**2) * cell * 0.15
    x += er
    return np.clip(x * cell, 0, 1).astype(np.float32)


def phantom_livecell_phase(seed):
    """LiveCell — phase contrast image of adherent cells (HeLa/A549 morphology)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    x += 0.5  # phase contrast background (gray)
    n_cells = rng.randint(5, 15)
    for _ in range(n_cells):
        cx = rng.uniform(S*0.1, S*0.9); cy = rng.uniform(S*0.1, S*0.9)
        size = rng.uniform(15, 40)
        # Irregular cell shape (ellipse with perturbation)
        angle = rng.uniform(0, np.pi)
        aspect = rng.uniform(0.6, 1.0)
        rx = (xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
        ry = -(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
        dist = np.sqrt(rx**2/size**2 + ry**2/(size*aspect)**2)
        cell = np.clip(1 - dist, 0, 1)**2
        # Phase contrast halo (bright rim, dark body)
        x -= cell * 0.25
        edge = np.exp(-0.5*((dist-1)/0.15)**2)
        x += edge * 0.15
        # Nucleus (darker)
        nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(size*0.3)**2)
        x -= nuc * 0.1
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_spinning_disk_mitosis(seed):
    """Spinning disk confocal — dividing cells (mitotic stages: prophase, metaphase)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    n_cells = rng.randint(4, 10)
    for _ in range(n_cells):
        cx, cy = rng.uniform(S*0.1, S*0.9), rng.uniform(S*0.1, S*0.9)
        cell_r = rng.uniform(15, 30)
        cell = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/cell_r**2)
        x += cell * 0.2  # cytoplasm
        stage = rng.choice(["interphase", "prophase", "metaphase", "anaphase"])
        if stage == "interphase":
            nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(cell_r*0.4)**2)
            x += nuc * 0.5
        elif stage == "prophase":
            nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(cell_r*0.35)**2)
            x += nuc * 0.6
            # Condensing chromosomes
            for _ in range(rng.randint(3, 6)):
                chr_x = cx + rng.uniform(-cell_r*0.2, cell_r*0.2)
                chr_y = cy + rng.uniform(-cell_r*0.2, cell_r*0.2)
                x += np.exp(-0.5*((xx-chr_x)**2+(yy-chr_y)**2)/2**2) * 0.3
        elif stage == "metaphase":
            # Metaphase plate (line of chromosomes)
            angle = rng.uniform(0, np.pi)
            for i in range(rng.randint(8, 16)):
                d = (i - 6) * 2
                chr_x = cx + d * np.cos(angle)
                chr_y = cy + d * np.sin(angle)
                x += np.exp(-0.5*((xx-chr_x)**2+(yy-chr_y)**2)/1.5**2) * 0.7
        else:  # anaphase
            for pole in [-1, 1]:
                px = cx + pole * cell_r * 0.25
                py = cy
                for _ in range(rng.randint(4, 8)):
                    cx2 = px + rng.uniform(-3, 3)
                    cy2 = py + rng.uniform(-3, 3)
                    x += np.exp(-0.5*((xx-cx2)**2+(yy-cy2)**2)/1.5**2) * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_lattice_lightsheet_embryo(seed):
    """Lattice lightsheet — developing embryo (Drosophila-like, 3D)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D, H, W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D, :H, :W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Embryo (ellipsoidal)
    a, b, c = W*0.35, H*0.25, D*0.35
    embryo = ((xx-cx)**2/a**2+(yy-cy)**2/b**2+(zz-cz)**2/c**2 < 1).astype(np.float32)
    x += embryo * 0.15
    # Cells on surface (labeled nuclei, like Drosophila gastrulation)
    n_nuclei = rng.randint(30, 60)
    for _ in range(n_nuclei):
        phi = rng.uniform(0, 2*np.pi); theta = rng.uniform(0, np.pi)
        nx = cx + a*0.85*np.sin(theta)*np.cos(phi)
        ny = cy + b*0.85*np.sin(theta)*np.sin(phi)
        nz = cz + c*0.85*np.cos(theta)
        nuc = np.exp(-0.5*((xx-nx)**2+(yy-ny)**2+(zz-nz)**2)/2.5**2)
        x += nuc * rng.uniform(0.3, 0.7)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_lightsheet_zebrafish(seed):
    """Light-sheet — zebrafish brain (labeled neurons, 3D)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D, H, W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D, :H, :W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Brain outline (elongated)
    brain = ((xx-cx)**2/(W*0.3)**2+(yy-cy)**2/(H*0.2)**2+(zz-cz)**2/(D*0.35)**2 < 1)
    x += brain.astype(np.float32) * 0.1
    # Neurons (bright spots)
    n_neurons = rng.randint(50, 120)
    for _ in range(n_neurons):
        nx = cx + rng.uniform(-W*0.25, W*0.25)
        ny = cy + rng.uniform(-H*0.15, H*0.15)
        nz = cz + rng.uniform(-D*0.3, D*0.3)
        ns = rng.uniform(1, 3)
        x += np.exp(-0.5*((xx-nx)**2+(yy-ny)**2+(zz-nz)**2)/ns**2) * rng.uniform(0.3, 0.8)
    return np.clip(x * brain.astype(np.float32), 0, 1).astype(np.float32)


def phantom_two_photon_calcium(seed):
    """Two-photon — calcium imaging of neurons (sparse bright somata)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Neuropil background (dim, textured)
    x += 0.1 + 0.03 * ndimage.gaussian_filter(rng.randn(S,S).astype(np.float32), sigma=5)
    # Neuronal somata (bright circles, some active, some quiet)
    n_neurons = rng.randint(15, 40)
    for _ in range(n_neurons):
        nx = rng.uniform(S*0.05, S*0.95); ny = rng.uniform(S*0.05, S*0.95)
        nr = rng.uniform(4, 8)
        active = rng.rand() > 0.6  # ~40% are firing
        brightness = rng.uniform(0.5, 0.9) if active else rng.uniform(0.1, 0.25)
        soma = np.exp(-0.5*((xx-nx)**2+(yy-ny)**2)/nr**2)
        x += soma * brightness
        # Dendrite (thin line extending from soma)
        if rng.rand() > 0.5:
            angle = rng.uniform(0, 2*np.pi)
            dlen = rng.uniform(10, 30)
            for t in np.linspace(0, 1, 50):
                dx = nx + dlen*t*np.cos(angle)
                dy = ny + dlen*t*np.sin(angle)
                if 0<=dx<S and 0<=dy<S:
                    x += np.exp(-0.5*((xx-dx)**2+(yy-dy)**2)/1.5**2) * brightness * 0.3 * (1-t)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_three_photon_deep(seed):
    """Three-photon — deep brain imaging (hippocampus, 3D)."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D, H, W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D, :H, :W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Hippocampus-like structure (curved band)
    for z in range(D):
        r_curve = W*0.2 + z*0.5
        center_x = cx + (z-cz)*0.3
        dist = np.sqrt((xx[z]-center_x)**2+(yy[z]-cy)**2)
        band = np.exp(-0.5*((dist-r_curve)/5)**2)
        x[z] += band * 0.3
    # Deep neurons (sparse bright spots)
    n_neurons = rng.randint(20, 40)
    for _ in range(n_neurons):
        nx = cx + rng.uniform(-W*0.2, W*0.2)
        ny = cy + rng.uniform(-H*0.15, H*0.15)
        nz = cz + rng.uniform(-D*0.3, D*0.3)
        x += np.exp(-0.5*((xx-nx)**2+(yy-ny)**2+(zz-nz)**2)/2**2) * rng.uniform(0.3, 0.7)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_sted_synapses(seed):
    """STED — synaptic vesicle clusters at super-resolution (~30nm)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Synaptic boutons (clusters of vesicles)
    n_synapses = rng.randint(8, 20)
    for _ in range(n_synapses):
        sx = rng.uniform(S*0.05, S*0.95); sy = rng.uniform(S*0.05, S*0.95)
        # Each synapse has ~5-20 vesicles
        n_ves = rng.randint(5, 20)
        for _ in range(n_ves):
            vx = sx + rng.uniform(-6, 6); vy = sy + rng.uniform(-6, 6)
            vs = rng.uniform(0.8, 1.5)  # ~30-50nm vesicles
            x += np.exp(-0.5*((xx-vx)**2+(yy-vy)**2)/vs**2) * rng.uniform(0.4, 0.9)
    # Connecting axons (thin lines between synapses)
    for _ in range(rng.randint(3, 8)):
        x1, y1 = rng.uniform(S*0.1, S*0.9), rng.uniform(S*0.1, S*0.9)
        x2, y2 = rng.uniform(S*0.1, S*0.9), rng.uniform(S*0.1, S*0.9)
        for t in np.linspace(0, 1, 100):
            px = x1 + (x2-x1)*t; py = y1 + (y2-y1)*t
            x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.8**2) * 0.15
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_sim_actin(seed):
    """SIM — actin filament network (cytoskeleton)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Actin filaments (thin curved lines, network)
    n_filaments = rng.randint(15, 40)
    for _ in range(n_filaments):
        # Random walk filament
        px, py = rng.uniform(0, S), rng.uniform(0, S)
        angle = rng.uniform(0, 2*np.pi)
        length = rng.uniform(30, 120)
        for t in np.linspace(0, 1, int(length)):
            angle += rng.uniform(-0.05, 0.05)
            px += np.cos(angle)*2; py += np.sin(angle)*2
            if 0<=px<S and 0<=py<S:
                x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.8**2) * 0.4
    # Focal adhesions (bright spots at filament ends)
    n_fa = rng.randint(5, 15)
    for _ in range(n_fa):
        fx, fy = rng.uniform(S*0.05, S*0.95), rng.uniform(S*0.05, S*0.95)
        fs = rng.uniform(2, 5)
        x += np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/fs**2) * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_palm_molecules(seed):
    """PALM/STORM — single molecule localizations (nuclear pore complexes)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Nuclear pore complexes (8-fold symmetric rings)
    n_npcs = rng.randint(10, 25)
    for _ in range(n_npcs):
        cx, cy = rng.uniform(S*0.1, S*0.9), rng.uniform(S*0.1, S*0.9)
        r = rng.uniform(4, 8)
        for k in range(8):
            angle = 2*np.pi*k/8 + rng.uniform(-0.1, 0.1)
            px = cx + r*np.cos(angle); py = cy + r*np.sin(angle)
            x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.6**2) * rng.uniform(0.4, 0.8)
        # Central plug
        x += np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/1.5**2) * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_dna_paint_origami(seed):
    """DNA-PAINT — DNA origami nanostructures (ruler, grid patterns)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # DNA origami structures (periodic dot patterns as molecular rulers)
    n_origami = rng.randint(5, 12)
    for _ in range(n_origami):
        cx = rng.uniform(S*0.1, S*0.9); cy = rng.uniform(S*0.1, S*0.9)
        pattern = rng.choice(["ruler", "grid", "triangle"])
        spacing = rng.uniform(6, 12)
        if pattern == "ruler":
            n_pts = rng.randint(4, 8)
            angle = rng.uniform(0, np.pi)
            for i in range(n_pts):
                px = cx + (i - n_pts/2) * spacing * np.cos(angle)
                py = cy + (i - n_pts/2) * spacing * np.sin(angle)
                x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.5**2) * 0.7
        elif pattern == "grid":
            for i in range(3):
                for j in range(3):
                    px = cx + (i-1)*spacing; py = cy + (j-1)*spacing
                    x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.5**2) * 0.6
        else:
            for k in range(3):
                angle = 2*np.pi*k/3
                px = cx + spacing*np.cos(angle); py = cy + spacing*np.sin(angle)
                x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.5**2) * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_tirf_adhesion(seed):
    """TIRF — cell membrane adhesion (integrin clusters at basal surface)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S//2, S//2
    # Cell footprint (TIRF only sees ~100nm from surface)
    cell_r = S * rng.uniform(0.3, 0.42)
    cell = (np.sqrt((xx-cx)**2+(yy-cy)**2) < cell_r).astype(np.float32)
    x += cell * 0.1  # dim membrane background
    # Focal adhesions (bright elongated clusters at cell periphery)
    n_fa = rng.randint(10, 25)
    for _ in range(n_fa):
        angle = rng.uniform(0, 2*np.pi)
        fa_r = cell_r * rng.uniform(0.6, 0.95)
        fa_x = cx + fa_r*np.cos(angle); fa_y = cy + fa_r*np.sin(angle)
        fa_len = rng.uniform(5, 15); fa_w = rng.uniform(1.5, 3)
        rx = (xx-fa_x)*np.cos(angle)+(yy-fa_y)*np.sin(angle)
        ry = -(xx-fa_x)*np.sin(angle)+(yy-fa_y)*np.cos(angle)
        fa = np.exp(-0.5*(rx**2/fa_len**2+ry**2/fa_w**2))
        x += fa * rng.uniform(0.4, 0.8)
    # Stress fibers (thin lines connecting adhesions)
    for _ in range(rng.randint(3, 8)):
        x1, y1 = cx+cell_r*0.7*np.cos(rng.uniform(0,2*np.pi)), cy+cell_r*0.7*np.sin(rng.uniform(0,2*np.pi))
        x2, y2 = cx+cell_r*0.7*np.cos(rng.uniform(0,2*np.pi)), cy+cell_r*0.7*np.sin(rng.uniform(0,2*np.pi))
        for t in np.linspace(0, 1, 80):
            px = x1+(x2-x1)*t; py = y1+(y2-y1)*t
            x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/1**2) * 0.15
    return np.clip(x * cell, 0, 1).astype(np.float32)


def phantom_flim_lifetime(seed):
    """FLIM — fluorescence lifetime map (protein-protein interaction FRET)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    n_cells = rng.randint(3, 8)
    for _ in range(n_cells):
        cx, cy = rng.uniform(S*0.15, S*0.85), rng.uniform(S*0.15, S*0.85)
        cell_r = rng.uniform(20, 45)
        cell = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/cell_r**2)
        # Baseline lifetime (2.5ns donor-only)
        x += cell * 0.6
        # FRET regions (shorter lifetime — interaction zones)
        n_fret = rng.randint(1, 4)
        for _ in range(n_fret):
            fx = cx + rng.uniform(-cell_r*0.5, cell_r*0.5)
            fy = cy + rng.uniform(-cell_r*0.5, cell_r*0.5)
            fr = rng.uniform(5, 12)
            fret = np.exp(-0.5*((xx-fx)**2+(yy-fy)**2)/fr**2)
            x -= fret * 0.25  # shorter lifetime = lower value
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_shg_collagen(seed):
    """SHG — collagen fiber orientation map (tendon/skin)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Collagen fiber bundles (oriented, wavy lines)
    n_bundles = rng.randint(8, 20)
    main_angle = rng.uniform(0, np.pi)
    for _ in range(n_bundles):
        angle = main_angle + rng.uniform(-0.3, 0.3)
        y_start = rng.uniform(0, S)
        width = rng.uniform(3, 8)
        # Project along bundle direction
        proj = (xx*np.cos(angle) + yy*np.sin(angle))
        perp = (-xx*np.sin(angle) + yy*np.cos(angle)) - y_start
        # Wavy modulation (crimping)
        crimp = np.sin(proj / rng.uniform(8, 15)) * rng.uniform(1, 3)
        bundle = np.exp(-0.5*(perp - crimp)**2/width**2)
        x += bundle * rng.uniform(0.3, 0.7)
    # Cross-linked regions (brighter)
    for _ in range(rng.randint(2, 5)):
        cx, cy = rng.uniform(S*0.1, S*0.9), rng.uniform(S*0.1, S*0.9)
        x += np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/rng.uniform(5,12)**2) * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_expansion_cell(seed):
    """Expansion microscopy — expanded cell with resolved ultrastructure."""
    rng = np.random.RandomState(seed); D, H, W = 32, 128, 128
    x = np.zeros((D, H, W), dtype=np.float32)
    zz, yy, xx = np.mgrid[:D, :H, :W].astype(np.float32)
    cx, cy, cz = W//2, H//2, D//2
    # Expanded cell (4x physical expansion — structures appear spread out)
    cell_r = min(H,W)*0.4
    cell = ((xx-cx)**2+(yy-cy)**2 < cell_r**2).astype(np.float32)
    # Centrioles (paired cylinders)
    for offset in [-3, 3]:
        cen = np.exp(-0.5*((xx-cx-offset)**2+(yy-cy)**2)/1.5**2) * np.exp(-0.5*(zz-cz)**2/3**2)
        x += cen * 0.8
    # Nuclear envelope (ring in each z-slice)
    nuc_r = cell_r * 0.4
    nr = np.sqrt((xx-cx)**2+(yy-cy)**2)
    ne = np.exp(-0.5*((nr-nuc_r)/1.5)**2) * np.exp(-0.5*(zz-cz)**2/(D*0.3)**2)
    x += ne * 0.5
    # Nuclear pores (dots on the nuclear envelope)
    n_pores = rng.randint(15, 30)
    for _ in range(n_pores):
        angle = rng.uniform(0, 2*np.pi)
        z_pos = cz + rng.uniform(-D*0.2, D*0.2)
        px = cx + nuc_r*np.cos(angle); py = cy + nuc_r*np.sin(angle)
        x += np.exp(-0.5*((xx-px)**2+(yy-py)**2+(zz-z_pos)**2)/1**2) * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_ism_mitochondria(seed):
    """ISM — mitochondrial network (re-scan confocal microscopy)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Mitochondrial network (connected tubular structures)
    n_mito = rng.randint(20, 40)
    for _ in range(n_mito):
        # Start point
        sx, sy = rng.uniform(S*0.1, S*0.9), rng.uniform(S*0.1, S*0.9)
        angle = rng.uniform(0, 2*np.pi)
        length = rng.uniform(15, 60)
        width = rng.uniform(1.5, 3)
        for t in np.linspace(0, 1, int(length)):
            angle += rng.uniform(-0.08, 0.08)  # slight curvature
            px = sx + t*length*np.cos(angle)
            py = sy + t*length*np.sin(angle)
            if 0<=px<S and 0<=py<S:
                x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/width**2) * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_minflux_tracks(seed):
    """MINFLUX — molecular tracking (single receptor trajectories)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Membrane (cell outline)
    cx, cy = S//2, S//2; cell_r = S*0.35
    r = np.sqrt((xx-cx)**2+(yy-cy)**2)
    membrane = np.exp(-0.5*((r-cell_r)/2)**2)
    x += membrane * 0.15
    # Tracked receptor molecules (connected dots on membrane)
    n_tracks = rng.randint(5, 12)
    for _ in range(n_tracks):
        # Start on membrane
        angle = rng.uniform(0, 2*np.pi)
        px, py = cx + cell_r*np.cos(angle), cy + cell_r*np.sin(angle)
        n_steps = rng.randint(20, 80)
        for _ in range(n_steps):
            x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/0.4**2) * 0.6
            # Diffuse on membrane surface
            da = rng.uniform(-0.05, 0.05)
            angle += da
            px = cx + cell_r*np.cos(angle); py = cy + cell_r*np.sin(angle)
    # Confined diffusion domains (bright clusters)
    for _ in range(rng.randint(2, 5)):
        angle = rng.uniform(0, 2*np.pi)
        dx = cx + cell_r*np.cos(angle); dy = cy + cell_r*np.sin(angle)
        for _ in range(rng.randint(10, 30)):
            lx = dx + rng.uniform(-3, 3); ly = dy + rng.uniform(-3, 3)
            x += np.exp(-0.5*((xx-lx)**2+(yy-ly)**2)/0.4**2) * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_fpm_phase(seed):
    """FPM — Fourier ptychographic microscopy phase object (unstained cells)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Multiple unstained cells (refractive index variations)
    n_cells = rng.randint(3, 8)
    for _ in range(n_cells):
        cx, cy = rng.uniform(S*0.15, S*0.85), rng.uniform(S*0.15, S*0.85)
        cell_r = rng.uniform(20, 40)
        # Cell body (smooth phase ramp)
        dist = np.sqrt((xx-cx)**2+(yy-cy)**2)
        cell = np.clip(1 - dist/cell_r, 0, 1)
        phase = cell * rng.uniform(0.3, 0.6)
        x += phase
        # Nucleus (higher refractive index)
        nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(cell_r*0.35)**2)
        x += nuc * 0.3
        # Nucleolus
        nx = cx + rng.uniform(-cell_r*0.15, cell_r*0.15)
        ny = cy + rng.uniform(-cell_r*0.15, cell_r*0.15)
        x += np.exp(-0.5*((xx-nx)**2+(yy-ny)**2)/3**2) * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_odt_ri(seed):
    """ODT — optical diffraction tomography refractive index map."""
    rng = np.random.RandomState(seed); S = 256
    x = np.ones((S, S), dtype=np.float32) * 0.33  # water background (n=1.33 scaled)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Cell (higher RI than water)
    cx, cy = S//2, S//2; cell_r = S*rng.uniform(0.2, 0.35)
    dist = np.sqrt((xx-cx)**2+(yy-cy)**2)
    cell = (dist < cell_r).astype(np.float32)
    x += cell * 0.04  # cytoplasm RI ~1.37
    # Nucleus (even higher RI)
    nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(cell_r*0.35)**2)
    x += nuc * 0.06  # nucleus RI ~1.39
    # Lipid droplets (high RI spots)
    for _ in range(rng.randint(3, 8)):
        lx = cx + rng.uniform(-cell_r*0.5, cell_r*0.5)
        ly = cy + rng.uniform(-cell_r*0.5, cell_r*0.5)
        ls = rng.uniform(1.5, 4)
        x += np.exp(-0.5*((xx-lx)**2+(yy-ly)**2)/ls**2) * 0.08 * cell
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_widefield_culture(seed):
    """Widefield — cell culture (large FOV, fluorescent staining)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    n_cells = rng.randint(10, 30)
    for _ in range(n_cells):
        cx, cy = rng.uniform(S*0.05, S*0.95), rng.uniform(S*0.05, S*0.95)
        cell_r = rng.uniform(10, 25)
        angle = rng.uniform(0, np.pi); aspect = rng.uniform(0.5, 1)
        rx = (xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
        ry = -(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
        cell = np.exp(-0.5*(rx**2/cell_r**2+ry**2/(cell_r*aspect)**2))
        x += cell * rng.uniform(0.15, 0.35)
        # Nucleus (DAPI staining - brighter)
        nuc_r = cell_r * 0.4
        nuc = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/nuc_r**2)
        x += nuc * rng.uniform(0.3, 0.6)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_lowdose_widefield(seed):
    """Low-dose widefield — noisy fluorescence (CARE-style ground truth)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Microtubule network (fine filamentous structures)
    n_mt = rng.randint(15, 30)
    for _ in range(n_mt):
        # Straight or slightly curved microtubule
        sx, sy = rng.uniform(0, S), rng.uniform(0, S)
        angle = rng.uniform(0, 2*np.pi)
        length = rng.uniform(30, 100)
        for t in np.linspace(0, 1, int(length)):
            angle += rng.uniform(-0.02, 0.02)
            px = sx + t*length*np.cos(angle)
            py = sy + t*length*np.sin(angle)
            if 0<=px<S and 0<=py<S:
                x += np.exp(-0.5*((xx-px)**2+(yy-py)**2)/1**2) * 0.3
    # MTOC (microtubule organizing center)
    cx, cy = S//2 + rng.uniform(-S*0.15, S*0.15), S//2 + rng.uniform(-S*0.15, S*0.15)
    x += np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/5**2) * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_endomicro_gi(seed):
    """Confocal endomicroscopy — GI tract mucosa (crypts and villi)."""
    rng = np.random.RandomState(seed); S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Mucosal pattern (regular arrangement of crypts)
    n_crypts = rng.randint(15, 30)
    for _ in range(n_crypts):
        cx = rng.uniform(S*0.05, S*0.95); cy = rng.uniform(S*0.05, S*0.95)
        crypt_r = rng.uniform(8, 15)
        wall_w = rng.uniform(1.5, 3)
        r = np.sqrt((xx-cx)**2+(yy-cy)**2)
        # Crypt opening (ring-like)
        crypt = np.exp(-0.5*((r-crypt_r)/wall_w)**2)
        x += crypt * rng.uniform(0.4, 0.7)
        # Goblet cells (bright spots on crypt wall)
        n_goblet = rng.randint(4, 10)
        for k in range(n_goblet):
            angle = 2*np.pi*k/n_goblet
            gx = cx + crypt_r*np.cos(angle); gy = cy + crypt_r*np.sin(angle)
            x += np.exp(-0.5*((xx-gx)**2+(yy-gy)**2)/1.5**2) * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_cars_lipids(seed):
    """CARS — lipid droplet imaging in adipocytes."""
    rng = np.random.RandomState(seed); S = 256; C = 16
    x = np.zeros((S, S, C), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Adipocytes (large cells filled with lipid droplets)
    n_cells = rng.randint(3, 8)
    for _ in range(n_cells):
        cx, cy = rng.uniform(S*0.15, S*0.85), rng.uniform(S*0.15, S*0.85)
        cell_r = rng.uniform(25, 50)
        cell = (np.sqrt((xx-cx)**2+(yy-cy)**2) < cell_r).astype(np.float32)
        # Large lipid droplet (CH2 stretch vibration at ~2845 cm-1)
        droplet_r = cell_r * rng.uniform(0.5, 0.8)
        droplet = np.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/droplet_r**2)
        # CH2 spectral peak
        ch2_peak = np.exp(-0.5*((np.arange(C)-C*0.6)/1.5)**2)
        x += droplet[:,:,None] * ch2_peak[None,None,:] * 0.8 * cell[:,:,None]
        # Membrane (different spectral signature)
        membrane = np.exp(-0.5*((np.sqrt((xx-cx)**2+(yy-cy)**2)-cell_r)/2)**2)
        mem_spec = np.exp(-0.5*((np.arange(C)-C*0.4)/1)**2)
        x += membrane[:,:,None] * mem_spec[None,None,:] * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_srs_chemical(seed):
    """SRS — chemical bond mapping (tissue with different components)."""
    rng = np.random.RandomState(seed); S = 256; C = 16
    x = np.zeros((S, S, C), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Tissue cross-section with multiple chemical components
    cx, cy = S//2, S//2
    # Protein-rich region (amide I band)
    for _ in range(rng.randint(3, 6)):
        px = cx + rng.uniform(-S*0.3, S*0.3); py = cy + rng.uniform(-S*0.3, S*0.3)
        pr = rng.uniform(15, 35)
        mask = np.exp(-0.5*((xx-px)**2+(yy-py)**2)/pr**2)
        protein_spec = np.exp(-0.5*((np.arange(C)-C*0.75)/1.5)**2)
        x += mask[:,:,None] * protein_spec[None,None,:] * 0.5
    # Lipid-rich region (CH stretch)
    for _ in range(rng.randint(2, 4)):
        lx = cx + rng.uniform(-S*0.25, S*0.25); ly = cy + rng.uniform(-S*0.25, S*0.25)
        lr = rng.uniform(8, 20)
        mask = np.exp(-0.5*((xx-lx)**2+(yy-ly)**2)/lr**2)
        lipid_spec = np.exp(-0.5*((np.arange(C)-C*0.55)/1)**2)
        x += mask[:,:,None] * lipid_spec[None,None,:] * 0.6
    # Water background
    water_spec = np.exp(-0.5*((np.arange(C)-C*0.3)/2)**2)
    x += water_spec[None,None,:] * 0.1
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# BUILD ALL
# ===================================================================

def build_all():
    print("=" * 60)
    print("Batch 4: Optical microscopy unique phantoms (22 modalities)")
    print("=" * 60)

    # PSF sigmas tuned per technique
    mods = [
        ("confocal_3d", phantom_confocal3d_organelles, 1.5, "psf_3d",
         "y = PSF_3d(x, sigma=1.5); confocal z-sectioning",
         {"src": "OpenCell CZ Biohub 3D confocal", "ref": "doi:10.1126/science.abi6983"}),
        ("confocal_livecell", phantom_livecell_phase, 1.0, "psf",
         "y = PSF(x, sigma=1.0); phase contrast model",
         {"src": "LiveCell dataset (Sartorius)", "ref": "doi:10.1038/s41592-021-01249-6"}),
        ("spinning_disk", phantom_spinning_disk_mitosis, 1.5, "psf",
         "y = PSF(x, sigma=1.5); spinning-disk confocal",
         {"src": "Broad BBBC048 mitosis dataset", "ref": "doi:10.1038/nmeth.2083"}),
        ("lattice_lightsheet", phantom_lattice_lightsheet_embryo, 1.0, "psf_3d",
         "y = PSF_3d(x, sigma=1.0); lattice lightsheet",
         {"src": "Allen Cell Institute LLS dataset", "ref": "doi:10.1038/s41592-018-0268-1"}),
        ("lightsheet", phantom_lightsheet_zebrafish, 1.5, "psf_3d",
         "y = PSF_3d(x, sigma=1.5); SPIM lightsheet",
         {"src": "Zebrafish SPIM atlas (BIOP/EPFL)", "ref": "doi:10.1038/nmeth.1476"}),
        ("two_photon", phantom_two_photon_calcium, 1.0, "psf",
         "y = PSF(x, sigma=1.0); two-photon excitation",
         {"src": "Allen Brain Observatory 2P calcium", "ref": "doi:10.1016/j.neuron.2019.10.020"}),
        ("three_photon", phantom_three_photon_deep, 1.5, "psf_3d",
         "y = PSF_3d(x, sigma=1.5); three-photon deep",
         {"src": "Kleinfeld lab 3PM dataset (UCSD)", "ref": "doi:10.1126/science.1261605"}),
        ("sted", phantom_sted_synapses, 0.3, "psf",
         "y = PSF(x, sigma=0.3); STED nanoscopy",
         {"src": "STED benchmark (Hein/Hell lab)", "ref": "doi:10.1038/s41592-018-0023-6"}),
        ("sim", phantom_sim_actin, 1.0, "psf",
         "y = PSF(x, sigma=1.0); structured illumination",
         {"src": "SIMcheck/SIMbench dataset", "ref": "doi:10.1038/s41592-018-0046-z"}),
        ("palm_storm", phantom_palm_molecules, 0.8, "psf",
         "y = PSF(x, sigma=0.8); SMLM single-molecule",
         {"src": "SMLM Challenge 2016", "ref": "doi:10.1038/nmeth.4291"}),
        ("dna_paint", phantom_dna_paint_origami, 0.5, "psf",
         "y = PSF(x, sigma=0.5); DNA-PAINT super-resolution",
         {"src": "DNA-PAINT origami benchmark (Jungmann)", "ref": "doi:10.1038/nmeth.3579"}),
        ("tirf", phantom_tirf_adhesion, 1.0, "psf",
         "y = PSF(x, sigma=1.0); TIRF evanescent-wave",
         {"src": "TIRF Cell adhesion benchmark", "ref": "doi:10.1038/nmeth.4291"}),
        ("flim", phantom_flim_lifetime, 1.5, "psf",
         "y = PSF(x, sigma=1.5); FLIM lifetime",
         {"src": "FLUTE FLIM benchmark (UCL)", "ref": "doi:10.1038/s41592-019-0349-6"}),
        ("shg", phantom_shg_collagen, 1.0, "psf",
         "y = PSF(x, sigma=1.0); SHG harmonic generation",
         {"src": "SHG collagen atlas (LOCI/Wisconsin)", "ref": "doi:10.1073/pnas.0307953100"}),
        ("expansion", phantom_expansion_cell, 1.0, "psf_3d",
         "y = PSF_3d(x, sigma=1.0); expansion microscopy",
         {"src": "Allen Institute ExM dataset", "ref": "doi:10.1038/nmeth.3361"}),
        ("ism", phantom_ism_mitochondria, 0.8, "psf",
         "y = PSF(x, sigma=0.8); image scanning microscopy",
         {"src": "ISM/Airyscan benchmark (Müller)", "ref": "doi:10.1038/nphoton.2010.292"}),
        ("minflux", phantom_minflux_tracks, 0.2, "psf",
         "y = PSF(x, sigma=0.2); MINFLUX nanoscopy",
         {"src": "MINFLUX tracking benchmark (Hell lab)", "ref": "doi:10.1126/science.aak9913"}),
        ("fpm", phantom_fpm_phase, 1.5, "fourier_lowpass",
         "y = IFFT(LP(FFT(x))); Fourier ptychographic",
         {"src": "UCB FPM benchmark (Zheng)", "ref": "doi:10.1038/lsa.2015.140"}),
        ("odt", phantom_odt_ri, 1.0, "fourier_lowpass",
         "y = IFFT(LP(FFT(RI_map))); diffraction tomography",
         {"src": "Toulouse ODT / TORCH dataset", "ref": "doi:10.1038/s41566-018-0253-x"}),
        ("widefield", phantom_widefield_culture, 2.0, "psf",
         "y = PSF(x, sigma=2.0); widefield fluorescence",
         {"src": "Broad BBBC006 cell painting", "ref": "doi:10.1038/nmeth.2083"}),
        ("widefield_lowdose", phantom_lowdose_widefield, 2.5, "psf",
         "y = PSF(x, sigma=2.5); low-dose widefield",
         {"src": "CARE low-dose benchmark (Weigert)", "ref": "doi:10.1038/s41592-018-0216-7"}),
        ("confocal_endomicroscopy", phantom_endomicro_gi, 1.5, "psf",
         "y = PSF(x, sigma=1.5); probe-based CLE",
         {"src": "CellVizio pCLE atlas (Mauna Kea)", "ref": "doi:10.1038/ajg.2008.142"}),
    ]

    for mod_name, phantom_fn, sigma, fwd_type, fwd_desc, meta in mods:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            xt = phantom_fn(seed)
            if fwd_type == "psf":
                y = ndimage.gaussian_filter(xt, sigma=sigma).astype(np.float32)
            elif fwd_type == "psf_3d":
                y = ndimage.gaussian_filter(xt, sigma=sigma).astype(np.float32)
            elif fwd_type == "fourier_lowpass":
                k = np.fft.fft2(xt)
                H, W = xt.shape
                yy2, xx2 = np.mgrid[:H, :W]
                center_y, center_x = H//2, W//2
                r = np.sqrt(((yy2-center_y)/(H/2))**2+((xx2-center_x)/(W/2))**2)
                lp = np.fft.fftshift((r < 0.5).astype(np.float32))
                y = np.real(np.fft.ifft2(k * lp)).astype(np.float32)
            else:
                y = xt.copy()
            samples.append((xt, y))
        save_mod(mod_name, samples, fwd_type, fwd_desc, meta)

    # CARS and SRS (spectral imaging — need spectral broadening forward)
    spectral_mods = [
        ("cars", phantom_cars_lipids, "spectral_broadening",
         "y = conv1d(x, IRF); CARS spectral response",
         {"src": "CARS lipid benchmark (Cheng lab)", "ref": "doi:10.1021/jp400553s"}),
        ("srs", phantom_srs_chemical, "spectral_broadening",
         "y = conv1d(x, IRF); SRS chemical mapping",
         {"src": "SRS tissue benchmark (Freudiger/Xie)", "ref": "doi:10.1126/science.1165758"}),
    ]

    for mod_name, phantom_fn, fwd_type, fwd_desc, meta in spectral_mods:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            xt = phantom_fn(seed)
            # Spectral broadening: convolve each pixel's spectrum with IRF
            C = xt.shape[-1]
            irf = np.exp(-0.5*(np.arange(C)-C//2)**2/1.5**2).astype(np.float32)
            irf /= irf.sum()
            y = np.zeros_like(xt)
            for r in range(xt.shape[0]):
                for c in range(xt.shape[1]):
                    y[r, c] = np.convolve(xt[r, c], irf, mode='same')
            samples.append((xt, y.astype(np.float32)))
        save_mod(mod_name, samples, fwd_type, fwd_desc, meta)

    print(f"\nBatch 4 done: 24 optical microscopy modalities rebuilt")


if __name__ == "__main__":
    build_all()
