"""
Build standard cryo-EM dataset with correct forward model.

Cryo-EM forward model:
  x_true = clean 2D projection of 3D protein structure (256x256)
  H = CTF (Contrast Transfer Function) convolution in Fourier domain
  y_ideal = CTF(x_true) = real(IFFT(FFT(x_true) * CTF(k)))

The CTF encodes defocus, spherical aberration, and electron wavelength:
  CTF(k) = -sin(gamma(k))
  gamma(k) = pi * lambda * defocus * |k|^2 - pi/2 * Cs * lambda^3 * |k|^4

We create synthetic protein-like phantoms with recognizable structure
(ring-shaped complexes, channels, asymmetric subunits) at various
viewing angles, similar to TRPV1/ribosome/GroEL/proteasome.

Reference: Frank, J. "Three-Dimensional Electron Microscopy of
           Macromolecular Assemblies" (Oxford, 2006)
           Liao et al., Nature 504:107 (2013) - TRPV1

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "cryo_em" / "standard"
OUTDIR.mkdir(parents=True, exist_ok=True)

N = 256          # image size
N_SAMPLES = 10   # number of samples (different viewing angles / structures)


def make_protein_phantom(seed=42, variant="ring"):
    """
    Create a synthetic 2D protein projection with recognizable structure.
    Mimics common cryo-EM targets: ring complexes, asymmetric particles.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((N, N), dtype=np.float32)
    cx, cy = N // 2, N // 2
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)

    if variant == "ring":
        # Ring-shaped complex (like GroEL, proteasome)
        # Outer ring
        r_outer = rng.uniform(N * 0.25, N * 0.35)
        r_inner = r_outer * rng.uniform(0.4, 0.6)
        ring_width = rng.uniform(N * 0.05, N * 0.08)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        ring = np.exp(-0.5 * ((dist - r_outer) / ring_width)**2)
        img += ring * rng.uniform(0.6, 1.0)

        # Inner ring
        inner_ring = np.exp(-0.5 * ((dist - r_inner) / (ring_width * 0.7))**2)
        img += inner_ring * rng.uniform(0.3, 0.6)

        # Subunits around the ring (7-fold or 4-fold symmetry)
        n_sub = rng.choice([4, 5, 6, 7])
        sub_r = (r_outer + r_inner) / 2
        sub_size = ring_width * 1.2
        angle_offset = rng.uniform(0, 2 * np.pi)
        for k in range(n_sub):
            angle = angle_offset + 2 * np.pi * k / n_sub
            sx = cx + sub_r * np.cos(angle)
            sy = cy + sub_r * np.sin(angle)
            sub = np.exp(-0.5 * ((xx - sx)**2 + (yy - sy)**2) / sub_size**2)
            img += sub * rng.uniform(0.4, 0.8)

        # Central pore
        pore = np.exp(-0.5 * dist**2 / (r_inner * 0.3)**2)
        img += pore * rng.uniform(0.1, 0.3)

    elif variant == "channel":
        # Ion channel (like TRPV1) - tetrameric with transmembrane domains
        r_main = rng.uniform(N * 0.2, N * 0.3)
        # 4 subunits
        for k in range(4):
            angle = np.pi / 4 + np.pi / 2 * k + rng.uniform(-0.1, 0.1)
            sx = cx + r_main * np.cos(angle)
            sy = cy + r_main * np.sin(angle)
            # Each subunit is elongated (transmembrane helix bundle)
            sigma_x = rng.uniform(N * 0.04, N * 0.07)
            sigma_y = rng.uniform(N * 0.06, N * 0.10)
            # Rotate subunit
            dx = xx - sx
            dy = yy - sy
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rx = dx * cos_a + dy * sin_a
            ry = -dx * sin_a + dy * cos_a
            sub = np.exp(-0.5 * (rx**2 / sigma_x**2 + ry**2 / sigma_y**2))
            img += sub * rng.uniform(0.5, 1.0)

        # Central pore (lighter)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        pore = np.exp(-0.5 * dist**2 / (N * 0.05)**2)
        img += pore * rng.uniform(0.2, 0.4)

        # Extracellular domain (top part when viewed from side)
        if rng.rand() > 0.5:
            ey = cy - rng.uniform(N * 0.15, N * 0.25)
            extra = np.exp(-0.5 * ((xx - cx)**2 / (N * 0.15)**2 +
                                    (yy - ey)**2 / (N * 0.08)**2))
            img += extra * rng.uniform(0.3, 0.6)

    elif variant == "asymmetric":
        # Asymmetric particle (like ribosome)
        # Large subunit
        lx = cx + rng.uniform(-N * 0.05, N * 0.05)
        ly = cy + rng.uniform(-N * 0.05, N * 0.05)
        large = np.exp(-0.5 * ((xx - lx)**2 / (N * 0.15)**2 +
                                (yy - ly)**2 / (N * 0.12)**2))
        img += large * 0.8

        # Small subunit
        angle = rng.uniform(0, 2 * np.pi)
        offset = N * 0.15
        sx = lx + offset * np.cos(angle)
        sy = ly + offset * np.sin(angle)
        small = np.exp(-0.5 * ((xx - sx)**2 / (N * 0.09)**2 +
                                (yy - sy)**2 / (N * 0.08)**2))
        img += small * 0.6

        # Internal features
        for _ in range(rng.randint(3, 7)):
            fx = cx + rng.uniform(-N * 0.15, N * 0.15)
            fy = cy + rng.uniform(-N * 0.15, N * 0.15)
            fs = rng.uniform(N * 0.02, N * 0.05)
            feat = np.exp(-0.5 * ((xx - fx)**2 + (yy - fy)**2) / fs**2)
            img += feat * rng.uniform(0.2, 0.5)

    elif variant == "filament":
        # Filament-like structure (like actin, microtubule)
        angle = rng.uniform(-np.pi / 4, np.pi / 4)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        dx = (xx - cx) * cos_a + (yy - cy) * sin_a
        dy = -(xx - cx) * sin_a + (yy - cy) * cos_a
        # Main filament body
        fil = np.exp(-0.5 * dy**2 / (N * 0.04)**2) * (np.abs(dx) < N * 0.35)
        img += fil * 0.7

        # Helical repeat pattern along filament
        period = rng.uniform(N * 0.08, N * 0.15)
        helical = np.cos(2 * np.pi * dx / period) * 0.3
        helical = np.clip(helical, 0, 1)
        img += fil * helical * 0.4

    # Add slight background variation
    bg = ndimage.gaussian_filter(rng.randn(N, N).astype(np.float32), sigma=N * 0.1)
    img += bg * 0.05

    # Normalize
    img = np.clip(img, 0, None)
    img = img / (img.max() + 1e-10)

    # Apply slight Gaussian smoothing (finite resolution of projection)
    img = ndimage.gaussian_filter(img, sigma=1.5)
    img = img / (img.max() + 1e-10)

    return img.astype(np.float32)


def apply_ctf(x_true, defocus_um=2.0, pixel_size_A=1.5, voltage_kV=300.0, cs_mm=2.7):
    """
    Apply CTF (Contrast Transfer Function) in Fourier domain.

    CTF(k) = -sin(gamma(k))
    gamma(k) = pi * lambda * defocus * |k|^2 - pi/2 * Cs * lambda^3 * |k|^4

    This is the standard cryo-EM forward model.
    """
    n = x_true.shape[0]

    # Relativistic electron wavelength (Angstroms)
    kV = voltage_kV * 1000.0
    lam_A = 12.26 / np.sqrt(kV * (1 + 0.978e-6 * kV))

    # Frequency grid (1/Angstrom)
    freqs = np.fft.fftfreq(n, d=pixel_size_A)
    ky, kx = np.meshgrid(freqs, freqs, indexing="ij")
    k2 = kx**2 + ky**2

    # CTF phase
    defocus_A = defocus_um * 1e4
    cs_A = cs_mm * 1e7
    gamma = np.pi * lam_A * defocus_A * k2 - 0.5 * np.pi * cs_A * lam_A**3 * k2**2
    ctf = -np.sin(gamma).astype(np.float64)

    # Apply CTF
    F = np.fft.fft2(x_true.astype(np.float64))
    F_ctf = F * ctf
    y = np.real(np.fft.ifft2(F_ctf)).astype(np.float32)

    return y


def to_uint8(arr):
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_images(x_true, y_ideal, img_dir):
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(x_true), mode="L").save(str(img_dir / "x_true.png"))
    Image.fromarray(to_uint8(y_ideal), mode="L").save(str(img_dir / "y_ideal.png"))


def build():
    print("=" * 60)
    print("Cryo-EM standard dataset builder (proper CTF forward model)")
    print("=" * 60)

    # Remove old files
    for old in OUTDIR.glob("standard_cryo_em_*.h5"):
        old.unlink()

    variants = ["ring", "channel", "asymmetric", "filament",
                "ring", "channel", "asymmetric", "filament",
                "ring", "channel"]
    defocus_values = [1.0, 1.5, 2.0, 2.5, 3.0, 1.2, 1.8, 2.2, 2.8, 3.5]

    print(f"\nBuilding {N_SAMPLES} samples ({N}x{N}) with CTF forward model")
    print(f"  Variants: ring, channel, asymmetric, filament")
    print(f"  Defocus range: 1.0 - 3.5 um")

    out_files = []
    for i in range(N_SAMPLES):
        seed = 42 + i * 137
        variant = variants[i]
        defocus = defocus_values[i]

        x_true = make_protein_phantom(seed=seed, variant=variant)
        y_ideal = apply_ctf(x_true, defocus_um=defocus, pixel_size_A=1.5,
                            voltage_kV=300.0, cs_mm=2.7)

        # Save HDF5
        h5_path = OUTDIR / f"standard_cryo_em_{i:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"] = "ctf_convolution"
            hp.attrs["forward_type"] = "ctf_fourier"
            hp.attrs["defocus_um"] = defocus
            hp.attrs["pixel_size_A"] = 1.5
            hp.attrs["voltage_kV"] = 300.0
            hp.attrs["cs_mm"] = 2.7
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"] = "Synthetic protein phantom (cryo-EM standard)"
            meta.attrs["reference"] = "Frank, 3D-EM of Macromolecular Assemblies (Oxford, 2006)"
            meta.attrs["variant"] = variant
            meta.attrs["sample_index"] = i
            meta.attrs["date_built"] = "2026-03-14"
            meta.attrs["synthetic"] = True

        out_files.append(h5_path.name)

        # Save images
        img_dir = OUTDIR / "images" / f"sample_{i:02d}"
        save_images(x_true, y_ideal, img_dir)

        print(f"  {i:02d}: {variant:12s} defocus={defocus:.1f}um  "
              f"x=[{x_true.min():.3f},{x_true.max():.3f}] "
              f"y=[{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    # Metadata
    meta = {
        "modality": "cryo_em",
        "canonical_dataset": "Synthetic cryo-EM protein phantoms with CTF",
        "source_type": "synthetic",
        "reference": "Frank, Three-Dimensional Electron Microscopy of Macromolecular Assemblies (Oxford, 2006)",
        "n_samples": N_SAMPLES,
        "x_true_shape": [N, N],
        "y_ideal_shape": [N, N],
        "forward_type": "ctf_fourier",
        "forward_model": "y = IFFT(FFT(x_true) * CTF(k)); CTF(k) = -sin(pi*lam*df*k^2 - pi/2*Cs*lam^3*k^4)",
        "pixel_size_A": 1.5,
        "voltage_kV": 300.0,
        "cs_mm": 2.7,
        "defocus_range_um": [1.0, 3.5],
        "noise": "none",
        "mismatch": "none",
        "date_built": "2026-03-14",
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    spec = {
        "modality": "cryo_em",
        "operator": "ctf_convolution",
        "forward_type": "ctf_fourier",
        "pixel_size_A": 1.5,
        "voltage_kV": 300.0,
        "cs_mm": 2.7,
        "noise": "none",
        "mismatch": "none",
        "split": "standard",
    }
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDone: {N_SAMPLES} cryo-EM samples with proper CTF forward model")
    return True


if __name__ == "__main__":
    build()
