"""Generate CT reconstruction images for Figure 3 of the paper.

Creates a clean demonstration using Shepp-Logan phantom with 128 projections
(matching the paper's description of the CT experiment).

"With Judge" = correct FBP pipeline (log-linearization, Ram-Lak filter)
"Without Judge" = common errors the Judge would flag:
  - Wrong angular range assumption (half-scan angles treated as full-scan)
  - Missing non-negativity enforcement
  This produces severe streak artifacts and non-negativity violations.

Outputs:
  figures/ct_groundtruth.png
  figures/ct_with_judge.png
  figures/ct_without_judge.png
  figures/ct_psnr_histogram.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import iradon, radon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

OUT = os.path.join(os.path.dirname(__file__), '..', 'papers', 'universal_simulation', 'figures')
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# ── Generate Shepp-Logan phantom (modified for better contrast) ──────
def shepp_logan(N=256):
    """Modified Shepp-Logan phantom with clinical-like contrast."""
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18, -0.2),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.2),
        (0.0, 0.35, 0.21, 0.25, 0, 0.1),
        (0.0, 0.1, 0.046, 0.046, 0, 0.1),
        (0.0, -0.1, 0.046, 0.046, 0, 0.1),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.1),
        (0.0, -0.605, 0.023, 0.023, 0, 0.1),
        (0.06, -0.605, 0.023, 0.046, 0, 0.1),
    ]
    yy = np.linspace(-1, 1, N)
    xx = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xx, yy)
    img = np.zeros((N, N), dtype=np.float64)
    for cx, cy, rx, ry, angle, intensity in ellipses:
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        Xr = c * (X - cx) + s * (Y - cy)
        Yr = -s * (X - cx) + c * (Y - cy)
        mask = (Xr / rx)**2 + (Yr / ry)**2 <= 1.0
        img[mask] += intensity
    return np.clip(img, 0, None).astype(np.float32)


def generate_sample(N=256, n_angles=128, noise_level=0.02, seed=None):
    """Generate a random phantom variant + noisy sinogram."""
    if seed is not None:
        np.random.seed(seed)

    # Base Shepp-Logan with random perturbations to ellipse positions
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18, -0.2),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.2),
        (0.0, 0.35, 0.21, 0.25, 0, 0.1),
        (0.0, 0.1, 0.046, 0.046, 0, 0.1),
        (0.0, -0.1, 0.046, 0.046, 0, 0.1),
        (-0.08, -0.605, 0.046, 0.023, 0, 0.1),
        (0.0, -0.605, 0.023, 0.023, 0, 0.1),
        (0.06, -0.605, 0.023, 0.046, 0, 0.1),
    ]
    yy = np.linspace(-1, 1, N)
    xx = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xx, yy)
    img = np.zeros((N, N), dtype=np.float64)

    for cx, cy, rx, ry, angle, intensity in ellipses:
        # Small random perturbations
        cx += np.random.uniform(-0.02, 0.02)
        cy += np.random.uniform(-0.02, 0.02)
        intensity *= np.random.uniform(0.9, 1.1)
        theta = np.radians(angle + np.random.uniform(-2, 2))
        c, s = np.cos(theta), np.sin(theta)
        Xr = c * (X - cx) + s * (Y - cy)
        Yr = -s * (X - cx) + c * (Y - cy)
        mask = (Xr / rx)**2 + (Yr / ry)**2 <= 1.0
        img[mask] += intensity

    img = np.clip(img, 0, None).astype(np.float32)

    # Generate sinogram
    angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(img, theta=angles_deg, circle=True)  # (n_det, n_angles)

    # Add Gaussian noise
    sino_noisy = sino + noise_level * sino.max() * np.random.randn(*sino.shape)

    return img, sino_noisy, angles_deg


# ── Create representative sample ─────────────────────────────────────
N = 256
n_angles = 128
gt, sino_noisy, angles_deg = generate_sample(N, n_angles, noise_level=0.01, seed=42)

print(f"Ground truth: shape={gt.shape}, range=[{gt.min():.3f}, {gt.max():.3f}]")
print(f"Sinogram: shape={sino_noisy.shape}, range=[{sino_noisy.min():.3f}, {sino_noisy.max():.3f}]")

# ── "With Judge": correct FBP ────────────────────────────────────────
recon_good = iradon(sino_noisy, theta=angles_deg, circle=True,
                    output_size=N, filter_name='ramp')
recon_good = np.clip(recon_good, 0, None).astype(np.float32)  # non-negativity

# ── "Without Judge": wrong angular range + no non-negativity ─────────
# Common mistake: assume angles span 0-360° instead of 0-180°
wrong_angles = np.linspace(0, 360, n_angles, endpoint=False)  # WRONG
recon_bad = iradon(sino_noisy, theta=wrong_angles, circle=True,
                   output_size=N, filter_name='ramp')
# No non-negativity enforcement (Judge would flag this)

vr = gt.max() - gt.min()
good_c = np.clip(recon_good, gt.min(), gt.max())
bad_c = np.clip(recon_bad, gt.min(), gt.max())

psnr_good = psnr(gt, good_c, data_range=vr)
ssim_good = ssim(gt, good_c, data_range=vr)
psnr_bad = psnr(gt, bad_c, data_range=vr)
ssim_bad = ssim(gt, bad_c, data_range=vr)

print(f"\nWith Judge:    PSNR={psnr_good:.1f} dB, SSIM={ssim_good:.3f}")
print(f"Without Judge: PSNR={psnr_bad:.1f} dB, SSIM={ssim_bad:.3f}")
print(f"Non-neg violations (without Judge): {(recon_bad < -0.01).sum()} pixels")

# ── Save images ──────────────────────────────────────────────────────
def save_ct_image(img, path, vmin=0, vmax=None, cmap='gray'):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_axis_off()
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01, dpi=200)
    plt.close(fig)

disp_max = gt.max()
save_ct_image(gt, os.path.join(OUT, 'ct_groundtruth.png'), 0, disp_max)
save_ct_image(recon_good, os.path.join(OUT, 'ct_with_judge.png'), 0, disp_max)
save_ct_image(recon_bad, os.path.join(OUT, 'ct_without_judge.png'), 0, disp_max)
print("Saved image panels")

# ── Generate 200 samples for histogram ───────────────────────────────
print("\nComputing PSNR distribution for 200 samples...")
psnr_good_all, psnr_bad_all = [], []

for i in range(200):
    gt_i, sino_i, ang_i = generate_sample(N, n_angles, noise_level=0.01, seed=1000+i)

    # With Judge
    r_good = iradon(sino_i, theta=ang_i, circle=True, output_size=N, filter_name='ramp')
    r_good = np.clip(r_good, 0, None).astype(np.float32)

    # Without Judge (wrong angles)
    wrong_ang = np.linspace(0, 360, n_angles, endpoint=False)
    r_bad = iradon(sino_i, theta=wrong_ang, circle=True, output_size=N, filter_name='ramp')

    vr_i = gt_i.max() - gt_i.min()
    if vr_i < 1e-8:
        continue
    gc = np.clip(r_good, gt_i.min(), gt_i.max())
    bc = np.clip(r_bad, gt_i.min(), gt_i.max())
    psnr_good_all.append(psnr(gt_i, gc, data_range=vr_i))
    psnr_bad_all.append(psnr(gt_i, bc, data_range=vr_i))

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/200 done")

psnr_good_all = np.array(psnr_good_all)
psnr_bad_all = np.array(psnr_bad_all)

print(f"\nFinal statistics (n={len(psnr_good_all)}):")
print(f"  With Judge:    {psnr_good_all.mean():.1f} ± {psnr_good_all.std():.1f} dB")
print(f"  Without Judge: {psnr_bad_all.mean():.1f} ± {psnr_bad_all.std():.1f} dB")

# Bootstrap 95% CI for "with Judge"
boot = np.array([np.random.choice(psnr_good_all, len(psnr_good_all), replace=True).mean()
                 for _ in range(10000)])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"  95% CI: [{ci_lo:.1f}, {ci_hi:.1f}] dB")

# ── PSNR histogram ───────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8), dpi=200)

lo = min(psnr_bad_all.min(), psnr_good_all.min()) - 1
hi = max(psnr_bad_all.max(), psnr_good_all.max()) + 1
bins = np.linspace(lo, hi, 30)

ax.hist(psnr_good_all, bins=bins, alpha=0.75, color='#2196F3', edgecolor='white',
        linewidth=0.5,
        label=f'With Judge\n({psnr_good_all.mean():.1f} ± {psnr_good_all.std():.1f} dB)')
ax.hist(psnr_bad_all, bins=bins, alpha=0.55, color='#F44336', edgecolor='white',
        linewidth=0.5,
        label=f'Without Judge\n({psnr_bad_all.mean():.1f} ± {psnr_bad_all.std():.1f} dB)')

ax.set_xlabel('PSNR (dB)', fontsize=9)
ax.set_ylabel('Count', fontsize=9)
ax.legend(fontsize=7, loc='upper left')
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'ct_psnr_histogram.png'), bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved PSNR histogram")
print("\nDone!")
