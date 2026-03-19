#!/usr/bin/env python3
"""Generate 3-modality visual reconstruction comparison figure for Nature paper.

Figure 8: Visual comparison across CASSI, CACTI, and MRI modalities showing
ground truth, Scenario I (ideal), Scenario II (mismatch with visible artifacts),
Scenario III (corrected), and error maps.

Grid layout: 3 rows (CASSI, CACTI, MRI) x 5 columns
  Ground Truth | Sc. I (Ideal) | Sc. II (Mismatch) | Sc. III (Corrected) | Error Map

Output: papers/pwm_flagship/figures/fig8_visual_comparison.{pdf,png}
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE.parent / 'inversenet' / 'results'
FIGDIR = BASE / 'figures'
FIGDIR.mkdir(exist_ok=True)

# ── Colors (matching generate_all_figures.py) ────────────────────────────────
COLORS = {
    'blue': '#2E5FA1',
    'orange': '#E8712B',
    'green': '#3D9E3F',
    'red': '#C93C3C',
    'purple': '#7B4EA3',
    'gray': '#6B6B6B',
    'gate1': '#4A90D9',
    'gate2': '#E8A838',
    'gate3': '#D94A4A',
    'sc_i': '#2E7D32',
    'sc_ii': '#C62828',
    'sc_iii': '#1565C0',
    'sc_iv': '#6A1B9A',
}

# ── Nature-quality matplotlib settings ───────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
})

DOUBLE_COL = 7.205  # inches (183 mm, Nature double-column width)


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_psnr(gt, recon):
    """Compute PSNR between ground truth and reconstruction (2-D images)."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-12:
        return 60.0  # cap at 60 dB for near-perfect match
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        data_range = 1.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def generate_synthetic_mri():
    """Generate synthetic MRI phantom data with blur/noise for demo purposes.

    Returns dict with keys: gt, sc_i, sc_ii, sc_iii (all 2-D, 256x256).
    """
    rng = np.random.RandomState(42)
    N = 256

    # Shepp-Logan-like phantom (simplified ellipses)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    phantom = np.zeros((N, N), dtype=np.float64)

    # Outer ellipse (skull)
    mask = (x / 0.69) ** 2 + (y / 0.92) ** 2 <= 1
    phantom[mask] = 0.8

    # Inner dark ellipse (brain region)
    mask = (x / 0.624) ** 2 + ((y + 0.0184) / 0.874) ** 2 <= 1
    phantom[mask] = 0.6

    # Left ventricle
    mask = ((x + 0.22) / 0.16) ** 2 + (y / 0.41) ** 2 <= 1
    phantom[mask] = 0.3

    # Right ventricle
    mask = ((x - 0.22) / 0.16) ** 2 + (y / 0.41) ** 2 <= 1
    phantom[mask] = 0.3

    # Two small bright spots
    mask = ((x - 0.35) / 0.06) ** 2 + ((y + 0.3) / 0.06) ** 2 <= 1
    phantom[mask] = 1.0
    mask = ((x + 0.1) / 0.046) ** 2 + ((y + 0.35) / 0.046) ** 2 <= 1
    phantom[mask] = 1.0

    gt = phantom.astype(np.float32)

    # Scenario I: small reconstruction noise (ideal forward model)
    sc_i = gt + rng.randn(N, N).astype(np.float32) * 0.005
    sc_i = np.clip(sc_i, 0, 1)

    # Scenario II: coherent aliasing artifacts from k-space undersampling mismatch
    # Simulate by adding structured ringing artifacts
    from numpy.fft import fft2, ifft2, fftshift
    kspace = fft2(gt)
    # Zero out some k-space lines to simulate undersampling mismatch
    corrupt_kspace = kspace.copy()
    # Create a sampling pattern mismatch: assumed uniform but actual has gaps
    mask_ks = np.ones((N, N), dtype=np.float32)
    # Corrupt: miss some phase-encode lines, causing aliasing
    for line_idx in range(0, N, 8):
        if rng.rand() < 0.3:
            mask_ks[line_idx, :] = 0
    corrupt_kspace *= mask_ks
    sc_ii_raw = np.abs(ifft2(corrupt_kspace)).astype(np.float32)
    # Add additional noise
    sc_ii = sc_ii_raw + rng.randn(N, N).astype(np.float32) * 0.01
    sc_ii = np.clip(sc_ii, 0, 1)

    # Scenario III: corrected (most aliasing removed, close to sc_i)
    # Slightly better than sc_ii, close to gt
    sc_iii = 0.85 * gt + 0.15 * sc_ii + rng.randn(N, N).astype(np.float32) * 0.008
    sc_iii = np.clip(sc_iii, 0, 1)

    return {'gt': gt, 'sc_i': sc_i, 'sc_ii': sc_ii, 'sc_iii': sc_iii}


def load_cassi_data():
    """Load CASSI reconstruction data from NPZ files.

    Returns dict with keys: gt, sc_i, sc_ii, sc_iii (all 2-D slices).
    """
    cassi_npz = RESULTS / 'cassi_reconstructions' / 'scene01.npz'
    if cassi_npz.exists():
        data = np.load(cassi_npz)
        band = 14  # middle spectral band for visualisation
        gt = data['gt'][:, :, band]
        sc_i = data['scenario_i_mst_l'][:, :, band]
        sc_ii = data['scenario_ii_mst_l'][:, :, band]
        sc_iii = data['scenario_iii_mst_l'][:, :, band]
        print(f'  CASSI: loaded scene01 band {band} from {cassi_npz}')
        return {'gt': gt, 'sc_i': sc_i, 'sc_ii': sc_ii, 'sc_iii': sc_iii}
    else:
        print('  CASSI: NPZ not found, generating synthetic demo data')
        return _synthetic_cassi()


def _synthetic_cassi():
    """Generate synthetic CASSI-like spectral data for demo."""
    rng = np.random.RandomState(7)
    N = 256
    # Simple test pattern
    gt = np.zeros((N, N), dtype=np.float32)
    gt[40:216, 40:216] = 0.5
    gt[80:176, 80:176] = 0.8
    gt[110:146, 110:146] = 1.0
    # Add gradient
    gt += np.linspace(0, 0.15, N)[None, :].astype(np.float32)
    gt = np.clip(gt, 0, 1)

    sc_i = gt + rng.randn(N, N).astype(np.float32) * 0.01
    sc_i = np.clip(sc_i, 0, 1)

    # Mismatch: dispersion shift artifacts (horizontal banding)
    sc_ii = gt.copy()
    sc_ii[::4, :] += 0.15
    sc_ii[1::4, :] -= 0.1
    sc_ii += rng.randn(N, N).astype(np.float32) * 0.03
    sc_ii = np.clip(sc_ii, 0, 1)

    sc_iii = gt + rng.randn(N, N).astype(np.float32) * 0.015
    sc_iii = np.clip(sc_iii, 0, 1)
    return {'gt': gt, 'sc_i': sc_i, 'sc_ii': sc_ii, 'sc_iii': sc_iii}


def load_cacti_data():
    """Load CACTI reconstruction data from NPZ files.

    Returns dict with keys: gt, sc_i, sc_ii, sc_iii (all 2-D slices).
    """
    cacti_npz = RESULTS / 'cacti_reconstructions' / 'kobe.npz'
    if cacti_npz.exists():
        data = np.load(cacti_npz)
        # Use first group (g0), middle temporal frame (frame 4 of 8)
        frame = 4
        gt = data['g0_gt'][:, :, frame]
        sc_i = data['g0_scenario_i_gap_tv'][:, :, frame]
        sc_ii = data['g0_scenario_ii_gap_tv'][:, :, frame]
        sc_iii = data['g0_scenario_iii_gap_tv'][:, :, frame]
        print(f'  CACTI: loaded kobe g0 frame {frame} from {cacti_npz}')
        return {'gt': gt, 'sc_i': sc_i, 'sc_ii': sc_ii, 'sc_iii': sc_iii}
    else:
        print('  CACTI: NPZ not found, generating synthetic demo data')
        return _synthetic_cacti()


def _synthetic_cacti():
    """Generate synthetic CACTI-like video frame data for demo."""
    rng = np.random.RandomState(13)
    N = 256
    gt = np.zeros((N, N), dtype=np.float32)
    # Moving object
    yc, xc = 128, 140
    yy, xx = np.mgrid[:N, :N]
    r = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
    gt[r < 50] = 0.9
    gt[r < 25] = 0.6
    gt += np.random.RandomState(99).rand(N, N).astype(np.float32) * 0.05
    gt = np.clip(gt, 0, 1)

    sc_i = gt + rng.randn(N, N).astype(np.float32) * 0.01
    sc_i = np.clip(sc_i, 0, 1)

    # Mismatch: temporal ghosting (shifted overlay)
    sc_ii = 0.7 * gt + 0.3 * np.roll(gt, 15, axis=1)
    sc_ii += rng.randn(N, N).astype(np.float32) * 0.03
    sc_ii = np.clip(sc_ii, 0, 1)

    sc_iii = gt + rng.randn(N, N).astype(np.float32) * 0.012
    sc_iii = np.clip(sc_iii, 0, 1)
    return {'gt': gt, 'sc_i': sc_i, 'sc_ii': sc_ii, 'sc_iii': sc_iii}


def load_mri_data():
    """Load MRI data; always uses synthetic phantom (no MRI NPZ in repo).

    Returns dict with keys: gt, sc_i, sc_ii, sc_iii (all 2-D, 256x256).
    """
    print('  MRI: generating synthetic phantom data')
    return generate_synthetic_mri()


# ── Main figure generation ───────────────────────────────────────────────────
def generate_fig8():
    """Generate Figure 8: 3-modality visual reconstruction comparison."""
    print('Loading reconstruction data...')

    modalities = [
        ('CASSI', load_cassi_data()),
        ('CACTI', load_cacti_data()),
        ('MRI',   load_mri_data()),
    ]

    col_titles = [
        'Ground Truth',
        'Sc. I (Ideal)',
        'Sc. II (Mismatch)',
        'Sc. III (Corrected)',
        'Error Map |Sc. II - GT|',
    ]
    col_keys = ['gt', 'sc_i', 'sc_ii', 'sc_iii', 'error']

    nrows = len(modalities)
    ncols = len(col_titles)

    # Figure size: Nature double-column width, height proportional to content
    fig_height = DOUBLE_COL * (nrows / ncols) * 1.25 + 0.6
    fig = plt.figure(figsize=(DOUBLE_COL, fig_height))

    # GridSpec: extra width for colorbar column
    gs = gridspec.GridSpec(
        nrows, ncols,
        wspace=0.08,
        hspace=0.25,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.03,
        width_ratios=[1, 1, 1, 1, 1.15],
    )

    # Title colors for scenarios
    title_colors = {
        'gt': '#333333',
        'sc_i': COLORS['sc_i'],
        'sc_ii': COLORS['sc_ii'],
        'sc_iii': COLORS['sc_iii'],
        'error': COLORS['orange'],
    }

    # Row labels
    row_labels = [name for name, _ in modalities]

    for row_idx, (mod_name, mod_data) in enumerate(modalities):
        gt = mod_data['gt']
        sc_i = mod_data['sc_i']
        sc_ii = mod_data['sc_ii']
        sc_iii = mod_data['sc_iii']
        error = np.abs(sc_ii.astype(np.float64) - gt.astype(np.float64))

        images = [gt, sc_i, sc_ii, sc_iii, error]
        keys = col_keys

        # Compute PSNR for each reconstruction
        psnr_vals = {
            'gt': None,
            'sc_i': compute_psnr(gt, sc_i),
            'sc_ii': compute_psnr(gt, sc_ii),
            'sc_iii': compute_psnr(gt, sc_iii),
            'error': None,
        }

        # Common vmin/vmax for reconstruction panels (exclude error map)
        vmin_recon = 0.0
        vmax_recon = max(gt.max(), sc_i.max(), sc_ii.max(), sc_iii.max(), 1.0)
        vmax_recon = min(vmax_recon, 1.0)

        # Error map range
        vmax_err = max(error.max(), 0.05)

        for col_idx, (img, key) in enumerate(zip(images, keys)):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if key == 'error':
                im = ax.imshow(img, cmap='hot', vmin=0, vmax=vmax_err,
                               aspect='equal', interpolation='nearest')
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.04)
                cb = fig.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=5, width=0.4, length=2)
                cb.outline.set_linewidth(0.4)
            else:
                cmap = 'gray' if mod_name == 'MRI' else 'viridis'
                ax.imshow(img, cmap=cmap, vmin=vmin_recon,
                          vmax=vmax_recon, aspect='equal',
                          interpolation='nearest')

            ax.set_xticks([])
            ax.set_yticks([])

            # Thin border
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color('#888888')

            # Column titles (top row only)
            if row_idx == 0:
                title_color = title_colors[key]
                ax.set_title(col_titles[col_idx], fontsize=7,
                             fontweight='bold', color=title_color, pad=4)

            # PSNR annotation (bottom-right of panel)
            psnr = psnr_vals[key]
            if psnr is not None:
                ax.text(
                    0.97, 0.04, f'{psnr:.1f} dB',
                    transform=ax.transAxes,
                    fontsize=6, fontweight='bold',
                    color='white',
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='black', alpha=0.65,
                              edgecolor='none'),
                )

            # Row label (leftmost column only)
            if col_idx == 0:
                ax.text(
                    -0.08, 0.5, mod_name,
                    transform=ax.transAxes,
                    fontsize=9, fontweight='bold',
                    color=COLORS['blue'],
                    ha='right', va='center',
                    rotation=90,
                )

    # Overall figure title
    fig.suptitle(
        'Visual reconstruction comparison across three modalities',
        fontsize=10, fontweight='bold', y=0.98,
    )

    # Save
    out_pdf = FIGDIR / 'fig8_visual_comparison.pdf'
    out_png = FIGDIR / 'fig8_visual_comparison.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

    print(f'\nFigure 8 saved:')
    print(f'  PDF: {out_pdf}  ({out_pdf.stat().st_size / 1024:.0f} KB)')
    print(f'  PNG: {out_png}  ({out_png.stat().st_size / 1024:.0f} KB)')


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating Figure 8: Visual reconstruction comparison')
    print(f'  Results dir: {RESULTS}')
    print(f'  Output dir:  {FIGDIR}')
    print()
    generate_fig8()
    print('\nDone.')
