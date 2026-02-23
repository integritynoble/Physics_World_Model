#!/usr/bin/env python3
"""CACTI Benchmark Test — All 4 Solvers on 6 Scenes.

Tests GAP-TV, PnP-FFDNet, ELP-Unfolding, and EfficientSCI on the full
grayscale benchmark with real checkpoints and data.

Usage:
    /home/spiritai/pwm/Physics_World_Model/.venv/bin/python3 scripts/test_cacti_all_solvers.py
"""
import os
import sys
import time
import warnings
import numpy as np
import scipy.io as sio

warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

import torch

# ---- Metrics ----
def psnr(gt, rec, maxval=1.0):
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(maxval ** 2 / mse)

def ssim_frame(gt, rec):
    from skimage.metrics import structural_similarity
    return structural_similarity(gt, rec, data_range=1.0)

# ---- Data loading ----
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "CACTI", "simulation")
MASK_DIR = os.path.join(PROJECT_ROOT, "datasets", "CACTI", "mask")

SCENE_MAP = {
    "kobe32": ("kobe_cacti.mat", 32),
    "crash32": ("crash32_cacti.mat", 32),
    "aerial32": ("aerial32_cacti.mat", 32),
    "traffic48": ("traffic_cacti.mat", 48),
    "runner40": ("runner8_cacti.mat", 8),   # Smaller version available
    "drop40": ("drop8_cacti.mat", 8),       # Smaller version available
}

def load_scene(scene_name):
    """Load a benchmark scene, returning list of (meas, mask, orig_8frames) groups."""
    fname, total_frames = SCENE_MAP[scene_name]
    path = os.path.join(DATA_DIR, fname)
    data = sio.loadmat(path)

    mask = data["mask"].astype(np.float32)  # (256, 256, 8)
    meas_all = data["meas"].astype(np.float32)  # (256, 256, N_groups)
    orig_all = data["orig"].astype(np.float32)  # (256, 256, total_frames)

    # Normalize to [0, 1]
    if orig_all.max() > 1.5:
        orig_all = orig_all / 255.0
    if meas_all.max() > 10.0:
        meas_all = meas_all / 255.0
    if mask.max() > 1.5:
        mask = mask / mask.max()

    n_groups = meas_all.shape[2]
    n_frames = 8
    groups = []
    for g in range(n_groups):
        y = meas_all[:, :, g]
        gt_start = g * n_frames
        gt_end = gt_start + n_frames
        if gt_end <= orig_all.shape[2]:
            gt = orig_all[:, :, gt_start:gt_end]
            groups.append((y, mask, gt))

    return groups


# ---- Solver wrappers ----
def solve_gap_tv(y, mask, **kwargs):
    """GAP-TV reconstruction."""
    from pwm_core.recon.cacti_solvers import gap_tv_cacti
    return gap_tv_cacti(y, mask, iterations=100, tv_weight=0.1, tv_iter=5)

def solve_pnp_ffdnet(y, mask, device="cpu", **kwargs):
    """PnP-FFDNet reconstruction using original FFDNet model."""
    from pwm_core.recon.cacti_solvers import _gap_denoise_core

    ffdnet_pkg = "/home/spiritai/PnP-SCI_python-master/packages"
    ffdnet_weights = os.path.join(ffdnet_pkg, "ffdnet", "models", "net_gray.pth")

    if ffdnet_pkg not in sys.path:
        sys.path.insert(0, ffdnet_pkg)
    from ffdnet.models import FFDNet

    dev = torch.device(device)
    net = FFDNet(num_input_channels=1)
    state_dict = torch.load(ffdnet_weights, map_location=dev, weights_only=False)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(cleaned, strict=True)
    net = net.to(dev)
    net.eval()

    # GAP + FFDNet denoiser
    sigma_list = [50/255, 25/255, 12/255]
    iter_list = [10, 10, 10]

    h, w, nF = mask.shape
    Phi_sum = np.sum(mask, axis=2)
    Phi_sum[Phi_sum == 0] = 1

    x = y[:, :, np.newaxis] * mask / Phi_sum[:, :, np.newaxis]
    y1 = np.zeros_like(y)

    for sigma, n_iter in zip(sigma_list, iter_list):
        for _ in range(n_iter):
            yb = np.sum(x * mask, axis=2)
            y1 = y1 + (y - yb)
            x = x + ((y1 - yb) / Phi_sum)[:, :, np.newaxis] * mask

            for f in range(nF):
                frame_t = torch.from_numpy(x[:, :, f].copy()).unsqueeze(0).unsqueeze(0).float()
                sigma_t = torch.FloatTensor([sigma])
                if "cuda" in device:
                    frame_t = frame_t.to(dev)
                    sigma_t = sigma_t.to(dev)
                with torch.no_grad():
                    noise_est = net(frame_t, sigma_t)
                x[:, :, f] = (frame_t - noise_est).squeeze().cpu().numpy()

            x = np.clip(x, 0, 1)

    return x.astype(np.float32)

def solve_elp(y, mask, device="cpu", **kwargs):
    """ELP-Unfolding reconstruction using original model."""
    elp_repo = "/home/spiritai/ELP-Unfolding-master"
    elp_ckpt = os.path.join(elp_repo, "trained_dataset", "ckptall.pth")

    if elp_repo not in sys.path:
        sys.path.insert(0, elp_repo)
    from SCI_Modelcollect import SCI_backwardcollect

    dev = torch.device(device)
    argdict = {
        "init_channels": 512,
        "pres_channels": 512,
        "init_input": 8,
        "pres_input": 8,
        "priors": 6,
        "iter__number": 8,
    }
    model = SCI_backwardcollect(argdict).to(dev)
    ckpt = torch.load(elp_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["color_SCI_backward_dict"], strict=False)
    model.eval()

    nF = mask.shape[2]
    H, W = y.shape[:2]

    mask_t = torch.from_numpy(mask.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(dev)
    meas_t = torch.from_numpy(y.copy()).unsqueeze(0).unsqueeze(0).float().to(dev)
    img_out_ori = torch.ones(1, nF, H, W, device=dev)

    with torch.no_grad():
        x_list, _ = model(mask_t, meas_t, img_out_ori)

    recon = x_list[-1].squeeze(0).clamp(0, 1).cpu().numpy()  # (T, H, W)
    return recon.transpose(1, 2, 0).astype(np.float32)  # -> (H, W, T)

def solve_efficientsci(y, mask, device="cpu", **kwargs):
    """EfficientSCI reconstruction using original model."""
    esci_repo = "/home/spiritai/EfficientSCI-main"
    esci_ckpt = os.path.join(esci_repo, "checkpoints", "efficientsci_base.pth")

    if esci_repo not in sys.path:
        sys.path.insert(0, esci_repo)
    from cacti.models.efficientsci import EfficientSCI

    dev = torch.device(device)
    model = EfficientSCI(in_ch=64, units=8, group_num=4, color_ch=1).to(dev)
    ckpt = torch.load(esci_ckpt, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    nF = mask.shape[2]
    H, W = y.shape[:2]

    Phi = torch.from_numpy(mask.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(dev)
    Phi_s = Phi.sum(dim=1, keepdim=True)
    Phi_s[Phi_s == 0] = 1
    meas_t = torch.from_numpy(y.copy()).unsqueeze(0).unsqueeze(0).float().to(dev)

    with torch.no_grad():
        outputs = model(meas_t, Phi, Phi_s)

    recon = outputs[-1].squeeze(0).clamp(0, 1).cpu().numpy()  # (T, H, W)
    return recon.transpose(1, 2, 0).astype(np.float32)  # -> (H, W, T)


SOLVERS = {
    "GAP-TV": solve_gap_tv,
    "PnP-FFDNet": solve_pnp_ffdnet,
    "ELP-Unfolding": solve_elp,
    "EfficientSCI": solve_efficientsci,
}

# ---- Reference values from README ----
REF_PSNR = {
    "kobe32":    {"GAP-TV": 24.00, "PnP-FFDNet": 30.33, "ELP-Unfolding": 34.08, "EfficientSCI": 35.76},
    "crash32":   {"GAP-TV": 25.40, "PnP-FFDNet": 24.69, "ELP-Unfolding": 29.39, "EfficientSCI": 31.12},
    "aerial32":  {"GAP-TV": 26.13, "PnP-FFDNet": 24.36, "ELP-Unfolding": 30.54, "EfficientSCI": 31.50},
    "traffic48": {"GAP-TV": 21.06, "PnP-FFDNet": 23.88, "ELP-Unfolding": 31.34, "EfficientSCI": 32.29},
    "runner40":  {"GAP-TV": 28.70, "PnP-FFDNet": 32.97, "ELP-Unfolding": 38.17, "EfficientSCI": 41.89},
    "drop40":    {"GAP-TV": 34.42, "PnP-FFDNet": 39.91, "ELP-Unfolding": 40.09, "EfficientSCI": 45.10},
}

REF_SSIM = {
    "kobe32":    {"GAP-TV": 0.7461, "PnP-FFDNet": 0.9253, "ELP-Unfolding": 0.9644, "EfficientSCI": 0.9758},
    "crash32":   {"GAP-TV": 0.8649, "PnP-FFDNet": 0.8332, "ELP-Unfolding": 0.9537, "EfficientSCI": 0.9726},
    "aerial32":  {"GAP-TV": 0.8510, "PnP-FFDNet": 0.8200, "ELP-Unfolding": 0.9398, "EfficientSCI": 0.9542},
    "traffic48": {"GAP-TV": 0.7063, "PnP-FFDNet": 0.8299, "ELP-Unfolding": 0.9623, "EfficientSCI": 0.9691},
    "runner40":  {"GAP-TV": 0.8908, "PnP-FFDNet": 0.9357, "ELP-Unfolding": 0.9744, "EfficientSCI": 0.9868},
    "drop40":    {"GAP-TV": 0.9654, "PnP-FFDNet": 0.9863, "ELP-Unfolding": 0.9798, "EfficientSCI": 0.9950},
}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Results storage
    results_psnr = {}
    results_ssim = {}

    scenes_to_test = ["kobe32", "crash32", "aerial32", "traffic48"]
    # runner40/drop40 have smaller datasets (8 frames only) - test separately

    for scene in scenes_to_test:
        print(f"{'='*60}")
        print(f"Scene: {scene}")
        print(f"{'='*60}")

        groups = load_scene(scene)
        print(f"  Loaded {len(groups)} measurement groups")

        results_psnr[scene] = {}
        results_ssim[scene] = {}

        for solver_name, solver_fn in SOLVERS.items():
            print(f"\n  Solver: {solver_name}")
            group_psnrs = []
            group_ssims = []

            t0 = time.time()
            for gi, (y, mask, gt) in enumerate(groups):
                try:
                    recon = solver_fn(y, mask, device=device)

                    # Compute per-frame PSNR and SSIM
                    nF = gt.shape[2]
                    frame_psnrs = []
                    frame_ssims = []
                    for f in range(nF):
                        gt_f = gt[:, :, f]
                        rec_f = recon[:, :, f]
                        fp = psnr(gt_f, rec_f, maxval=1.0)
                        fs = ssim_frame(gt_f, rec_f)
                        frame_psnrs.append(fp)
                        frame_ssims.append(fs)

                    group_psnr = np.mean(frame_psnrs)
                    group_ssim = np.mean(frame_ssims)
                    group_psnrs.append(group_psnr)
                    group_ssims.append(group_ssim)

                    print(f"    Group {gi}: PSNR={group_psnr:.2f} dB, SSIM={group_ssim:.4f}")
                except Exception as e:
                    print(f"    Group {gi}: ERROR - {e}")
                    group_psnrs.append(0)
                    group_ssims.append(0)

            elapsed = time.time() - t0
            avg_psnr = np.mean(group_psnrs) if group_psnrs else 0
            avg_ssim = np.mean(group_ssims) if group_ssims else 0
            results_psnr[scene][solver_name] = avg_psnr
            results_ssim[scene][solver_name] = avg_ssim

            ref_p = REF_PSNR.get(scene, {}).get(solver_name, "N/A")
            print(f"    Average: PSNR={avg_psnr:.2f} dB (ref: {ref_p}), SSIM={avg_ssim:.4f}, Time={elapsed:.1f}s")

    # Also test runner and drop with available data
    for scene in ["runner40", "drop40"]:
        print(f"\n{'='*60}")
        print(f"Scene: {scene} (8-frame version)")
        print(f"{'='*60}")

        try:
            groups = load_scene(scene)
            print(f"  Loaded {len(groups)} measurement groups")

            results_psnr[scene] = {}
            results_ssim[scene] = {}

            for solver_name, solver_fn in SOLVERS.items():
                print(f"\n  Solver: {solver_name}")
                group_psnrs = []
                group_ssims = []

                t0 = time.time()
                for gi, (y, mask, gt) in enumerate(groups):
                    try:
                        recon = solver_fn(y, mask, device=device)
                        nF = gt.shape[2]
                        frame_psnrs = []
                        frame_ssims = []
                        for f in range(nF):
                            fp = psnr(gt[:, :, f], recon[:, :, f], maxval=1.0)
                            fs = ssim_frame(gt[:, :, f], recon[:, :, f])
                            frame_psnrs.append(fp)
                            frame_ssims.append(fs)
                        gp = np.mean(frame_psnrs)
                        gs = np.mean(frame_ssims)
                        group_psnrs.append(gp)
                        group_ssims.append(gs)
                        print(f"    Group {gi}: PSNR={gp:.2f} dB, SSIM={gs:.4f}")
                    except Exception as e:
                        print(f"    Group {gi}: ERROR - {e}")

                elapsed = time.time() - t0
                avg_psnr = np.mean(group_psnrs) if group_psnrs else 0
                avg_ssim = np.mean(group_ssims) if group_ssims else 0
                results_psnr[scene][solver_name] = avg_psnr
                results_ssim[scene][solver_name] = avg_ssim

                ref_p = REF_PSNR.get(scene, {}).get(solver_name, "N/A")
                print(f"    Average: PSNR={avg_psnr:.2f} dB (ref: {ref_p}), SSIM={avg_ssim:.4f}, Time={elapsed:.1f}s")
        except Exception as e:
            print(f"  SKIP: {e}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("SUMMARY: PSNR (dB)")
    print(f"{'='*80}")
    header = f"{'Scene':<15}"
    for s in SOLVERS:
        header += f"{'  ' + s:>18}"
    print(header)
    print("-" * 80)

    solver_avgs_psnr = {s: [] for s in SOLVERS}
    solver_avgs_ssim = {s: [] for s in SOLVERS}

    for scene in results_psnr:
        row = f"{scene:<15}"
        for s in SOLVERS:
            val = results_psnr[scene].get(s, 0)
            ref = REF_PSNR.get(scene, {}).get(s, 0)
            row += f"  {val:6.2f} ({ref:5.2f})"
            if scene in ["kobe32", "crash32", "aerial32", "traffic48"]:
                solver_avgs_psnr[s].append(val)
                solver_avgs_ssim[s].append(results_ssim[scene].get(s, 0))
        print(row)

    # Print averages (only for the 4 main scenes with matching ref data)
    row = f"{'Avg(4 scenes)':<15}"
    for s in SOLVERS:
        avg = np.mean(solver_avgs_psnr[s]) if solver_avgs_psnr[s] else 0
        row += f"  {avg:6.2f}       "
    print(row)

    print(f"\n{'='*80}")
    print("SUMMARY: SSIM")
    print(f"{'='*80}")
    print(header)
    print("-" * 80)
    for scene in results_ssim:
        row = f"{scene:<15}"
        for s in SOLVERS:
            val = results_ssim[scene].get(s, 0)
            ref = REF_SSIM.get(scene, {}).get(s, 0)
            row += f"  {val:.4f}({ref:.4f})"
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
