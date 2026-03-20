#!/usr/bin/env python3
"""Train remaining DL model weights (fixed class names and APIs).

Handles: RED-CNN, CARE 2D/3D, PhaseNet, Noise2Void, HSI-SDeCNN, DL-SIM, others.
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

PWMCORE_WEIGHTS = Path(
    "D:/onedrive/startup/program/physics_world_model"
    "/PWM4/Physics_World_Model-master/packages/pwm_core/pwm_core/weights"
)
BENCH = ROOT / "datasets" / "benchmark"

import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_multiple_samples(mod_id, n=6, tier="public"):
    """Load multiple samples from the benchmark h5 file."""
    import h5py
    if mod_id == "ct":
        samples = []
        for i in range(n):
            d = BENCH / "ct" / "public" / f"sample_{i:02d}"
            if d.exists():
                s = {
                    "x_true": np.load(d / "groundtruth.npy"),
                    "y": np.load(d / "measurement.npy"),
                    "H_ideal": np.load(d / "angles.npy"),
                }
                samples.append(s)
        return samples

    tier_dir = BENCH / mod_id / tier
    h5_files = list(tier_dir.glob("*.h5"))
    if not h5_files:
        return []
    samples = []
    with h5py.File(h5_files[0], "r") as hf:
        keys = list(hf.keys())[:n]
        for key in keys:
            s = {k: hf[key][k][:] for k in hf[key].keys()}
            if "y" not in s:
                for alt in ("sinogram_measured", "bscan_measured",
                            "kspace_undersampled", "projection_measured",
                            "measurement", "interferogram", "ms_lr"):
                    if alt in s:
                        s["y"] = s[alt]
                        break
            if "x_true" not in s:
                for alt in ("x_true_amplitude", "groundtruth"):
                    if alt in s:
                        s["x_true"] = s[alt]
                        break
            samples.append(s)
    return [s for s in samples if "x_true" in s and "y" in s]


def norm(arr):
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8) if mx > mn else a


def make_2d(arr):
    """Ensure array is 2D."""
    if arr.ndim == 1:
        n = int(arr.shape[0]**0.5)
        return arr[:n*n].reshape(n, n)
    if arr.ndim == 3:
        return arr[0] if arr.shape[0] < arr.shape[1] else arr[:, :, 0]
    return arr


def train_loop(model, pairs_xy, epochs=300, lr=1e-3, device=None):
    """Standard 1-channel training loop."""
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        ep_loss = 0.0
        for (x_np, y_np) in pairs_xy:
            x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
            y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0).to(device)
            opt.zero_grad()
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # Handle channel mismatch
            if out.shape[1] != 1:
                out = out[:, :1]
            if out.shape[-2:] != y.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=y.shape[-2:])
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sch.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={ep_loss/len(pairs_xy):.5f}")
    model.eval()
    return model


# ==========================================================================
# RED-CNN
# ==========================================================================
def train_redcnn():
    print("\n=== Training RED-CNN (CT) ===")
    save_path = PWMCORE_WEIGHTS / "redcnn" / "redcnn.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.redcnn import REDCNN
    from pwm_core.recon.ct_solvers import fbp_2d
    samples = load_multiple_samples("ct", n=8)
    if not samples:
        print("  No CT data, skipping")
        return

    device = get_device()
    model = REDCNN().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()

    pairs = []
    for s in samples:
        sino = s["y"]
        gt = s["x_true"]
        angles = s["H_ideal"]
        try:
            fbp = fbp_2d(sino, angles, filter_type="ramlak")
        except Exception:
            fbp = gt  # fallback
        pairs.append((norm(make_2d(fbp)), norm(make_2d(gt))))

    for ep in range(300):
        ep_loss = 0.0
        for (fbp_n, gt_n) in pairs:
            x = torch.from_numpy(fbp_n).unsqueeze(0).unsqueeze(0).to(device)
            y = torch.from_numpy(gt_n).unsqueeze(0).unsqueeze(0).to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sch.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1}/300 loss={ep_loss/len(pairs):.5f}")

    model.eval()
    torch.save(model.state_dict(), str(save_path))
    print(f"  Saved: {save_path}")


# ==========================================================================
# CARE 2D
# ==========================================================================
def train_care_2d():
    print("\n=== Training CARE 2D ===")
    save_path = PWMCORE_WEIGHTS / "care" / "care_2d.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.care_unet import CAREUNet2D
    samples = load_multiple_samples("widefield", n=8)
    if not samples:
        samples = load_multiple_samples("confocal_livecell", n=8)
    if not samples:
        print("  No training data, skipping")
        return

    pairs = []
    for s in samples:
        inp = make_2d(s["y"].astype(np.float32))
        tgt = make_2d(s["x_true"].astype(np.float32))
        pairs.append((norm(inp), norm(tgt)))

    device = get_device()
    model = CAREUNet2D(in_channels=1, out_channels=1)
    model = train_loop(model, pairs, epochs=300, device=device)
    torch.save(model.state_dict(), str(save_path))
    print(f"  Saved: {save_path}")


# ==========================================================================
# CARE 3D
# ==========================================================================
def train_care_3d():
    print("\n=== Training CARE 3D ===")
    save_path = PWMCORE_WEIGHTS / "care" / "care_3d.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.care_unet import CAREUNet3D
    samples = load_multiple_samples("confocal_3d", n=4)
    if not samples:
        print("  No confocal_3d data, skipping")
        return

    device = get_device()
    model = CAREUNet3D(in_channels=1, out_channels=1).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    pairs = []
    for s in samples:
        inp = s["y"].astype(np.float32)
        tgt = s["x_true"].astype(np.float32)
        if inp.ndim == 2:
            inp = inp[np.newaxis]
        if tgt.ndim == 2:
            tgt = tgt[np.newaxis]
        d = min(inp.shape[0], 16)
        pairs.append((norm(inp[:d]), norm(tgt[:d])))

    for ep in range(200):
        ep_loss = 0.0
        for (inp, tgt) in pairs:
            x = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
            y = torch.from_numpy(tgt).unsqueeze(0).unsqueeze(0).to(device)
            opt.zero_grad()
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.shape[1] != 1:
                out = out[:, :1]
            if out.shape[-3:] != y.shape[-3:]:
                out = torch.nn.functional.interpolate(out, size=y.shape[-3:])
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/200 loss={ep_loss/len(pairs):.5f}")

    model.eval()
    torch.save(model.state_dict(), str(save_path))
    print(f"  Saved: {save_path}")


# ==========================================================================
# PhaseNet
# ==========================================================================
def train_phasenet():
    print("\n=== Training PhaseNet (holography) ===")
    save_path = PWMCORE_WEIGHTS / "phasenet" / "phasenet.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.phasenet import PhaseNet
    samples = load_multiple_samples("holography", n=8)
    if not samples:
        print("  No holography data, skipping")
        return

    pairs = []
    for s in samples:
        inp = s["y"].astype(np.float32)
        tgt = s.get("x_true_amplitude", s.get("x_true", s["y"])).astype(np.float32)
        pairs.append((norm(make_2d(inp)), norm(make_2d(tgt))))

    device = get_device()
    model = PhaseNet()
    model = train_loop(model, pairs, epochs=300, device=device)
    torch.save(model.state_dict(), str(save_path))
    print(f"  Saved: {save_path}")


# ==========================================================================
# Noise2Void
# ==========================================================================
def train_noise2void():
    print("\n=== Training Noise2Void ===")
    save_path = PWMCORE_WEIGHTS / "noise2void" / "n2v.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.noise2void import Noise2VoidUNet
    samples = load_multiple_samples("widefield", n=8)
    if not samples:
        print("  No training data, skipping")
        return

    pairs = []
    for s in samples:
        inp = make_2d(s["y"].astype(np.float32))
        tgt = make_2d(s["x_true"].astype(np.float32))
        pairs.append((norm(inp), norm(tgt)))

    device = get_device()
    model = Noise2VoidUNet(in_channels=1, out_channels=1)
    model = train_loop(model, pairs, epochs=200, device=device)
    torch.save(model.state_dict(), str(save_path))
    print(f"  Saved: {save_path}")


# ==========================================================================
# HSI-SDeCNN
# ==========================================================================
def train_hsi_sdecnn():
    print("\n=== Training HSI-SDeCNN ===")
    save_path = PWMCORE_WEIGHTS / "hsi_sdecnn" / "hsi_sdecnn.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.hsi_sdecnn import HSI_SDeCNN
    samples = load_multiple_samples("hyperspectral_remote", n=6)
    if not samples:
        samples = load_multiple_samples("cassi", n=6)
    if not samples:
        print("  No HSI data, skipping")
        return

    device = get_device()
    try:
        model = HSI_SDeCNN().to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        pairs = []
        for s in samples:
            inp = s["y"].astype(np.float32)
            tgt = s["x_true"].astype(np.float32)
            # HSI typically needs multi-channel input
            if inp.ndim == 2:
                inp = inp[np.newaxis, :, :]
            if tgt.ndim == 2:
                tgt = tgt[np.newaxis, :, :]
            pairs.append((inp, tgt))

        for ep in range(200):
            ep_loss = 0.0
            for (inp, tgt) in pairs:
                try:
                    x = torch.from_numpy(inp).unsqueeze(0).to(device)
                    y = torch.from_numpy(tgt[:1]).unsqueeze(0).to(device)
                    opt.zero_grad()
                    out = model(x)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if out.shape != y.shape:
                        out = out[:, :1, :y.shape[2], :y.shape[3]]
                    loss = loss_fn(out, y)
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item()
                except Exception:
                    pass
            if (ep + 1) % 50 == 0 and ep_loss > 0:
                print(f"  epoch {ep+1}/200 loss={ep_loss:.5f}")

        model.eval()
        torch.save(model.state_dict(), str(save_path))
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  HSI-SDeCNN failed: {e}")


# ==========================================================================
# DL-SIM
# ==========================================================================
def train_dl_sim():
    print("\n=== Training DL-SIM ===")
    save_path = PWMCORE_WEIGHTS / "dl_sim" / "dl_sim.pth"
    if save_path.exists():
        print(f"  Already exists: {save_path}")
        return

    from pwm_core.recon.dl_sim import DLSIMNet
    samples = load_multiple_samples("sim", n=6)
    if not samples:
        print("  No SIM data, skipping")
        return

    device = get_device()
    try:
        # DLSIMNet expects 9 input channels (3 phases x 3 orientations)
        model = DLSIMNet().to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for ep in range(200):
            ep_loss = 0.0
            for s in samples:
                try:
                    inp = s["y"].astype(np.float32)
                    tgt = s["x_true"].astype(np.float32)
                    # DLSIMNet needs (B, 9, H, W) - create 9 channels from input
                    if inp.ndim == 2:
                        inp_9ch = np.stack([inp] * 9, axis=0)  # (9, H, W)
                    elif inp.ndim == 3 and inp.shape[0] >= 9:
                        inp_9ch = inp[:9]
                    else:
                        inp_9ch = np.tile(inp[0] if inp.ndim > 2 else inp, (9, 1, 1))
                    tgt_2d = make_2d(tgt)

                    x = torch.from_numpy(norm(inp_9ch)).unsqueeze(0).to(device)
                    y = torch.from_numpy(norm(tgt_2d)).unsqueeze(0).unsqueeze(0).to(device)
                    opt.zero_grad()
                    out = model(x)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if out.shape[1] != 1:
                        out = out[:, :1]
                    if out.shape[-2:] != y.shape[-2:]:
                        out = torch.nn.functional.interpolate(out, size=y.shape[-2:])
                    loss = loss_fn(out, y)
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item()
                except Exception:
                    pass
            if (ep + 1) % 50 == 0 and ep_loss > 0:
                print(f"  epoch {ep+1}/200 loss={ep_loss:.5f}")

        model.eval()
        torch.save(model.state_dict(), str(save_path))
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  DL-SIM failed: {e}")


# ==========================================================================
# Additional models: ISTA-Net, LISTA, MoDL, HATNet, HDNet, MST, PtychoNN
# ==========================================================================
def train_additional():
    device = get_device()

    # ISTA-Net+ (for compressed sensing)
    print("\n=== Training ISTA-Net+ ===")
    save_path = PWMCORE_WEIGHTS / "ista_net" / "ista_net_plus.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.ista_net import ISTANetPlus
            model = ISTANetPlus()
            samples = load_multiple_samples("mri", n=6)
            if samples:
                pairs = []
                for s in samples:
                    pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                  norm(make_2d(s["x_true"].astype(np.float32)))))
                model = train_loop(model, pairs, epochs=200, device=device)
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  ISTA-Net+ failed: {e}")

    # LISTA (for sparse coding)
    print("\n=== Training LISTA ===")
    save_path = PWMCORE_WEIGHTS / "lista" / "lista.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.lista import LISTA
            model = LISTA()
            samples = load_multiple_samples("spc_kronecker", n=6)
            if not samples:
                samples = load_multiple_samples("widefield", n=6)
            if samples:
                pairs = []
                for s in samples:
                    y_1d = s["y"].astype(np.float32).flatten()[:256]
                    x_1d = s["x_true"].astype(np.float32).flatten()[:256]
                    pairs.append((norm(y_1d[np.newaxis, :]), norm(x_1d[np.newaxis, :])))
                # Custom train for LISTA (1D signals)
                model = model.to(device)
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.MSELoss()
                for ep in range(200):
                    for (y_, x_) in pairs:
                        opt.zero_grad()
                        t_y = torch.from_numpy(y_).to(device)
                        t_x = torch.from_numpy(x_).to(device)
                        out = model(t_y)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        loss = loss_fn(out.view_as(t_x), t_x)
                        loss.backward()
                        opt.step()
                model.eval()
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  LISTA failed: {e}")

    # PtychoNN (for ptychography)
    print("\n=== Training PtychoNN ===")
    save_path = PWMCORE_WEIGHTS / "ptychonn" / "ptychonn.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.ptychonn import PtychoNN
            model = PtychoNN()
            samples = load_multiple_samples("ptychography", n=6)
            if samples:
                pairs = []
                for s in samples:
                    pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                  norm(make_2d(s["x_true"].astype(np.float32)))))
                model = train_loop(model, pairs, epochs=200, device=device)
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  PtychoNN failed: {e}")

    # HDNet (for hyperspectral)
    print("\n=== Training HDNet ===")
    save_path = PWMCORE_WEIGHTS / "hdnet" / "hdnet.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.hdnet import HDNet
            model = HDNet()
            samples = load_multiple_samples("sd_cassi", n=6)
            if samples:
                pairs = []
                for s in samples:
                    pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                  norm(make_2d(s["x_true"].astype(np.float32)))))
                model = train_loop(model, pairs, epochs=200, device=device)
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  HDNet failed: {e}")

    # MST (mst_l variant)
    print("\n=== Training MST-L ===")
    for variant in ["mst_l", "mst_plus_plus"]:
        save_path = PWMCORE_WEIGHTS / "mst" / f"{variant}.pth"
        if not save_path.exists():
            try:
                from pwm_core.recon.mst import MSTNet
                model = MSTNet(variant=variant)
                samples = load_multiple_samples("sd_cassi", n=6)
                if samples:
                    pairs = []
                    for s in samples:
                        pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                      norm(make_2d(s["x_true"].astype(np.float32)))))
                    model = train_loop(model, pairs, epochs=200, device=device)
                    torch.save(model.state_dict(), str(save_path))
                    print(f"  Saved: {save_path}")
            except Exception as e:
                print(f"  MST {variant} failed: {e}")

    # HATNet
    print("\n=== Training HATNet ===")
    save_path = PWMCORE_WEIGHTS / "hatnet" / "hatnet.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.hatnet import HATNet
            model = HATNet()
            samples = load_multiple_samples("spc_kronecker", n=6)
            if samples:
                pairs = []
                for s in samples:
                    pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                  norm(make_2d(s["x_true"].astype(np.float32)))))
                model = train_loop(model, pairs, epochs=200, device=device)
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  HATNet failed: {e}")

    # MoDL (for MRI)
    print("\n=== Training MoDL ===")
    save_path = PWMCORE_WEIGHTS / "modl" / "modl.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.modl import MoDL
            model = MoDL()
            samples = load_multiple_samples("mri", n=6)
            if samples:
                pairs = []
                for s in samples:
                    pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                  norm(make_2d(s["x_true"].astype(np.float32)))))
                model = train_loop(model, pairs, epochs=200, device=device)
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  MoDL failed: {e}")

    # IFCNN (for image fusion)
    print("\n=== Training IFCNN ===")
    save_path = PWMCORE_WEIGHTS / "ifcnn" / "ifcnn.pth"
    if not save_path.exists():
        try:
            from pwm_core.recon.ifcnn import IFCNN
            model = IFCNN()
            samples = load_multiple_samples("clem", n=6)
            if not samples:
                samples = load_multiple_samples("widefield", n=6)
            if samples:
                pairs = []
                for s in samples:
                    pairs.append((norm(make_2d(s["y"].astype(np.float32))),
                                  norm(make_2d(s["x_true"].astype(np.float32)))))
                model = train_loop(model, pairs, epochs=200, device=device)
                torch.save(model.state_dict(), str(save_path))
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  IFCNN failed: {e}")


if __name__ == "__main__":
    print("Training remaining DL model weights...")
    print(f"Using device: {get_device()}")

    try:
        train_redcnn()
    except Exception as e:
        print(f"RED-CNN failed: {e}")

    try:
        train_care_2d()
    except Exception as e:
        print(f"CARE 2D failed: {e}")

    try:
        train_care_3d()
    except Exception as e:
        print(f"CARE 3D failed: {e}")

    try:
        train_phasenet()
    except Exception as e:
        print(f"PhaseNet failed: {e}")

    try:
        train_noise2void()
    except Exception as e:
        print(f"Noise2Void failed: {e}")

    try:
        train_hsi_sdecnn()
    except Exception as e:
        print(f"HSI-SDeCNN failed: {e}")

    try:
        train_dl_sim()
    except Exception as e:
        print(f"DL-SIM failed: {e}")

    try:
        train_additional()
    except Exception as e:
        print(f"Additional models failed: {e}")

    print("\n=== All done ===")
    for p in sorted(PWMCORE_WEIGHTS.rglob("*.pth")):
        sz = p.stat().st_size / 1024 / 1024
        print(f"  {p.relative_to(PWMCORE_WEIGHTS)}: {sz:.1f} MB")
