#!/usr/bin/env python3
"""Run DL solvers on Modal GPU for algorithm verification.

Uses Modal T4 GPU to run PyTorch-based solvers on benchmark data.
Data is passed via function arguments (serialized numpy arrays).

Usage:
  python -m modal run scripts/modal_gpu_solvers.py
  python -m modal run scripts/modal_gpu_solvers.py --modality cup
"""
import modal
import json
import os
import sys

# Create Modal app with GPU-ready image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "h5py>=3.8",
        "torch>=2.0",
    )
)

app = modal.App("pwm-gpu-solvers", image=image)


@app.function(gpu="t4", timeout=600)
def gpu_solve(x_bytes: bytes, y_bytes: bytes, x_shape: list, y_shape: list,
              x_dtype: str, y_dtype: str, mod_name: str) -> dict:
    """Run GPU solvers on a single sample."""
    import numpy as np
    import torch
    import torch.nn as nn

    x = np.frombuffer(x_bytes, dtype=x_dtype).reshape(x_shape)
    y = np.frombuffer(y_bytes, dtype=y_dtype).reshape(y_shape)

    def psnr(gt, pred):
        gt, pred = gt.astype(np.float64), pred.astype(np.float64)
        mse = np.mean((gt - pred) ** 2)
        if mse < 1e-15: return 100.0
        mx = max(np.max(np.abs(gt)), 1e-10)
        return float(10 * np.log10(mx**2 / mse))

    def ssim(gt, pred):
        gt, pred = gt.astype(np.float64).ravel(), pred.astype(np.float64).ravel()
        mu_x, mu_y = gt.mean(), pred.mean()
        sig_x, sig_y = gt.std(), pred.std()
        sig_xy = np.mean((gt - mu_x) * (pred - mu_y))
        c1, c2 = (0.01 * max(abs(gt.max()), 1))**2, (0.03 * max(abs(gt.max()), 1))**2
        return float(((2*mu_x*mu_y+c1)*(2*sig_xy+c2))/((mu_x**2+mu_y**2+c1)*(sig_x**2+sig_y**2+c2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # ---- Solver 1: Self-supervised DnCNN ----
    try:
        class DnCNN(nn.Module):
            def __init__(self, ch=1, layers=7, feat=64):
                super().__init__()
                l = [nn.Conv2d(ch, feat, 3, padding=1), nn.ReLU(True)]
                for _ in range(layers - 2):
                    l += [nn.Conv2d(feat, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(True)]
                l.append(nn.Conv2d(feat, ch, 3, padding=1))
                self.net = nn.Sequential(*l)
            def forward(self, x): return x - self.net(x)

        # Prepare 2D input
        if y.ndim == 2:
            inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            ch = 1
        else:
            inp = torch.from_numpy(y.astype(np.float32).reshape(1, 1, *y.shape[:2])).to(device)
            ch = 1

        model = DnCNN(ch=ch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(150):
            noise = torch.randn_like(inp) * 0.03
            pred = model(inp + noise)
            loss = nn.MSELoss()(pred, inp)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(inp).cpu().numpy().squeeze()
        if out.shape == x.shape[:2] or out.shape == x.shape:
            out = np.clip(out, 0, max(np.max(x), 1e-10))
            if out.ndim < x.ndim:
                out = np.broadcast_to(out[..., np.newaxis], x.shape).copy() if x.ndim == 3 else out
            if out.shape == x.shape:
                results["dncnn"] = {"psnr_db": round(psnr(x, out), 2), "ssim": round(ssim(x, out), 4), "status": "completed"}
    except Exception as e:
        results["dncnn"] = {"status": "error", "error": str(e)[:200]}

    # ---- Solver 2: TV denoising (GPU Adam) ----
    try:
        if y.ndim == 2 and x.ndim == 2:
            inp = torch.from_numpy(y.astype(np.float32)).to(device)
            recon = inp.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([recon], lr=0.01)
            for _ in range(300):
                fid = 0.5 * torch.mean((recon - inp)**2)
                tv = torch.mean(torch.abs(recon[1:,:] - recon[:-1,:])) + torch.mean(torch.abs(recon[:,1:] - recon[:,:-1]))
                loss = fid + 0.001 * tv
                opt.zero_grad(); loss.backward(); opt.step()
            out = np.clip(recon.detach().cpu().numpy(), 0, max(np.max(x), 1e-10))
            if out.shape == x.shape:
                results["tv_gpu"] = {"psnr_db": round(psnr(x, out), 2), "ssim": round(ssim(x, out), 4), "status": "completed"}
    except Exception as e:
        results["tv_gpu"] = {"status": "error", "error": str(e)[:200]}

    # ---- Solver 3: U-Net (self-supervised) ----
    try:
        class MiniUNet(nn.Module):
            def __init__(self, ch=1):
                super().__init__()
                self.enc1 = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                self.pool = nn.MaxPool2d(2)
                self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.dec = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, ch, 3, padding=1))
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                d = self.up(e2)
                d = torch.cat([d[:,:,:e1.shape[2],:e1.shape[3]], e1], dim=1)
                return self.dec(d)

        if y.ndim == 2:
            # Pad to multiple of 2
            h, w = y.shape
            hp = h + h % 2
            wp = w + w % 2
            y_pad = np.zeros((hp, wp), dtype=np.float32)
            y_pad[:h, :w] = y
            inp = torch.from_numpy(y_pad).unsqueeze(0).unsqueeze(0).to(device)
            ch = 1

            model = MiniUNet(ch=ch).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
            model.train()
            for _ in range(200):
                noise = torch.randn_like(inp) * 0.02
                pred = model(inp + noise)
                loss = nn.MSELoss()(pred, inp)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                out = model(inp).cpu().numpy().squeeze()[:h, :w]
            out = np.clip(out, 0, max(np.max(x), 1e-10))
            if out.shape == x.shape:
                results["unet"] = {"psnr_db": round(psnr(x, out), 2), "ssim": round(ssim(x, out), 4), "status": "completed"}
    except Exception as e:
        results["unet"] = {"status": "error", "error": str(e)[:200]}

    return {"modality": mod_name, "device": str(device), "solvers": results}


@app.local_entrypoint()
def main(modality: str = ""):
    """Load data locally and send to GPU."""
    import numpy as np
    import h5py
    import glob

    benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "datasets", "benchmark")

    if modality:
        mods = [modality]
    else:
        mods = ["cup", "holography", "denoising", "cassi", "cacti",
                "mri", "pet", "spc", "active_thermography"]

    all_results = {}
    for mod in mods:
        mod_dir = os.path.join(benchmark_dir, mod)
        h5_files = sorted(glob.glob(os.path.join(mod_dir, "**/*.h5"), recursive=True))
        if not h5_files:
            print(f"  {mod}: no H5 files")
            continue

        # Load first sample
        with h5py.File(h5_files[0], "r") as f:
            sample_keys = [k for k in f.keys() if k.startswith("sample_")]
            if not sample_keys:
                continue
            s = f[sample_keys[0]]
            x = s["x_true"][:]
            y = s["y"][:] if "y" in s else x

        print(f"Testing {mod} (x={x.shape}, y={y.shape})...")
        try:
            result = gpu_solve.remote(
                x.tobytes(), y.tobytes(),
                list(x.shape), list(y.shape),
                str(x.dtype), str(y.dtype),
                mod
            )
            all_results[mod] = result
            if "solvers" in result:
                for sname, sdata in result["solvers"].items():
                    if isinstance(sdata, dict) and sdata.get("status") == "completed":
                        print(f"  {sname}: {sdata['psnr_db']} dB, SSIM={sdata['ssim']}")
                    elif isinstance(sdata, dict):
                        print(f"  {sname}: {sdata.get('status')} - {sdata.get('error', '')[:80]}")
        except Exception as e:
            print(f"  Error: {e}")
            all_results[mod] = {"error": str(e)}

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "benchmark_results", "gpu_solver_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
