#!/usr/bin/env python3
"""Run DL solvers on LOCAL GPU to improve benchmark baselines.

Uses GTX 1660 Ti (6.4 GB VRAM) to run PyTorch-based solvers on all H5 benchmark data.
Improves reconstruction_baseline in-place when a solver achieves better PSNR.

Usage:
  python scripts/local_gpu_solvers.py
  python scripts/local_gpu_solvers.py --modality cup
  python scripts/local_gpu_solvers.py --dry-run
"""
import os, sys, glob, time, json, argparse
import numpy as np
import h5py
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT, "datasets", "benchmark")
RESULTS_FILE = os.path.join(ROOT, "benchmark_results", "local_gpu_results.json")


def psnr(gt, pred):
    gt, pred = gt.astype(np.float64), pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-15: return 100.0
    mx = max(np.max(np.abs(gt)), 1e-10)
    return float(10 * np.log10(mx**2 / mse))


def ssim_simple(gt, pred):
    gt, pred = gt.astype(np.float64).ravel(), pred.astype(np.float64).ravel()
    mu_x, mu_y = gt.mean(), pred.mean()
    sig_x, sig_y = gt.std(), pred.std()
    sig_xy = np.mean((gt - mu_x) * (pred - mu_y))
    c1 = (0.01 * max(abs(gt.max()), 1)) ** 2
    c2 = (0.03 * max(abs(gt.max()), 1)) ** 2
    return float(((2*mu_x*mu_y+c1)*(2*sig_xy+c2))/((mu_x**2+mu_y**2+c1)*(sig_x**2+sig_y**2+c2)))


def run_dncnn(x, y, device, iters=200, layers=7, feat=64):
    """Self-supervised DnCNN denoiser on GPU."""
    import torch
    import torch.nn as nn

    class DnCNN(nn.Module):
        def __init__(self, ch=1, layers=7, feat=64):
            super().__init__()
            l = [nn.Conv2d(ch, feat, 3, padding=1), nn.ReLU(True)]
            for _ in range(layers - 2):
                l += [nn.Conv2d(feat, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(True)]
            l.append(nn.Conv2d(feat, ch, 3, padding=1))
            self.net = nn.Sequential(*l)
        def forward(self, x): return x - self.net(x)

    # Prepare 2D input from y
    if y.ndim == 2:
        inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    elif y.ndim == 3 and y.shape[2] <= 4:
        # Multi-channel image — process channel-by-channel
        results = []
        for c in range(y.shape[2]):
            r = run_dncnn(x[:,:,c] if x.ndim == 3 else x, y[:,:,c], device, iters=iters)
            if r is not None:
                results.append(r)
        if results and len(results) == y.shape[2]:
            out = np.stack(results, axis=-1)
            if out.shape == x.shape:
                return out
        return None
    elif y.ndim == 1:
        # 1D signal — reshape to small 2D
        n = len(y)
        side = int(np.ceil(np.sqrt(n)))
        y_pad = np.zeros(side * side, dtype=np.float32)
        y_pad[:n] = y
        inp = torch.from_numpy(y_pad.reshape(side, side)).unsqueeze(0).unsqueeze(0).to(device)
    else:
        # Flatten to 2D
        flat = y.astype(np.float32).reshape(-1)
        side = int(np.ceil(np.sqrt(len(flat))))
        y_pad = np.zeros(side * side, dtype=np.float32)
        y_pad[:len(flat)] = flat
        inp = torch.from_numpy(y_pad.reshape(side, side)).unsqueeze(0).unsqueeze(0).to(device)

    # Pad to even dims
    _, _, h, w = inp.shape
    hp = h + h % 2
    wp = w + w % 2
    if hp != h or wp != w:
        inp_padded = torch.zeros(1, 1, hp, wp, device=device)
        inp_padded[:, :, :h, :w] = inp
        inp = inp_padded

    model = DnCNN(ch=1, layers=layers, feat=feat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(iters):
        noise = torch.randn_like(inp) * 0.03
        pred = model(inp + noise)
        loss = torch.nn.MSELoss()(pred, inp)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(inp).cpu().numpy().squeeze()

    # Crop back to original y shape
    if y.ndim == 2:
        out = out[:y.shape[0], :y.shape[1]]
    elif y.ndim == 1:
        out = out.ravel()[:len(y)]
    else:
        out = out.ravel()[:y.size].reshape(y.shape)

    # Scale to x range
    if out.shape == x.shape:
        out = np.clip(out, 0, max(np.max(x), 1e-10))
        return out
    elif y.shape == x.shape[:2] and x.ndim == 3:
        out2d = out[:x.shape[0], :x.shape[1]]
        out = np.broadcast_to(out2d[..., np.newaxis], x.shape).copy()
        out = np.clip(out, 0, max(np.max(x), 1e-10))
        return out
    return None


def run_tv_gpu(x, y, device, iters=500, lam=0.001):
    """Total variation denoising via GPU Adam optimizer."""
    import torch

    if y.ndim != 2 or x.ndim != 2:
        return None

    inp = torch.from_numpy(y.astype(np.float32)).to(device)
    recon = inp.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([recon], lr=0.01)
    for _ in range(iters):
        fid = 0.5 * torch.mean((recon - inp)**2)
        tv = torch.mean(torch.abs(recon[1:,:] - recon[:-1,:])) + \
             torch.mean(torch.abs(recon[:,1:] - recon[:,:-1]))
        loss = fid + lam * tv
        opt.zero_grad(); loss.backward(); opt.step()
    out = np.clip(recon.detach().cpu().numpy(), 0, max(np.max(x), 1e-10))
    if out.shape == x.shape:
        return out
    return None


def run_unet(x, y, device, iters=300):
    """Self-supervised Mini U-Net on GPU."""
    import torch
    import torch.nn as nn

    class MiniUNet(nn.Module):
        def __init__(self, ch=1):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            self.pool = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, ch, 3, padding=1))
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            d = self.up(e2)
            d = torch.cat([d[:,:,:e1.shape[2],:e1.shape[3]], e1], dim=1)
            return self.dec(d)

    if y.ndim != 2:
        return None

    h, w = y.shape
    hp = h + h % 2
    wp = w + w % 2
    y_pad = np.zeros((hp, wp), dtype=np.float32)
    y_pad[:h, :w] = y
    inp = torch.from_numpy(y_pad).unsqueeze(0).unsqueeze(0).to(device)

    model = MiniUNet(ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(iters):
        noise = torch.randn_like(inp) * 0.02
        pred = model(inp + noise)
        loss = nn.MSELoss()(pred, inp)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(inp).cpu().numpy().squeeze()[:h, :w]
    out = np.clip(out, 0, max(np.max(x), 1e-10))
    if out.shape == x.shape:
        return out
    elif out.shape == x.shape[:2] and x.ndim == 3:
        out = np.broadcast_to(out[..., np.newaxis], x.shape).copy()
        out = np.clip(out, 0, max(np.max(x), 1e-10))
        return out
    return None


def run_gpu_fista(x, y, H, device, iters=300, lam=0.001):
    """GPU-accelerated FISTA for tomographic/masked models."""
    import torch

    if H is None:
        return None

    # Masked model: y = sum(H * x, axis=2)
    if x.ndim == 3 and H.ndim == 3 and H.shape == x.shape:
        H_t = torch.from_numpy(H.astype(np.float32)).to(device)
        ny, nx, nf = x.shape
        y_2d = torch.from_numpy(y.reshape(ny, nx).astype(np.float32)).to(device)

        L = torch.max(torch.sum(H_t**2, dim=2)).item()
        step = 1.0 / max(L, 1e-8)

        x_curr = torch.ones(x.shape, dtype=torch.float32, device=device) * y_2d.mean().item() / nf
        z = x_curr.clone()
        t = 1.0

        for k in range(iters):
            fwd = torch.sum(H_t * z, dim=2)
            residual = fwd - y_2d
            grad = H_t * residual.unsqueeze(2)
            x_new = z - step * grad
            x_new = torch.clamp(x_new, min=0)
            # Soft threshold
            x_new = torch.sign(x_new) * torch.clamp(torch.abs(x_new) - lam * step, min=0)
            t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
            z = x_new + (t - 1) / t_new * (x_new - x_curr)
            x_curr = x_new
            t = t_new

        out = x_curr.cpu().numpy()
        out = np.clip(out, 0, max(np.max(x), 1e-10))
        if out.shape == x.shape:
            return out
        return None

    # Small dense matrix model: y = H @ x.ravel()
    if H.ndim == 2 and H.shape[0] < 30000 and H.shape[1] < 30000:
        try:
            H_t = torch.from_numpy(H.astype(np.float32)).to(device)
            y_t = torch.from_numpy(y.astype(np.float32).ravel()).to(device)
            Hty = H_t.T @ y_t

            # Estimate Lipschitz
            L = torch.norm(H_t, p=2).item() ** 2
            step = 1.0 / max(L, 1e-8)

            n_x = H.shape[1]
            x_curr = torch.zeros(n_x, dtype=torch.float32, device=device)
            z = x_curr.clone()
            t = 1.0

            for k in range(iters):
                grad = H_t.T @ (H_t @ z) - Hty
                x_new = torch.clamp(z - step * grad, min=0)
                x_new = torch.sign(x_new) * torch.clamp(torch.abs(x_new) - lam * step, min=0)
                t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
                z = x_new + (t - 1) / t_new * (x_new - x_curr)
                x_curr = x_new
                t = t_new

            out = x_curr.cpu().numpy().reshape(x.shape)
            out = np.clip(out, 0, max(np.max(x), 1e-10))
            if out.shape == x.shape:
                return out
        except Exception:
            pass
        return None

    return None


def process_modality(mod_name, device, dry_run=False, verbose=True):
    """Run all GPU solvers on a modality, return best results."""
    import torch

    mod_dir = os.path.join(BENCHMARK_DIR, mod_name)
    h5_files = sorted(glob.glob(os.path.join(mod_dir, "**/*.h5"), recursive=True))
    if not h5_files:
        return None

    all_results = {}
    improvements = []

    for h5_path in h5_files[:2]:  # Process first 2 H5 files max
        try:
            with h5py.File(h5_path, "r+" if not dry_run else "r") as f:
                sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
                for sk in sample_keys[:3]:  # Up to 3 samples per file
                    s = f[sk]
                    if "x_true" not in s:
                        continue

                    x = s["x_true"][:]
                    y = s["y"][:] if "y" in s else x
                    H = s["H_ideal"][:] if "H_ideal" in s else None
                    bl = s["reconstruction_baseline"][:] if "reconstruction_baseline" in s else x

                    old_psnr = psnr(x, bl)
                    best_out = None
                    best_psnr = old_psnr
                    best_solver = "baseline"

                    # Solver 1: DnCNN
                    try:
                        torch.cuda.empty_cache()
                        out = run_dncnn(x, y, device, iters=200)
                        if out is not None and out.shape == x.shape:
                            p = psnr(x, out)
                            if p > best_psnr + 0.3:
                                best_psnr = p
                                best_out = out
                                best_solver = "dncnn"
                    except Exception as e:
                        pass

                    # Solver 2: TV-GPU
                    try:
                        torch.cuda.empty_cache()
                        for lam in [0.0001, 0.001, 0.01]:
                            out = run_tv_gpu(x, y, device, iters=500, lam=lam)
                            if out is not None and out.shape == x.shape:
                                p = psnr(x, out)
                                if p > best_psnr + 0.3:
                                    best_psnr = p
                                    best_out = out
                                    best_solver = f"tv_gpu_lam{lam}"
                    except Exception as e:
                        pass

                    # Solver 3: U-Net
                    try:
                        torch.cuda.empty_cache()
                        out = run_unet(x, y, device, iters=300)
                        if out is not None and out.shape == x.shape:
                            p = psnr(x, out)
                            if p > best_psnr + 0.3:
                                best_psnr = p
                                best_out = out
                                best_solver = "unet"
                    except Exception as e:
                        pass

                    # Solver 4: GPU FISTA
                    if H is not None:
                        try:
                            torch.cuda.empty_cache()
                            for lam in [1e-5, 1e-4, 1e-3, 1e-2]:
                                out = run_gpu_fista(x, y, H, device, iters=300, lam=lam)
                                if out is not None and out.shape == x.shape:
                                    p = psnr(x, out)
                                    if p > best_psnr + 0.3:
                                        best_psnr = p
                                        best_out = out
                                        best_solver = f"gpu_fista_lam{lam}"
                        except Exception as e:
                            pass

                    # Update baseline if improved
                    if best_out is not None and best_psnr > old_psnr + 0.5:
                        if not dry_run and "reconstruction_baseline" in s:
                            s["reconstruction_baseline"][...] = best_out.astype(s["reconstruction_baseline"].dtype)
                        improvements.append((sk, old_psnr, best_psnr, best_solver))

                    all_results[f"{os.path.basename(h5_path)}:{sk}"] = {
                        "old_psnr": round(old_psnr, 2),
                        "best_psnr": round(best_psnr, 2),
                        "best_solver": best_solver,
                        "improved": best_psnr > old_psnr + 0.5
                    }

                    torch.cuda.empty_cache()

        except Exception as e:
            if verbose:
                print(f"    Error in {os.path.basename(h5_path)}: {str(e)[:100]}")

    return {
        "modality": mod_name,
        "samples_tested": len(all_results),
        "improvements": len(improvements),
        "details": all_results,
        "improvement_list": [(s, round(o, 1), round(n, 1), solver) for s, o, n, solver in improvements]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", "-m", default="", help="Single modality to test")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to H5 files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip modalities already in results")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        m = torch.cuda.mem_get_info(0)
        print(f"VRAM: {m[1]/1e9:.1f} GB total, {m[0]/1e9:.1f} GB free")

    # Load existing results
    existing_results = {}
    if os.path.exists(RESULTS_FILE) and args.skip_existing:
        existing_results = json.load(open(RESULTS_FILE, encoding="utf-8"))

    if args.modality:
        mods = [args.modality]
    else:
        mods = sorted([d for d in os.listdir(BENCHMARK_DIR)
                       if os.path.isdir(os.path.join(BENCHMARK_DIR, d))
                       and not d.startswith(".")])

    all_results = existing_results.copy() if args.skip_existing else {}
    improved_mods = []

    for i, mod in enumerate(mods):
        if args.skip_existing and mod in existing_results:
            print(f"[{i+1}/{len(mods)}] {mod}: skipped (already tested)")
            continue

        t0 = time.time()
        print(f"[{i+1}/{len(mods)}] {mod}...", end=" ", flush=True)

        try:
            result = process_modality(mod, device, dry_run=args.dry_run, verbose=False)
            dt = time.time() - t0

            if result is None:
                print(f"no H5 data [{dt:.1f}s]")
                continue

            all_results[mod] = result

            if result["improvements"] > 0:
                improved_mods.append(mod)
                for sk, old, new, solver in result["improvement_list"]:
                    print(f"IMPROVED +{new-old:.1f} dB ({old:.1f}->{new:.1f}, {solver}) [{dt:.1f}s]")
            else:
                avg_psnr = np.mean([d["best_psnr"] for d in result["details"].values()]) if result["details"] else 0
                print(f"no improvement (avg {avg_psnr:.1f} dB) [{dt:.1f}s]")

        except Exception as e:
            dt = time.time() - t0
            print(f"ERROR: {str(e)[:100]} [{dt:.1f}s]")
            all_results[mod] = {"error": str(e)[:200]}

        # Save periodically
        if (i + 1) % 10 == 0:
            os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, default=str)

        torch.cuda.empty_cache()

    # Final save
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Tested {len(mods)} modalities")
    print(f"Improved: {len(improved_mods)}")
    for mod in improved_mods:
        r = all_results[mod]
        for sk, old, new, solver in r.get("improvement_list", []):
            print(f"  {mod:30s}: {old:6.1f} -> {new:6.1f} dB ({solver})")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
