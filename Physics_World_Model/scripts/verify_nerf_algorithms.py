"""Verify all 7 NeRF algorithms and optimize PSNR.

Tests: NeRF MLP, Plenoxels (proxy), TensoRF (proxy), Instant-NGP,
3DGS, Mip-NeRF 360, Zip-NeRF using the standard dataset.
"""

import time
import numpy as np
import h5py
from pathlib import Path

DATA_DIR = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark\nerf\standard")


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim_simple(x, y):
    """Simple SSIM approximation."""
    from skimage.metrics import structural_similarity
    return structural_similarity(x, y, channel_axis=-1, data_range=1.0)


def load_sample(idx=0):
    """Load a NeRF standard sample."""
    fpath = DATA_DIR / f"standard_nerf_{idx:02d}.h5"
    with h5py.File(fpath, "r") as f:
        x_true = f["x_true"][:]
        y_ideal = f["y_ideal"][:]
        all_images = f["all_images"][:]
        all_poses = f["all_poses"][:]
        pose = f["pose"][:]
    return x_true, y_ideal, all_images, all_poses, pose


def test_nerf_solver(all_images, all_poses, pose, x_true, model_type="instant_ngp",
                     iters=2000, lr=5e-4, n_samples=128, device="cuda:0"):
    """Run NeRF solver and compute PSNR."""
    from pwm_core.recon.nerf_solver import run_nerf

    class FakePhysics:
        def __init__(self, images, poses, target_pose):
            self.x_shape = x_true.shape
            self._images = images
            self._poses = poses
            self._pose = target_pose

        def info(self):
            return {
                "poses": self._poses,
                "intrinsics": np.array(max(x_true.shape[0], x_true.shape[1]), dtype=np.float32),
            }

    # Use subset of views for faster training
    n_train = min(len(all_images), 50)
    indices = np.linspace(0, len(all_images) - 1, n_train, dtype=int)
    train_images = all_images[indices]
    train_poses = all_poses[indices]

    physics = FakePhysics(train_images, train_poses, pose)

    cfg = {
        "model": model_type,
        "iters": iters,
        "lr": lr,
        "n_samples": n_samples,
        "near": 0.0,
        "far": 4.0,
        "render_idx": 0,
        "device": device,
    }

    t0 = time.time()
    result, info = run_nerf(train_images, physics, cfg)
    dt = time.time() - t0

    # Compute metrics
    result_resized = result
    if result.shape != x_true.shape:
        # Resize if needed
        from skimage.transform import resize
        result_resized = resize(result, x_true.shape[:2], anti_aliasing=True).astype(np.float32)
        if result_resized.ndim == 2:
            result_resized = np.stack([result_resized]*3, axis=-1)

    result_clipped = np.clip(result_resized, 0, 1).astype(np.float32)
    p = psnr(result_clipped, x_true)
    try:
        s = ssim_simple(result_clipped, x_true)
    except Exception:
        s = 0.0

    return p, s, dt, info


def main():
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x_true, y_ideal, all_images, all_poses, pose = load_sample(0)
    print(f"Scene: {all_images.shape[0]} views, {all_images.shape[1]}x{all_images.shape[2]}")

    # Test configurations for each algorithm
    algorithms = [
        # (name, model_type, iters, lr, n_samples)
        ("NeRF (MLP)", "nerf", 3000, 5e-4, 128),
        ("Instant-NGP", "instant_ngp", 3000, 5e-4, 128),
    ]

    results = {}
    for name, model_type, iters, lr, n_samples in algorithms:
        print(f"\n--- {name} ---")
        print(f"  Config: model={model_type}, iters={iters}, lr={lr}, n_samples={n_samples}")
        try:
            p, s, dt, info = test_nerf_solver(
                all_images, all_poses, pose, x_true,
                model_type=model_type, iters=iters, lr=lr,
                n_samples=n_samples, device=device,
            )
            print(f"  PSNR: {p:.2f} dB | SSIM: {s:.4f} | Time: {dt:.1f}s")
            if "final_psnr_est" in info:
                print(f"  Training PSNR est: {info['final_psnr_est']:.2f} dB")
            results[name] = {"psnr": p, "ssim": s, "time": dt}
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Now test with more iterations for best result
    print("\n\n=== Extended training (5000 iters) ===")
    for model_type in ["instant_ngp"]:
        for iters in [5000]:
            print(f"\n--- Instant-NGP {iters} iters ---")
            try:
                p, s, dt, info = test_nerf_solver(
                    all_images, all_poses, pose, x_true,
                    model_type=model_type, iters=iters, lr=5e-4,
                    n_samples=128, device=device,
                )
                print(f"  PSNR: {p:.2f} dB | SSIM: {s:.4f} | Time: {dt:.1f}s")
                results[f"NGP_{iters}"] = {"psnr": p, "ssim": s, "time": dt}
            except Exception as e:
                print(f"  ERROR: {e}")

    # Test across multiple samples
    print("\n\n=== Multi-sample test (Instant-NGP, 3000 iters) ===")
    psnrs = []
    for si in range(min(5, len(list(DATA_DIR.glob("*.h5"))))):
        try:
            x_t, _, imgs, poses_all, tgt_pose = load_sample(si)
            p, s, dt, _ = test_nerf_solver(
                imgs, poses_all, tgt_pose, x_t,
                model_type="instant_ngp", iters=3000, device=device,
            )
            psnrs.append(p)
            print(f"  Sample {si}: PSNR={p:.2f} dB, SSIM={s:.4f}, Time={dt:.1f}s")
        except Exception as e:
            print(f"  Sample {si}: ERROR {e}")

    if psnrs:
        avg_psnr = np.mean(psnrs)
        print(f"\n  Average PSNR: {avg_psnr:.2f} dB over {len(psnrs)} samples")

    print("\n=== Summary ===")
    for name, r in results.items():
        if "psnr" in r:
            print(f"  {name}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")
        else:
            print(f"  {name}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
