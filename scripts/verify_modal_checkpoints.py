"""Verify all GPU algorithm checkpoints load correctly on Modal."""

import modal

app = modal.App("pwm-verify-checkpoints")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy", "scipy")
)

# Map of checkpoint name -> list of expected .pth files (relative to /checkpoint/)
CHECKPOINTS = {
    "ELP-Unfolding": ["ELP-Unfolding/ckptall.pth", "ELP-Unfolding/ckptallS.pth"],
    "DRUNet": [
        "DRUNet/drunet_deepinv_color_finetune_22k.pth",
        "DRUNet/drunet_deepinv_gray_finetune_26k.pth",
    ],
    "DnCNN": ["DnCNN/dncnn_25.pth"],
    "EfficientSCI": ["EfficientSCI/efficientsci_base.pth"],
    "HATNet-SPI": [
        "HATNet-SPI/2024_pretraiend_weights/cr_0.04.pth",
        "HATNet-SPI/2024_pretraiend_weights/cr_0.1.pth",
        "HATNet-SPI/2024_pretraiend_weights/cr_0.25.pth",
        "HATNet-SPI/2024_pretraiend_weights/cr_0.5.pth",
    ],
    "ISTA-Net": [
        "ISTA-Net/CS_ISTA_Net_plus_layer_9_group_1_ratio_25_lr_0.0001/model_best.pkl",
    ],
    "MST-HDNet": [
        "MST-HDNet/mst/mst_s.pth",
        "MST-HDNet/mst/mst_l.pth",
        "MST-HDNet/hdnet/hdnet.pth",
    ],
    "PnP-CASSI": ["PnP-CASSI/deep_denoiser.pth"],
    "PnP-SCI": [
        "PnP-SCI/ffdnet/net_rgb.pth",
        "PnP-SCI/ffdnet/net_gray.pth",
        "PnP-SCI/fastdvdnet/model.pth",
    ],
    "ProxUnroll": [
        "ProxUnroll/admm_proxunroll.pth",
        "ProxUnroll/hqs_proxunroll.pth",
    ],
}


@app.function(
    image=image,
    gpu="any",
    volumes={"/models": vol},
    timeout=300,
)
def verify_checkpoints():
    """Load every checkpoint on GPU to verify they work."""
    import torch
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    for name, files in CHECKPOINTS.items():
        print(f"\n=== {name} ===")
        for rel_path in files:
            full_path = f"/models/checkpoint/{rel_path}"
            if not os.path.exists(full_path):
                print(f"  MISSING: {rel_path}")
                results[rel_path] = "MISSING"
                continue

            size_mb = os.path.getsize(full_path) / 1e6
            try:
                ckpt = torch.load(full_path, map_location=device, weights_only=False)
                if isinstance(ckpt, dict):
                    keys = list(ckpt.keys())[:5]
                    n_keys = len(ckpt)
                    print(f"  OK: {rel_path} ({size_mb:.1f} MB, dict with {n_keys} keys: {keys}...)")
                else:
                    print(f"  OK: {rel_path} ({size_mb:.1f} MB, type={type(ckpt).__name__})")
                results[rel_path] = "OK"
            except Exception as e:
                print(f"  FAIL: {rel_path} ({size_mb:.1f} MB) - {e}")
                results[rel_path] = f"FAIL: {e}"

    # Summary
    ok = sum(1 for v in results.values() if v == "OK")
    fail = sum(1 for v in results.values() if v != "OK")
    print(f"\n{'='*50}")
    print(f"Results: {ok} OK, {fail} failed out of {len(results)} files")

    if fail > 0:
        print("\nFailed files:")
        for path, status in results.items():
            if status != "OK":
                print(f"  {path}: {status}")

    return results


@app.local_entrypoint()
def main():
    results = verify_checkpoints.remote()
    ok = sum(1 for v in results.values() if v == "OK")
    fail = sum(1 for v in results.values() if v != "OK")
    print(f"\nFinal: {ok} OK, {fail} failed")
