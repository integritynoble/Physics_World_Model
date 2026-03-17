"""Unified CASSI model inference wrapper.

Supports all pretrained models from the MST benchmark (caiyuanhao1998/MST):
  TSA-Net, DGSMP, Lambda-Net, ADMM-Net, GAP-Net,
  MST++, CST (S/M/L/L+), DAUHST (2/3/5/9 stg),
  BIRNAT, BiSRNet

Additional models from dedicated repos:
  RDLUF-MixS2 (ShawnDong98/RDLUF_MixS2)
  SSR (ZhangJC-2k/SSR) — S/M/L variants
  PADUT (MyuLi/PADUT) — 3/5/7/12 stage variants

Each model has a specific input preprocessing protocol:
  H:  shift_back(Y / nC * 2)  -> (bs, 28, 256, 256)
  HM: H * mask3d              -> (bs, 28, 256, 256)
  Y:  raw measurement         -> (bs, 256, 310)
  Y1: Y unsqueezed            -> (bs, 1, 256, 310)

And mask protocol:
  None:       no mask input
  Phi:        shift(mask3d)               -> (bs, 28, 256, 310)
  Mask:       mask3d                      -> (bs, 28, 256, 256)
  Phi_PhiPhiT: (Phi_batch, Phi_s_batch)  -> tuple
  Mask_PhiS:  (mask3d, Phi_s.unsqueeze)  -> tuple (for SSR)
"""
from __future__ import annotations
import sys
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

_PKG_ROOT = Path(__file__).resolve().parent          # .../recon/
_PWM_CORE = _PKG_ROOT.parent                         # .../pwm_core/
_CKPT_ROOT = _PWM_CORE / "weights" / "cassi_checkpoints"

# ─── Model registry ───────────────────────────────────────────────────────────
# Each entry: (method_key, model_constructor_args, input_setting, input_mask,
#              checkpoint_subpath, reference_psnr)
MODEL_REGISTRY = {
    # All MST-benchmark models were trained with BINARY mask (mask > 0.5)
    # binary_mask=True means: binarize mask and regenerate y from x_true at test time
    "tsa_net": {
        "input_setting": "HM", "input_mask": None, "binary_mask": True,
        "ckpt": "tsa_net/tsa_net.pth", "ref_psnr": 31.5,
    },
    "dgsmp": {
        "input_setting": "Y", "input_mask": None, "binary_mask": True,
        "ckpt": "dgsmp/dgsmp.pth", "ref_psnr": 32.6,
    },
    "lambda_net": {
        "input_setting": "Y", "input_mask": "Phi", "binary_mask": True,
        "ckpt": "lambda_net/lambda_net.pth", "ref_psnr": 30.1,
    },
    "admm_net": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "admm_net/admm_net.pth", "ref_psnr": 29.1,
    },
    "gap_net": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "gap_net/gap_net.pth", "ref_psnr": 29.1,
    },
    "mst_plus_plus": {
        "input_setting": "H", "input_mask": "Mask", "binary_mask": True,
        "ckpt": "mst_plus_plus/mst_plus_plus.pth", "ref_psnr": 36.0,
    },
    "cst_s": {
        "input_setting": "H", "input_mask": "Mask", "binary_mask": True,
        "ckpt": "cst/cst_s.pth", "ref_psnr": 33.0,
    },
    "cst_m": {
        "input_setting": "H", "input_mask": "Mask", "binary_mask": True,
        "ckpt": "cst/cst_m.pth", "ref_psnr": 34.0,
    },
    "cst_l": {
        "input_setting": "H", "input_mask": "Mask", "binary_mask": True,
        "ckpt": "cst/cst_l.pth", "ref_psnr": 35.3,
    },
    "cst_l_plus": {
        "input_setting": "H", "input_mask": "Mask", "binary_mask": True,
        "ckpt": "cst/cst_l_plus.pth", "ref_psnr": 36.1,
    },
    "dauhst_2stg": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "dauhst/dauhst_2stg.pth", "ref_psnr": 35.0,
    },
    "dauhst_3stg": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "dauhst/dauhst_3stg.pth", "ref_psnr": 36.0,
    },
    "dauhst_5stg": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "dauhst/dauhst_5stg.pth", "ref_psnr": 37.3,
    },
    "dauhst_9stg": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "dauhst/dauhst_9stg.pth", "ref_psnr": 38.4,
    },
    "birnat": {
        "input_setting": "Y", "input_mask": "Phi", "binary_mask": True,
        "ckpt": "birnat/birnat.pth", "ref_psnr": 30.0,
    },
    "bisrnet": {
        "input_setting": "H", "input_mask": "Mask", "binary_mask": True,
        "ckpt": "bisrnet/bisrnet.pth", "ref_psnr": 33.0,
    },
    "hdnet": {
        "input_setting": "H", "input_mask": None, "binary_mask": True,
        "ckpt": "hdnet/hdnet.pth", "ref_psnr": 34.97,
    },
    "mst_s": {
        "input_setting": "H", "input_mask": "Phi", "binary_mask": True,
        "ckpt": "mst/mst_s.pth", "ref_psnr": 32.0,
    },
    "mst_m": {
        "input_setting": "H", "input_mask": "Phi", "binary_mask": True,
        "ckpt": "mst/mst_m.pth", "ref_psnr": 33.5,
    },
    "mst_l": {
        "input_setting": "H", "input_mask": "Phi", "binary_mask": True,
        "ckpt": "mst/mst_l.pth", "ref_psnr": 34.81,
    },
    # ── RDLUF-MixS2 (ShawnDong98/RDLUF_MixS2) ──
    "rdluf_mixs2_9stg": {
        "input_setting": "Y_norm", "input_mask": "Phi_roll", "binary_mask": True,
        "ckpt": "rdluf_mixs2/RDLUF_MixS2_9stage.pth", "ref_psnr": 39.6,
        "output_format": "tuple",  # returns (out, log_dict)
        "ckpt_format": "dict_model",  # checkpoint['model']
        "circular_shift": True,
    },
    # ── SSR (ZhangJC-2k/SSR) ──
    "ssr_s": {
        "input_setting": "Y1", "input_mask": "Mask_PhiS", "binary_mask": True,
        "ckpt": "ssr/model_S.pkl", "ref_psnr": 32.0,
        "output_format": "list",
        "ckpt_format": "dict_model",
        "stage": 3,
    },
    "ssr_m": {
        "input_setting": "Y1", "input_mask": "Mask_PhiS", "binary_mask": True,
        "ckpt": "ssr/model_M.pkl", "ref_psnr": 33.0,
        "output_format": "list",
        "ckpt_format": "dict_model",
        "stage": 6,
    },
    "ssr_l": {
        "input_setting": "Y1", "input_mask": "Mask_PhiS", "binary_mask": True,
        "ckpt": "ssr/model_L.pkl", "ref_psnr": 34.0,
        "output_format": "list",
        "ckpt_format": "dict_model",
        "stage": 9,
    },
    # ── PADUT (MyuLi/PADUT) ──
    # PADUT uses linear shift for measurement/Phi (same as MST-benchmark),
    # circular shift (torch.roll) only inside the model's forward pass
    "padut_3stg": {
        "input_setting": "Y", "input_mask": "Phi_PhiPhiT", "binary_mask": True,
        "ckpt": "padut/padut_3stg_official.pth", "ref_psnr": 36.95,
        "ckpt_format": "dict_model",
        "nums_stages": 2,
    },
}

# ─── Torch utilities ──────────────────────────────────────────────────────────

def _shift_torch(inputs, step=2):
    """Shift each spectral band rightward by step*i pixels. (bs, nC, H, W) -> (bs, nC, H, W+step*(nC-1))"""
    import torch
    bs, nC, row, col = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step, device=inputs.device, dtype=inputs.dtype)
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def _shift_back_torch(inputs, step=2):
    """Extract each band from dispersed measurement. (bs, H, W_meas) -> (bs, nC, H, W)"""
    import torch
    bs, row, col = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step, device=inputs.device, dtype=inputs.dtype)
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def _shift_3d(inputs, step=2):
    """Shift bands in 3D cube (used by ADMM/GAP for shift-and-sum)."""
    import torch
    bs, nC, row, col = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def _shift_back_3d(inputs, step=2):
    """Reverse shift_3d."""
    import torch
    bs, nC, row, col = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=(-1) * step * i, dims=2)
    return inputs


def _prepare_masks(mask_2d: np.ndarray, nC: int = 28, step: int = 2, device=None):
    """From a 2D mask (H, W), produce all mask variants needed by different models.

    Returns dict with keys: mask3d, Phi (shifted mask3d), Phi_s (sum of Phi**2).
    All as torch tensors with batch dim.
    """
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = mask_2d.shape
    # mask3d: (1, nC, H, W)
    mask3d = torch.from_numpy(
        np.tile(mask_2d[np.newaxis, :, :], (nC, 1, 1))
    ).unsqueeze(0).float().to(device)

    # Phi: shifted mask3d, (1, nC, H, W + (nC-1)*step)
    Phi = _shift_torch(mask3d, step)

    # Phi_s: sum(Phi**2, dim=1), floor at 1
    Phi_s = torch.sum(Phi ** 2, dim=1)  # (1, H, W_meas)
    Phi_s[Phi_s == 0] = 1

    # Phi_roll: circular-shifted mask3d using torch.roll (for RDLUF-MixS2, PADUT)
    mask3d_padded = torch.zeros(1, nC, H, W + (nC - 1) * step, device=device)
    mask3d_padded[:, :, :, :W] = mask3d
    Phi_roll = mask3d_padded.clone()
    for i in range(nC):
        Phi_roll[:, i, :, :] = torch.roll(Phi_roll[:, i, :, :], shifts=step * i, dims=2)

    # Phi_s_roll: sum(Phi_roll**2, dim=1), floor at 1
    Phi_s_roll = torch.sum(Phi_roll ** 2, dim=1)
    Phi_s_roll[Phi_s_roll == 0] = 1

    return {"mask3d": mask3d, "Phi": Phi, "Phi_s": Phi_s,
            "Phi_roll": Phi_roll, "Phi_s_roll": Phi_s_roll}


def _prepare_input(y: np.ndarray, mask_2d: np.ndarray, model_key: str,
                   nC: int = 28, step: int = 2, device=None):
    """Prepare model input and mask from raw measurement and 2D mask.

    Args:
        y: measurement, shape (H, W_meas) = (256, 310) for 28-band step=2
        mask_2d: coded aperture mask, shape (H, W) = (256, 256)
        model_key: key in MODEL_REGISTRY
        nC: number of spectral bands
        step: dispersion step

    Returns:
        (input_meas, input_mask) ready for model.forward()
    """
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = MODEL_REGISTRY[model_key]
    input_setting = spec["input_setting"]
    mask_type = spec["input_mask"]

    # y as tensor: (1, H, W_meas)
    y_t = torch.from_numpy(y).unsqueeze(0).float().to(device)

    masks = _prepare_masks(mask_2d, nC, step, device)

    # Prepare measurement input
    if input_setting == "Y":
        input_meas = y_t  # (1, 256, 310)
    elif input_setting == "Y1":
        input_meas = y_t.unsqueeze(1)  # (1, 1, 256, 310)
    elif input_setting == "Y_norm":
        # RDLUF-MixS2: measurement generated with circular shift + normalized
        input_meas = y_t / nC * 2  # (1, 256, 310)
    elif input_setting == "H":
        H = _shift_back_torch(y_t / nC * 2, step)  # (1, 28, 256, 256)
        input_meas = H
    elif input_setting == "HM":
        H = _shift_back_torch(y_t / nC * 2, step)
        HM = H * masks["mask3d"]
        input_meas = HM
    else:
        raise ValueError(f"Unknown input_setting: {input_setting}")

    # Prepare mask input
    if mask_type is None:
        input_mask = None
    elif mask_type == "Phi":
        input_mask = masks["Phi"]  # (1, 28, 256, 310) shifted
    elif mask_type == "Phi_roll":
        input_mask = masks["Phi_roll"]  # (1, 28, 256, 310) circular-shifted
    elif mask_type == "Mask":
        input_mask = masks["mask3d"]  # (1, 28, 256, 256)
    elif mask_type == "Phi_PhiPhiT":
        input_mask = (masks["Phi"], masks["Phi_s"])
    elif mask_type == "Mask_PhiS":
        # SSR format: unshifted mask3d + Phi_s with channel dim
        input_mask = (masks["mask3d"], masks["Phi_s"].unsqueeze(1))
    elif mask_type == "Phi_roll_PhiS":
        # PADUT: circular-shifted Phi + Phi_s from circular-shifted mask
        input_mask = (masks["Phi_roll"], masks["Phi_s_roll"])
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return input_meas, input_mask


def _build_model(model_key: str, device=None):
    """Construct the model architecture for a given key."""
    import torch

    # Import from cassi_arch package — add parent dir so 'cassi_arch' is importable
    arch_parent = str(_PKG_ROOT)
    if arch_parent not in sys.path:
        sys.path.insert(0, arch_parent)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_key == "tsa_net":
        from cassi_arch.TSA_Net import TSA_Net
        model = TSA_Net(in_ch=28, out_ch=28)
    elif model_key == "dgsmp":
        from cassi_arch.DGSMP import HSI_CS
        model = HSI_CS(Ch=28, stages=4)
    elif model_key == "lambda_net":
        from cassi_arch.Lambda_Net import Lambda_Net
        model = Lambda_Net(out_ch=28)
    elif model_key == "admm_net":
        from cassi_arch.ADMM_Net import ADMM_net
        model = ADMM_net()
    elif model_key == "gap_net":
        from cassi_arch.GAP_Net import GAP_net
        model = GAP_net()
    elif model_key == "mst_plus_plus":
        from cassi_arch.MST_Plus_Plus import MST_Plus_Plus
        model = MST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3)
    elif model_key.startswith("cst_"):
        from cassi_arch.CST import CST
        blocks_map = {
            "cst_s": ([1, 1, 2], True),
            "cst_m": ([2, 2, 2], True),
            "cst_l": ([2, 4, 6], True),
            "cst_l_plus": ([2, 4, 6], False),
        }
        num_blocks, sparse = blocks_map[model_key]
        model = CST(num_blocks=num_blocks, sparse=sparse)
    elif model_key.startswith("dauhst_"):
        from cassi_arch.DAUHST import DAUHST
        num_iter = int(model_key.split("_")[1].replace("stg", ""))
        model = DAUHST(num_iterations=num_iter)
    elif model_key == "birnat":
        from cassi_arch.BIRNAT import BIRNAT
        model = BIRNAT()
    elif model_key == "bisrnet":
        from cassi_arch.BiSRNet import BiSRNet
        model = BiSRNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1, 1, 1])
    elif model_key == "hdnet":
        from cassi_arch.HDNet import HDNetOriginal
        model = HDNetOriginal(in_ch=28, out_ch=28)
    elif model_key == "mst_s":
        from cassi_arch.MST_Plus_Plus import MST
        model = MST(dim=28, stage=2, num_blocks=[2, 2, 2])
    elif model_key == "mst_m":
        from cassi_arch.MST_Plus_Plus import MST
        model = MST(dim=28, stage=2, num_blocks=[2, 4, 2])
    elif model_key == "mst_l":
        # mst_l checkpoint uses the original MST architecture (mst.py MST class
        # with 3-element encoder stages including mask downsampling conv)
        import importlib.util as _ilu
        _mst_paths = [
            str(_PKG_ROOT / "mst.py"),  # local: .../recon/mst.py
            "/mst.py",                   # Modal: added as /mst.py in image
        ]
        _mst_mod = None
        for _mp in _mst_paths:
            _spec = _ilu.spec_from_file_location("_mst_orig", _mp)
            if _spec is not None:
                try:
                    _mst_mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mst_mod)
                    break
                except Exception:
                    pass
        if _mst_mod is None:
            raise ImportError("Cannot load mst.py for mst_l model")
        model = _mst_mod.MST(dim=28, stage=2, num_blocks=[4, 7, 5])
    elif model_key == "rdluf_mixs2_9stg":
        from cassi_arch.RDLUF_MixS2 import DUF_MixS2
        from argparse import Namespace
        opt = Namespace(
            in_dim=28, out_dim=28, dim=28, stage=9,
            DW_Expand=1, FFN_Expand=2.66,
            bias=False, LayerNorm_type="BiasFree",
            ffn_name="Gated_Dconv_FeedForward",
            act_fn_name="gelu",
            body_share_params=1,
            spatial_branch=1, spectral_branch=1,
            spatial_interaction=1, spectral_interaction=1,
            stage_interaction=1, block_interaction=1,
        )
        model = DUF_MixS2(opt)
    elif model_key.startswith("ssr_"):
        from cassi_arch.SSR import Net
        from argparse import Namespace
        variant_map = {"ssr_s": 3, "ssr_m": 6, "ssr_l": 9}
        n_stage = variant_map.get(model_key, 9)
        opt = Namespace(stage=n_stage, bands=28, size=256)
        model = Net(opt)
    elif model_key.startswith("padut_"):
        from cassi_arch.PADUT import PADUT
        spec = MODEL_REGISTRY[model_key]
        nums_stages = spec.get("nums_stages")
        if nums_stages is None:
            n_stg = int(model_key.split("_")[1].replace("stg", ""))
            nums_stages = n_stg - 1
        model = PADUT(in_c=28, n_feat=28, nums_stages=nums_stages)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    model = model.to(device)
    return model


def _load_checkpoint(model, model_key: str, ckpt_path: str = None):
    """Load pretrained weights into model."""
    import torch
    if ckpt_path is None:
        ckpt_path = str(_CKPT_ROOT / MODEL_REGISTRY[model_key]["ckpt"])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    spec = MODEL_REGISTRY[model_key]
    ckpt_fmt = spec.get("ckpt_format", "raw")

    if ckpt_fmt == "dict_model":
        # Checkpoint is a dict with 'model' key (RDLUF-MixS2, SSR)
        raw_sd = checkpoint["model"]
        # For RDLUF-MixS2: use EMA shadow_params for better quality
        if "ema" in checkpoint and isinstance(checkpoint.get("ema"), dict):
            ema_data = checkpoint["ema"]
            if "shadow_params" in ema_data:
                shadow = ema_data["shadow_params"]
                # shadow_params is ordered the same as model.parameters()
                # in the original trained model (same order as raw_sd keys)
                param_names = [k for k in raw_sd.keys()]
                if len(shadow) == len(param_names):
                    raw_sd = {param_names[i]: shadow[i] for i in range(len(shadow))}
    elif ckpt_fmt == "direct":
        # Checkpoint is the state_dict directly
        raw_sd = checkpoint
    else:
        # Raw: checkpoint itself is the state dict (MST-benchmark models)
        raw_sd = checkpoint

    # Handle module. prefix from DataParallel training
    state_dict = {k.replace("module.", ""): v for k, v in raw_sd.items()}

    # Try loading; if strict fails due to key naming, try key remapping
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # RDLUF-MixS2: checkpoint uses ffn.X but model has ffn.module.X
        # due to Residual wrapper. Remap checkpoint keys.
        model_keys = set(model.state_dict().keys())
        remapped = {}
        for k, v in state_dict.items():
            if k in model_keys:
                remapped[k] = v
            else:
                # Try adding .module after ffn
                k2 = k.replace(".ffn.", ".ffn.module.")
                if k2 in model_keys:
                    remapped[k2] = v
                else:
                    remapped[k] = v
        model.load_state_dict(remapped, strict=True)

    return model


# ─── Main inference function ──────────────────────────────────────────────────

_MODEL_CACHE = {}


def _regenerate_measurement(x_true: np.ndarray, mask_2d: np.ndarray,
                             nC: int = 28, step: int = 2,
                             circular: bool = False) -> np.ndarray:
    """Regenerate CASSI measurement from ground truth and mask.

    Args:
        x_true: HSI cube, shape (H, W, nC)
        mask_2d: coded aperture mask, shape (H, W)
        circular: if True, use circular shift (torch.roll convention) for RDLUF
    Returns:
        y: measurement, shape (H, W + (nC-1)*step)
    """
    H, W = mask_2d.shape
    W_meas = W + (nC - 1) * step
    y = np.zeros((H, W_meas), dtype=np.float32)
    if circular:
        # Circular shift convention (RDLUF-MixS2)
        for k in range(nC):
            masked = mask_2d * x_true[:, :, k]  # (H, W)
            padded = np.zeros((H, W_meas), dtype=np.float32)
            padded[:, :W] = masked
            y += np.roll(padded, shift=step * k, axis=1)
    else:
        for k in range(nC):
            y[:, k * step: k * step + W] += mask_2d * x_true[:, :, k]
    return y


def cassi_model_recon(y: np.ndarray, mask_2d: np.ndarray, model_key: str,
                      nC: int = 28, step: int = 2,
                      ckpt_path: str = None, device: str = None,
                      x_true: np.ndarray = None) -> np.ndarray:
    """Run CASSI reconstruction using a pretrained model.

    Args:
        y: measurement, shape (H, W_meas) e.g. (256, 310)
        mask_2d: coded aperture mask, shape (H, W) e.g. (256, 256)
        model_key: model identifier (e.g. 'dauhst_9stg', 'cst_l_plus', etc.)
        nC: number of spectral bands (default 28)
        step: dispersion step (default 2)
        ckpt_path: optional override for checkpoint path
        device: optional device string
        x_true: ground truth HSI cube (H, W, nC) — needed for models trained
                with binary mask to regenerate measurement. If None and model
                needs binary mask, the continuous mask is used as-is (lower PSNR).

    Returns:
        x_hat: reconstructed HSI cube, shape (H, W, nC) e.g. (256, 256, 28)
    """
    import torch

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

    spec = MODEL_REGISTRY[model_key]
    uses_binary = spec.get("binary_mask", False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Handle binary mask: binarize mask and regenerate measurement from x_true
    if uses_binary and x_true is not None:
        mask_2d = (mask_2d > 0.5).astype(np.float32)
        circular = spec.get("circular_shift", False) or spec.get("input_setting") == "Y_norm"
        y = _regenerate_measurement(x_true, mask_2d, nC, step, circular=circular)

    # Cache model to avoid reloading
    cache_key = (model_key, ckpt_path or "default", str(dev))
    if cache_key not in _MODEL_CACHE:
        model = _build_model(model_key, dev)
        model = _load_checkpoint(model, model_key, ckpt_path)
        model.eval()
        _MODEL_CACHE[cache_key] = model
    model = _MODEL_CACHE[cache_key]

    # Prepare input
    input_meas, input_mask = _prepare_input(y, mask_2d, model_key, nC, step, dev)

    # Run inference
    output_fmt = spec.get("output_format", "tensor")
    with torch.no_grad():
        if input_mask is not None:
            out = model(input_meas, input_mask)
        else:
            out = model(input_meas)

    # Handle different output formats
    if output_fmt == "tuple":
        # RDLUF-MixS2: returns (out_tensor, log_dict)
        out = out[0]
    elif output_fmt == "list":
        # SSR: returns list of stage outputs, take last
        n_stage = spec.get("stage", len(out))
        out = out[n_stage - 1] if isinstance(out, list) else out

    # out shape: (1, 28, 256, 256)
    x_hat = out.detach().cpu().numpy()[0]  # (28, 256, 256)
    x_hat = np.transpose(x_hat, (1, 2, 0))  # (256, 256, 28)
    x_hat = np.clip(x_hat, 0, 1).astype(np.float32)

    return x_hat


# ─── Solver interface functions (for algorithm_base) ──────────────────────────

def run_cassi_model(y: np.ndarray, operator: Any, cfg: Dict) -> np.ndarray:
    """Standard solver interface for CASSI model reconstruction.

    cfg must contain 'model_key'. Optionally 'ckpt_path', 'device', 'x_true'.
    operator must have .mask (2D), .n_bands, .step attributes.
    """
    model_key = cfg.get("model_key")
    if model_key is None:
        raise ValueError("cfg must contain 'model_key'")

    mask_2d = operator.mask if operator else cfg.get("mask")
    nC = operator.n_bands if operator else cfg.get("n_bands", 28)
    step = operator.step if operator else cfg.get("step", 2)

    return cassi_model_recon(
        y=y.astype(np.float32),
        mask_2d=mask_2d.astype(np.float32),
        model_key=model_key,
        nC=nC,
        step=step,
        ckpt_path=cfg.get("ckpt_path"),
        device=cfg.get("device"),
        x_true=cfg.get("x_true"),
    )
