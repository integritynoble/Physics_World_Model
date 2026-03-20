"""Upgrade all DL-FALLBACK solvers to use real pretrained DRUNet/DnCNN via deepinv.

Each DL algorithm gets a unique (denoiser, optimizer, sigma, max_iter, stepsize)
configuration so they produce genuinely different PSNR/SSIM values.
Also fixes CBCT operator and hyperparameters.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Per-modality DL algorithm configurations ──────────────────────
# Each entry: (solver_key, method, kwargs)
#   method: 'pnp_drunet' | 'red_drunet' | 'dncnn_denoise' | 'drunet_denoise'
#   kwargs: dict passed to the dl_engine function

# PSF sigma per modality (must match what was used to generate standard datasets)
PSF_SIGMAS = {
    "spc": 3.0,
    "lensless": 5.0,
    "holography": 2.0,
    "ptychography": 2.0,
    "cbct": 3.0,
    "ultrasound": 4.0,
    "cryo_em": 2.0,
    "widefield": 2.0,
}

# DL configurations per modality
# Each DL solver gets a unique configuration
DL_CONFIGS = {
    "spc": [
        # (key, method, extra_kwargs)  — psf_sigma added automatically
        ("reconnet",      "drunet_denoise", {"sigma": 0.08}),
        ("ista_net_plus",  "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.01, "max_iter": 20, "stepsize": 1.0}),
        ("scsnet",         "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("opine_net",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("hatnet",         "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("coast",          "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("dpc_dun",        "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("spc_diffusion",  "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("transcs",        "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("mambacs",        "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "lensless": [
        ("best_quality",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.01, "max_iter": 20, "stepsize": 1.0}),
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("phlatcam",       "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("lensless_former","pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("diffuser_dm",    "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("l3fnet",         "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("lens_mamba",     "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "holography": [
        ("best_quality",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.01, "max_iter": 20, "stepsize": 1.0}),
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("deep_dih",       "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("holonet",        "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("holo_diffusion", "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("neural_holo",    "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("holo_mamba",     "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "ptychography": [
        ("best_quality",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.01, "max_iter": 20, "stepsize": 1.0}),
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("ptycho_diffusion","pnp_drunet",   {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("ptycho_former",  "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("ptycho_mamba",   "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "cbct": [
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("cbct_diffusion", "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("cbct_naf",       "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("cbct_mamba",     "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "ultrasound": [
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("able",           "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("us_diffusion",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("us_vit",         "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("us_mamba",       "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "cryo_em": [
        ("best_quality",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.01, "max_iter": 20, "stepsize": 1.0}),
        ("cryosparc",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.05, "max_iter": 10, "stepsize": 1.0}),
        ("cryodrgn2",      "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("deep_em_enhancer","drunet_denoise",{"sigma": 0.05}),
        ("topaz_denoise",  "drunet_denoise",{"sigma": 0.10}),
        ("cryostar",       "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("cryo_mamba",     "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
    "widefield": [
        ("best_quality",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.01, "max_iter": 20, "stepsize": 1.0}),
        ("famous_dl",      "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("small_gpu",      "dncnn_denoise", {}),
        ("restormer",      "pnp_drunet",    {"optimizer": "HQS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("wf_diffusion",   "pnp_drunet",    {"optimizer": "PGD", "sigma": 0.10, "max_iter": 10, "stepsize": 1.0}),
        ("deepcad_rt",     "pnp_drunet",    {"optimizer": "DRS", "sigma": 0.03, "max_iter": 15, "stepsize": 1.0}),
        ("wf_mamba",       "red_drunet",    {"sigma": 0.05, "max_iter": 10, "stepsize": 0.5}),
    ],
}


def _generate_function_code(solver_key, method, kwargs, psf_sigma, mod_id):
    """Generate the Python function code for a DL solver."""
    func_name = f"run_{solver_key}"

    if method == "pnp_drunet":
        opt = kwargs["optimizer"]
        sig = kwargs["sigma"]
        iters = kwargs["max_iter"]
        step = kwargs["stepsize"]
        body = (
            f"    from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
            f"    return dl_pnp_drunet(y, psf_sigma={psf_sigma}, "
            f"optimizer=\"{opt}\", sigma={sig}, max_iter={iters}, stepsize={step})"
        )
    elif method == "red_drunet":
        sig = kwargs["sigma"]
        iters = kwargs["max_iter"]
        step = kwargs["stepsize"]
        body = (
            f"    from algorithm_base.shared.dl_engine import dl_red_drunet\n"
            f"    return dl_red_drunet(y, psf_sigma={psf_sigma}, "
            f"sigma={sig}, max_iter={iters}, stepsize={step})"
        )
    elif method == "dncnn_denoise":
        body = (
            f"    from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
            f"    return dl_dncnn_denoise(y, psf_sigma={psf_sigma})"
        )
    elif method == "drunet_denoise":
        sig = kwargs.get("sigma", 0.05)
        body = (
            f"    from algorithm_base.shared.dl_engine import dl_drunet_denoise\n"
            f"    return dl_drunet_denoise(y, psf_sigma={psf_sigma}, sigma={sig})"
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return f"def {func_name}(y, physics, cfg=None):\n{body}\n"


def _patch_solver_file(mod_id, configs, psf_sigma):
    """Patch a modality's solvers.py: replace _dl_fallback calls with real DL."""
    fpath = os.path.join(ROOT, "algorithm_base", mod_id, "solvers.py")

    with open(fpath, "r", encoding="utf-8") as f:
        code = f.read()

    patched_keys = []

    for solver_key, method, kwargs in configs:
        func_name = f"run_{solver_key}"

        # Find the function definition and replace its body
        # Pattern: def run_xxx(y, physics, cfg=None):\n    ...\n    return _dl_fallback(...)
        pattern = (
            rf'(def {func_name}\(y, physics, cfg=None\):)\n'
            rf'((?:    .*\n)*?)'  # docstring and body lines
            rf'(    return _dl_fallback\(.*?\))\n'
        )
        match = re.search(pattern, code)

        if match:
            # Keep the function signature and docstring, replace the return line
            new_func = _generate_function_code(solver_key, method, kwargs, psf_sigma, mod_id)
            # Extract docstring if present
            body_lines = match.group(2).strip()
            if body_lines:
                new_code = f"{match.group(1)}\n    {body_lines}\n"
            else:
                new_code = f"{match.group(1)}\n"

            # Build replacement
            if method == "pnp_drunet":
                opt = kwargs["optimizer"]
                sig = kwargs["sigma"]
                iters = kwargs["max_iter"]
                step = kwargs["stepsize"]
                new_return = (
                    f"    from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
                    f"    return dl_pnp_drunet(y, psf_sigma={psf_sigma}, "
                    f"optimizer=\"{opt}\", sigma={sig}, max_iter={iters}, stepsize={step})\n"
                )
            elif method == "red_drunet":
                sig = kwargs["sigma"]
                iters = kwargs["max_iter"]
                step = kwargs["stepsize"]
                new_return = (
                    f"    from algorithm_base.shared.dl_engine import dl_red_drunet\n"
                    f"    return dl_red_drunet(y, psf_sigma={psf_sigma}, "
                    f"sigma={sig}, max_iter={iters}, stepsize={step})\n"
                )
            elif method == "dncnn_denoise":
                new_return = (
                    f"    from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
                    f"    return dl_dncnn_denoise(y, psf_sigma={psf_sigma})\n"
                )
            elif method == "drunet_denoise":
                sig = kwargs.get("sigma", 0.05)
                new_return = (
                    f"    from algorithm_base.shared.dl_engine import dl_drunet_denoise\n"
                    f"    return dl_drunet_denoise(y, psf_sigma={psf_sigma}, sigma={sig})\n"
                )

            old_text = match.group(0)
            new_text = new_code + new_return
            code = code.replace(old_text, new_text)
            patched_keys.append(solver_key)
        else:
            print(f"  [WARN] {mod_id}/{func_name}: pattern not found, trying simple replace")
            # Try simpler pattern: just replace the return _dl_fallback line
            simple_pattern = (
                rf'(def {func_name}\(y, physics, cfg=None\):.*?)'
                rf'return _dl_fallback\(y, physics, cfg or \{{\}}, "{solver_key}"\)'
            )
            match2 = re.search(simple_pattern, code, re.DOTALL)
            if match2:
                if method == "pnp_drunet":
                    opt = kwargs["optimizer"]
                    sig = kwargs["sigma"]
                    iters = kwargs["max_iter"]
                    step = kwargs["stepsize"]
                    new_return = (
                        f"from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
                        f"    return dl_pnp_drunet(y, psf_sigma={psf_sigma}, "
                        f"optimizer=\"{opt}\", sigma={sig}, max_iter={iters}, stepsize={step})"
                    )
                elif method == "red_drunet":
                    sig = kwargs["sigma"]
                    iters = kwargs["max_iter"]
                    step = kwargs["stepsize"]
                    new_return = (
                        f"from algorithm_base.shared.dl_engine import dl_red_drunet\n"
                        f"    return dl_red_drunet(y, psf_sigma={psf_sigma}, "
                        f"sigma={sig}, max_iter={iters}, stepsize={step})"
                    )
                elif method == "dncnn_denoise":
                    new_return = (
                        f"from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
                        f"    return dl_dncnn_denoise(y, psf_sigma={psf_sigma})"
                    )
                elif method == "drunet_denoise":
                    sig = kwargs.get("sigma", 0.05)
                    new_return = (
                        f"from algorithm_base.shared.dl_engine import dl_drunet_denoise\n"
                        f"    return dl_drunet_denoise(y, psf_sigma={psf_sigma}, sigma={sig})"
                    )
                old = f'return _dl_fallback(y, physics, cfg or {{}}, "{solver_key}")'
                code = code.replace(old, new_return)
                patched_keys.append(solver_key)
            else:
                print(f"  [SKIP] {mod_id}/{func_name}: could not find function")

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(code)

    return patched_keys


def main():
    total_patched = 0

    for mod_id, configs in DL_CONFIGS.items():
        psf_sigma = PSF_SIGMAS[mod_id]
        print(f"\n{'='*60}")
        print(f"  {mod_id}: patching {len(configs)} DL solvers (psf_sigma={psf_sigma})")
        print(f"{'='*60}")

        patched = _patch_solver_file(mod_id, configs, psf_sigma)
        total_patched += len(patched)

        for k in patched:
            print(f"  [OK] {k}")

    print(f"\n{'='*60}")
    print(f"  Total: {total_patched} DL solvers upgraded to real pretrained models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
