"""Patch remaining 19 DL-FALLBACK solvers that weren't caught by the regex."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (file_path, old_return_line, new_body)
PATCHES = [
    # === LENSLESS ===
    ("algorithm_base/lensless/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "flatnet")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=5.0, optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)'),
    ("algorithm_base/lensless/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "le_admm_u")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=5.0, optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)'),
    ("algorithm_base/lensless/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "flatnet_lite")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=5.0)'),

    # === HOLOGRAPHY ===
    ("algorithm_base/holography/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "phasenet")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)'),
    ("algorithm_base/holography/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "pr_deep")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)'),
    ("algorithm_base/holography/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "phasegan")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=2.0)'),

    # === PTYCHOGRAPHY ===
    ("algorithm_base/ptychography/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "ptychonn")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)'),
    ("algorithm_base/ptychography/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "autophase")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)'),
    ("algorithm_base/ptychography/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "ptychonn_2")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=2.0)'),

    # === CBCT ===
    ("algorithm_base/cbct/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "fdk_dl")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=3.0, optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)'),
    ("algorithm_base/cbct/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "cbct_unet")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=3.0)'),

    # === ULTRASOUND ===
    ("algorithm_base/ultrasound/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "us_unet")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=4.0, optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)'),
    ("algorithm_base/ultrasound/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "us_cnn")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=4.0)'),

    # === CRYO_EM ===
    ("algorithm_base/cryo_em/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "relion")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)'),
    ("algorithm_base/cryo_em/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "cryodrgn")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.05, max_iter=10, stepsize=1.0)'),
    ("algorithm_base/cryo_em/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "cryoai")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=2.0)'),

    # === WIDEFIELD ===
    ("algorithm_base/widefield/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "care")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.01, max_iter=20, stepsize=1.0)'),
    ("algorithm_base/widefield/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "noise2void")',
     "from algorithm_base.shared.dl_engine import dl_pnp_drunet\n"
     '    return dl_pnp_drunet(y, psf_sigma=2.0, optimizer="PGD", sigma=0.03, max_iter=15, stepsize=1.0)'),
    ("algorithm_base/widefield/solvers.py",
     'return _dl_fallback(y, physics, cfg or {}, "csbdeep")',
     "from algorithm_base.shared.dl_engine import dl_dncnn_denoise\n"
     '    return dl_dncnn_denoise(y, psf_sigma=2.0)'),
]


def main():
    patched = 0
    for relpath, old_line, new_body in PATCHES:
        fpath = os.path.join(ROOT, relpath)
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()

        if old_line in code:
            code = code.replace(old_line, new_body, 1)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(code)
            key = old_line.split('"')[1]
            print(f"  [OK] {relpath}: {key}")
            patched += 1
        else:
            key = old_line.split('"')[1] if '"' in old_line else "?"
            print(f"  [SKIP] {relpath}: {key} — already patched or not found")

    print(f"\n  Total: {patched} additional DL solvers upgraded")


if __name__ == "__main__":
    main()
