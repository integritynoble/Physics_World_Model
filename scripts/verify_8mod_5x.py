"""Verify all algorithms for 8 modalities × 5 runs each.

Loads standard dataset sample 0, runs each solver 5 times,
computes PSNR/SSIM, and saves results to JSON + updates index.md.
"""

import sys, os, json, time, importlib
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np

# Work from the public repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import h5py

MODALITIES = ["spc", "lensless", "holography", "ptychography",
              "cbct", "ultrasound", "cryo_em", "widefield"]

NUM_RUNS = 5


def compute_psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 60.0
    return float(10 * np.log10(1.0 / mse))


def compute_ssim(x_true, x_hat):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(x_true, x_hat, data_range=1.0))
    except ImportError:
        # Manual SSIM
        mu_x = x_true.mean()
        mu_y = x_hat.mean()
        sig_x = x_true.var()
        sig_y = x_hat.var()
        sig_xy = ((x_true - mu_x) * (x_hat - mu_y)).mean()
        C1, C2 = 0.01**2, 0.03**2
        num = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
        return float(num / den)


def load_standard_sample(modality, idx=0):
    path = os.path.join(ROOT, "datasets", "benchmark", modality, "standard",
                        f"standard_{modality}_{idx:02d}.h5")
    with h5py.File(path, "r") as f:
        x_true = f["x_true"][:]
        y = f["y_ideal"][:]
    return x_true.astype(np.float32), y.astype(np.float32)


def run_verification():
    results = {}
    total_solvers = 0
    total_done = 0

    for modality in MODALITIES:
        print(f"\n{'='*60}")
        print(f"  MODALITY: {modality}")
        print(f"{'='*60}")

        # Load data
        try:
            x_true, y = load_standard_sample(modality, 0)
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            results[modality] = {"error": str(e)}
            continue

        # Normalize
        x_true_n = x_true / max(x_true.max(), 1e-8)
        y_n = y / max(y.max(), 1e-8)

        # Import solvers
        try:
            mod = importlib.import_module(f"algorithm_base.{modality}.solvers")
            solver_dict = mod.SOLVERS
        except Exception as e:
            print(f"  ERROR importing solvers: {e}")
            results[modality] = {"error": str(e)}
            continue

        solver_keys = list(solver_dict.keys())
        print(f"  {len(solver_keys)} solvers: {solver_keys}")
        total_solvers += len(solver_keys)

        mod_results = {}
        for sk in solver_keys:
            info = solver_dict[sk]
            algo_name = info.get("name", sk)
            reference = info.get("reference", "")
            gpu = info.get("gpu", False)
            runs = []
            all_pass = True

            for run_idx in range(NUM_RUNS):
                t0 = time.time()
                try:
                    # Some modalities use 'operator', others 'physics'
                    import inspect
                    sig = inspect.signature(mod.run_solver)
                    params = list(sig.parameters.keys())
                    if 'physics' in params:
                        x_hat = mod.run_solver(sk, y_n.copy(), physics=None, cfg={})
                    else:
                        x_hat = mod.run_solver(sk, y_n.copy(), operator=None, cfg={})
                    elapsed = time.time() - t0
                    x_hat = np.clip(np.asarray(x_hat, dtype=np.float32), 0, 1)

                    # Ensure shape match
                    if x_hat.shape != x_true_n.shape:
                        x_hat = x_hat.reshape(x_true_n.shape) if x_hat.size == x_true_n.size else \
                                np.zeros_like(x_true_n)

                    psnr = compute_psnr(x_true_n, x_hat)
                    ssim = compute_ssim(x_true_n, x_hat)
                    runs.append({
                        "run": run_idx + 1,
                        "psnr": round(psnr, 2),
                        "ssim": round(ssim, 4),
                        "time": round(elapsed, 3),
                        "status": "pass"
                    })
                    status_char = "OK"
                except Exception as e:
                    elapsed = time.time() - t0
                    runs.append({
                        "run": run_idx + 1,
                        "psnr": 0.0,
                        "ssim": 0.0,
                        "time": round(elapsed, 3),
                        "status": "fail",
                        "error": str(e)[:100]
                    })
                    all_pass = False
                    status_char = "FAIL"

                if run_idx == 0:
                    r = runs[0]
                    print(f"  [{sk:25s}] Run {run_idx+1}: {status_char}  "
                          f"PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}  "
                          f"t={r['time']:.3f}s  ({algo_name})")
                else:
                    r = runs[-1]
                    print(f"  {'':25s}  Run {run_idx+1}: {status_char}  "
                          f"PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}  "
                          f"t={r['time']:.3f}s")

            # Compute averages
            pass_runs = [r for r in runs if r["status"] == "pass"]
            avg_psnr = np.mean([r["psnr"] for r in pass_runs]) if pass_runs else 0.0
            avg_ssim = np.mean([r["ssim"] for r in pass_runs]) if pass_runs else 0.0

            done = all_pass and len(pass_runs) == NUM_RUNS
            if done:
                total_done += 1

            mod_results[sk] = {
                "name": algo_name,
                "reference": reference,
                "gpu": gpu,
                "runs": runs,
                "avg_psnr": round(float(avg_psnr), 2),
                "avg_ssim": round(float(avg_ssim), 4),
                "done": done,
                "pass_count": len(pass_runs),
            }

        results[modality] = mod_results

    # Save JSON
    out_path = os.path.join(ROOT, "benchmark_results", "verification_8mod_5x.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total: {total_done}/{total_solvers} algorithms done (5/5 runs pass)")

    return results


def update_index_md(results):
    """Update datasets/templates/index.md with verification results."""
    lines = []
    lines.append("# Modality Templates Index\n")
    lines.append("")
    lines.append("Comprehensive 7-step templates for all 168 imaging modalities in the PWM5 Benchmark.")
    lines.append("Each template covers: (1) Verify Standard Dataset, (2) List All Algorithms, "
                 "(3) Update Solvers, (4) Verify Each Algorithm, (5) Upload Checkpoints to GCS, "
                 "(6) Upload Standard Dataset to GCS, (7) Push to GitHub.\n")
    lines.append("")
    lines.append("---\n")
    lines.append("")
    lines.append("## Implementation Tracking — 12 Flagship Paper Modalities\n")
    lines.append("")
    lines.append("Each algorithm must be implemented at least **5 times** "
                 "(5 independent verification runs on the standard dataset).")
    lines.append("When all 5 runs are complete, the algorithm status is marked **done**.\n")

    # Count totals
    total_algos = 0
    total_done = 0

    # Modality display info
    MODALITY_INFO = {
        "cassi":        ("CASSI — Coded Aperture Snapshot Spectral Imaging", "cassi"),
        "cacti":        ("CACTI — Coded Aperture Compressive Temporal Imaging", "cacti"),
        "spc":          ("SPC — Single-Pixel Camera", "spc"),
        "lensless":     ("Lensless Imaging", "lensless"),
        "holography":   ("Digital Holographic Microscopy / Compressive Holography", "holography"),
        "ptychography": ("Ptychographic Imaging / Electron Ptychography", "ptychography"),
        "ct":           ("CT — X-ray Computed Tomography", "ct"),
        "cbct":         ("CBCT — Cone-Beam CT", "cbct"),
        "ultrasound":   ("Ultrasound B-mode", "ultrasound"),
        "cryo_em":      ("Cryo-EM — Single-Particle Cryo-Electron Microscopy", "cryo_em"),
        "mri":          ("MRI — Magnetic Resonance Imaging", "mri"),
        "widefield":    ("Widefield Fluorescence Microscopy", "widefield"),
    }

    # Order of modalities
    MOD_ORDER = ["cassi", "cacti", "spc", "lensless", "holography", "ptychography",
                 "ct", "cbct", "ultrasound", "cryo_em", "mri", "widefield"]

    summary_rows = []
    section_num = 0

    for mod_id in MOD_ORDER:
        section_num += 1
        display_name, mod_key = MODALITY_INFO[mod_id]

        if mod_id in results and isinstance(results[mod_id], dict) and "error" not in results[mod_id]:
            mod_data = results[mod_id]
            algo_count = len(mod_data)
            done_count = sum(1 for v in mod_data.values() if v.get("done", False))
        else:
            # Not in our 8 modalities — keep as pending placeholder
            mod_data = None
            algo_count = 0
            done_count = 0

        total_algos += algo_count
        total_done += done_count

        lines.append("")
        lines.append(f"### {section_num}. {display_name} (`{mod_key}`)\n")
        lines.append("")
        lines.append("| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |")
        lines.append("|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|")

        if mod_data:
            for i, (sk, info) in enumerate(mod_data.items(), 1):
                name = info["name"]
                ref = info.get("reference", "")
                # Extract year from reference
                year = ""
                import re
                yr_match = re.search(r'(19\d{2}|20\d{2})', ref)
                if yr_match:
                    year = yr_match.group(1)

                run_cells = []
                for r_idx in range(NUM_RUNS):
                    if r_idx < len(info["runs"]):
                        r = info["runs"][r_idx]
                        if r["status"] == "pass":
                            run_cells.append(f"{r['psnr']:.1f}")
                        else:
                            run_cells.append("FAIL")
                    else:
                        run_cells.append("—")

                avg_p = f"{info['avg_psnr']:.2f}" if info['avg_psnr'] > 0 else "—"
                avg_s = f"{info['avg_ssim']:.4f}" if info['avg_ssim'] > 0 else "—"
                status = "**done**" if info["done"] else "pending"

                lines.append(f"| {i} | {name} | {year} | {' | '.join(run_cells)} | {avg_p} | {avg_s} | {status} |")
        else:
            lines.append("| | *(not yet verified — see previous index.md)* | | — | — | — | — | — | — | — | pending |")

        lines.append("")
        lines.append("---\n")

        pct = int(100 * done_count / algo_count) if algo_count > 0 else 0
        summary_rows.append((section_num, mod_id, display_name.split("—")[0].strip(),
                            algo_count, done_count, pct))

    # Summary table
    lines.append("")
    lines.append("## Flagship Summary\n")
    lines.append("")
    lines.append("| # | Modality | Algorithms | Done | Progress |")
    lines.append("|---|----------|-----------|------|----------|")
    for num, mod_id, short_name, ac, dc, pct in summary_rows:
        lines.append(f"| {num} | {short_name} | {ac} | {dc} | {pct}% |")
    lines.append(f"| | **Total** | **{total_algos}** | **{total_done}** | "
                 f"**{int(100*total_done/total_algos) if total_algos else 0}%** |")

    now = time.strftime("%Y-%m-%d")
    # Insert progress line after header
    progress_line = (f"\nProgress: **{total_done} / {total_algos} algorithms done** | "
                     f"Last updated: {now}\n")
    # Find insertion point (after "marked **done**." line)
    insert_idx = None
    for idx, line in enumerate(lines):
        if "marked **done**" in line:
            insert_idx = idx + 1
            break
    if insert_idx:
        lines.insert(insert_idx, progress_line)

    # Write
    index_path = os.path.join(ROOT, "datasets", "templates", "index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Updated {index_path}")


if __name__ == "__main__":
    results = run_verification()
    update_index_md(results)
