#!/usr/bin/env python3
"""
SpecLab Full Algorithm Test — tests every algorithm for each modality via the website API.
Extracts PSNR/SSIM from the HTML response.
"""

import re
import time
import json
import requests
from collections import defaultdict

BASE = "http://localhost:8100"

# Key modalities to test (those with known dataset support)
MODALITIES = [
    "ct", "mri", "cassi", "cacti", "ultrasound", "oct", "pet",
    "fundus", "mammography", "holography", "spect", "cbct",
    "photoacoustic", "phase_retrieval", "ptychography",
    "confocal_3d", "lightsheet", "sem", "tem", "cryo_em",
    "sar", "hyperspectral_remote", "lidar", "spc",
    "endoscopy", "fluoroscopy", "angiography", "dexa",
    "dot", "widefield", "phase_contrast", "sim",
    "sted", "palm_storm", "flim", "fpm",
    "nerf", "gaussian_splatting",
    "ghost_imaging", "quantum_illumination",
    "coded_exposure", "lensless", "event_camera",
    "structured_light", "tof_camera",
    "afm", "stm",
    "raman_imaging", "ftir_imaging",
    "streak_camera", "xfel_sfx",
]


def get_algorithms(modality: str) -> list[str]:
    """Fetch algorithm list for a modality from the SpecLab API."""
    resp = requests.get(f"{BASE}/api/v1/spec-chat/algorithms/{modality}", timeout=10)
    if resp.status_code != 200:
        return []
    # Parse <option value="...">...</option>
    return re.findall(r'value="([^"]+)"', resp.text)


def extract_metrics(html: str) -> dict:
    """Extract PSNR and SSIM from reconstruction HTML response."""
    # Strip keepalive whitespace
    html = html.strip()

    psnr = None
    ssim = None

    # Primary: extract from the Tailwind-styled metric boxes
    # Pattern: text-blue-800 font-mono">22.72 dB</span>
    metrics = re.findall(r'text-blue-800 font-mono">([^<]+)', html)
    if len(metrics) >= 1:
        m = re.search(r'([0-9]+\.?[0-9]*)\s*dB', metrics[0])
        if m:
            psnr = float(m.group(1))
    if len(metrics) >= 2:
        m = re.search(r'([0-9]+\.?[0-9]*)', metrics[1])
        if m:
            ssim = float(m.group(1))

    # Fallback: search for PSNR/SSIM labels near values (outside base64 data)
    if psnr is None:
        m = re.search(r'>PSNR</span>\s*<span[^>]*>([0-9]+\.?[0-9]*)\s*dB', html)
        if m:
            psnr = float(m.group(1))
    if ssim is None:
        m = re.search(r'>SSIM</span>\s*<span[^>]*>([0-9]+\.?[0-9]*)', html)
        if m:
            ssim = float(m.group(1))

    # Check for errors
    error = None
    if "Reconstruction failed" in html or "bg-red-50" in html:
        err_match = re.search(r'Reconstruction failed:\s*(.+?)</div>', html)
        if err_match:
            error = err_match.group(1).strip()
        else:
            error = "Unknown error"

    if "No benchmark data available" in html or "dataset-picker" in html.lower() or "doesn't have a PWM dataset" in html:
        error = "No dataset available"

    if "Server busy" in html:
        error = "Server busy"

    return {"psnr": psnr, "ssim": ssim, "error": error}


def run_reconstruction(modality: str, algorithm: str, timeout: int = 120) -> dict:
    """Run a single reconstruction and extract metrics."""
    try:
        resp = requests.post(
            f"{BASE}/api/v1/spec-chat/reconstruct-common",
            data={"modality": modality, "algorithm": algorithm, "sample_index": 0},
            timeout=timeout,
        )
        metrics = extract_metrics(resp.text)
        metrics["status_code"] = resp.status_code
        metrics["response_length"] = len(resp.text)
        return metrics
    except requests.Timeout:
        return {"psnr": None, "ssim": None, "error": "Timeout", "status_code": 0}
    except Exception as e:
        return {"psnr": None, "ssim": None, "error": str(e), "status_code": 0}


def main():
    results = {}
    total_tests = 0
    total_pass = 0
    total_fail = 0
    total_no_data = 0

    print("=" * 80)
    print("SpecLab Full Algorithm Test")
    print("=" * 80)

    for modality in MODALITIES:
        algorithms = get_algorithms(modality)
        if not algorithms:
            print(f"\n--- {modality}: No algorithms found (skipping) ---")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {modality.upper()} — {len(algorithms)} algorithms")
        print(f"{'=' * 60}")

        mod_results = []
        for algo in algorithms:
            total_tests += 1
            print(f"  Testing {algo}...", end=" ", flush=True)

            result = run_reconstruction(modality, algo)
            result["algorithm"] = algo
            mod_results.append(result)

            if result["error"]:
                if result["error"] == "No dataset available":
                    total_no_data += 1
                    print(f"NO DATA")
                    # If first algo has no data, skip remaining for this modality
                    for remaining_algo in algorithms[algorithms.index(algo)+1:]:
                        total_tests += 1
                        total_no_data += 1
                        mod_results.append({
                            "algorithm": remaining_algo,
                            "psnr": None, "ssim": None,
                            "error": "No dataset available (skipped)",
                            "status_code": 0,
                        })
                    break
                else:
                    total_fail += 1
                    print(f"FAIL — {result['error']}")
            elif result["psnr"] is not None:
                total_pass += 1
                ssim_str = f", SSIM={result['ssim']:.4f}" if result['ssim'] else ""
                print(f"PSNR={result['psnr']:.2f} dB{ssim_str}")
            else:
                total_fail += 1
                print(f"UNKNOWN (no metrics in response, len={result.get('response_length', 0)})")

            # Small delay to avoid overwhelming the server
            time.sleep(0.5)

        results[modality] = mod_results

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests:    {total_tests}")
    print(f"Passed (PSNR):  {total_pass}")
    print(f"Failed/Error:   {total_fail}")
    print(f"No dataset:     {total_no_data}")
    print()

    # Print modality summary table
    print(f"{'Modality':<25} {'Algos':>5} {'Pass':>5} {'Fail':>5} {'NoData':>6} {'Best PSNR':>10}")
    print("-" * 60)
    for mod, mod_results in results.items():
        n_algos = len(mod_results)
        n_pass = sum(1 for r in mod_results if r["psnr"] is not None)
        n_fail = sum(1 for r in mod_results if r["error"] and r["error"] != "No dataset available" and "skipped" not in str(r["error"]))
        n_nodata = sum(1 for r in mod_results if r["error"] and ("No dataset" in str(r["error"])))
        psnrs = [r["psnr"] for r in mod_results if r["psnr"] is not None]
        best = f"{max(psnrs):.2f} dB" if psnrs else "—"
        print(f"{mod:<25} {n_algos:>5} {n_pass:>5} {n_fail:>5} {n_nodata:>6} {best:>10}")

    # Save full results to JSON
    with open("/tmp/speclab_full_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to /tmp/speclab_full_test_results.json")

    return results


if __name__ == "__main__":
    main()
