#!/usr/bin/env python3
"""Check a single modality for errors on the live benchmark site.

Usage:
    python3 scripts/check_modality.py <modality_id>
    python3 scripts/check_modality.py --all          # check all 169
    python3 scripts/check_modality.py --list-errors   # list modalities with errors

Writes check.md into benchmarks/learn/<modality_id>/check.md
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://pwm.platformai.org"
REPO_ROOT = Path(__file__).resolve().parent.parent
LEARN_DIR = REPO_ROOT / "benchmarks" / "learn"


def fetch(url: str, timeout: int = 15):
    """Fetch URL, return (status_code, body_text)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PWM-Checker/1.0"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return 0, str(e)


def check_head(url: str, timeout: int = 10):
    """Check if URL is accessible (GET with range to avoid downloading full file)."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "PWM-Checker/1.0",
            "Range": "bytes=0-100",
        })
        resp = urllib.request.urlopen(req, timeout=timeout)
        resp.read()  # consume
        return resp.status  # 200 or 206
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def check_modality(modality_id: str) -> dict:
    """Run all checks for a modality, return results dict."""
    results = {
        "modality_id": modality_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": [],
        "passed": [],
    }

    # 1. Main benchmark page
    status, body = fetch(f"{BASE_URL}/benchmark/{modality_id}")
    if status != 200:
        results["errors"].append(f"Main page returned HTTP {status}")
        return results  # can't check further
    results["passed"].append(f"Main page loads (HTTP {status})")

    # 2. Check page title / header
    title_match = re.search(r"<title>(.+?)</title>", body)
    if title_match:
        title = title_match.group(1)
        if "Error" in title or "404" in title:
            results["errors"].append(f"Page title indicates error: {title}")
        else:
            results["passed"].append(f"Page title: {title}")

    # 3. Check leaderboard
    if "Challenge Leaderboard" in body:
        results["passed"].append("Challenge Leaderboard section present")
        # Count leaderboard entries
        lb_entries = body.count('class="px-4 py-3 font-semibold text-gray-800">')
        if lb_entries == 0:
            results["warnings"].append("Leaderboard section present but no entries found")
        else:
            results["passed"].append(f"Leaderboard has {lb_entries} entries")

        # Check for realistic scores
        scores = re.findall(r'font-bold text-amber-800">(\d+\.\d+)</td>', body)
        for s in scores:
            val = float(s)
            if val > 1.0:
                results["warnings"].append(f"Overall score {val} > 1.0 (unusual)")
            if val < 0.01:
                results["warnings"].append(f"Overall score {val} < 0.01 (suspiciously low)")
    else:
        results["errors"].append("No Challenge Leaderboard section found")

    # 4. Check PSNR values
    psnr_vals = re.findall(r'([\d.]+)\s*dB', body)
    for p in psnr_vals:
        val = float(p)
        if val > 60:
            results["warnings"].append(f"PSNR {val} dB > 60 (unrealistically high)")
        if val < 5:
            results["warnings"].append(f"PSNR {val} dB < 5 (unrealistically low)")

    # 5. Check spec notation
    if "font-mono text-indigo" in body:
        spec_match = re.search(r'font-mono text-indigo[^>]*>([^<]+)<', body)
        if spec_match:
            spec = spec_match.group(1).strip()
            if len(spec) < 5:
                results["warnings"].append(f"Spec notation very short: '{spec}'")
            else:
                results["passed"].append(f"Spec notation present: {spec[:60]}")
    else:
        results["warnings"].append("No spec notation found")

    # 6. Check description
    desc_match = re.search(r'<p class="mt-1 text-sm text-gray-500">([^<]+)</p>', body)
    if desc_match:
        desc = desc_match.group(1).strip()
        if len(desc) < 3:
            results["warnings"].append(f"Description very short: '{desc}'")
        else:
            results["passed"].append(f"Description: {desc[:80]}")
    else:
        results["warnings"].append("No description found")

    # 7. Check Data Preview section
    if "Data Preview" in body or "data-preview" in body.lower() or "Gallery" in body:
        results["passed"].append("Data Preview / Gallery section present")
    else:
        results["warnings"].append("No Data Preview / Gallery section found")

    # 8. Check gallery images (scene_00 gt.png)
    gt_url = f"{BASE_URL}/gcs/img/benchmark_gallery/{modality_id}/scene_00/gt.png"
    gt_status = check_head(gt_url)
    if gt_status in (200, 206):
        results["passed"].append("Gallery gt.png (scene_00) loads")
    else:
        results["warnings"].append(f"Gallery gt.png (scene_00) HTTP {gt_status}")

    # Check recon images
    for recon in ["recon_I.png", "recon_II.png", "recon_III.png"]:
        recon_url = f"{BASE_URL}/gcs/img/benchmark_gallery/{modality_id}/scene_00/{recon}"
        recon_status = check_head(recon_url)
        if recon_status in (200, 206):
            results["passed"].append(f"Gallery {recon} (scene_00) loads")
        else:
            results["warnings"].append(f"Gallery {recon} (scene_00) HTTP {recon_status}")

    # 9. Check challenge tier pages
    for tier in ["public", "dev"]:
        tier_url = f"{BASE_URL}/benchmark/{modality_id}/challenge/{tier}"
        tier_status, tier_body = fetch(tier_url)
        if tier_status == 200:
            results["passed"].append(f"Challenge {tier} page loads (HTTP 200)")
            # Check if it has download link
            if ".h5" in tier_body or "download" in tier_body.lower():
                results["passed"].append(f"Challenge {tier} page has dataset reference")
            else:
                results["warnings"].append(f"Challenge {tier} page has no dataset download link")
        else:
            results["errors"].append(f"Challenge {tier} page HTTP {tier_status}")

    # 10. Check challenge HDF5 on GCS (via proxy — HEAD only)
    h5_url = f"{BASE_URL}/gcs/challenge-data/v1.0/{modality_id}_challenge_public.h5"
    h5_status = check_head(h5_url)
    if h5_status in (200, 206):
        results["passed"].append("Public challenge HDF5 accessible on GCS")
    else:
        results["errors"].append(f"Public challenge HDF5 HTTP {h5_status}")

    # 11. Check learning materials
    learn_path = LEARN_DIR / modality_id
    if learn_path.exists():
        results["passed"].append("Learning materials directory exists")
        expected_files = [
            "README.md",
            "01_physics_fundamentals.md",
            "02_forward_model.md",
            "03_reconstruction_algorithms.md",
            "04_pwm_benchmark.md",
            "05_hands_on_tutorial.md",
        ]
        for f in expected_files:
            fp = learn_path / f
            if fp.exists():
                size = fp.stat().st_size
                if size < 100:
                    results["warnings"].append(f"Learn file {f} very small ({size} bytes)")
                else:
                    results["passed"].append(f"Learn file {f} exists ({size} bytes)")
            else:
                results["errors"].append(f"Learn file {f} missing")
    else:
        results["errors"].append("Learning materials directory missing")

    # 12. Check compete and contribute pages
    for page in ["compete", "contribute"]:
        page_url = f"{BASE_URL}/benchmark/{modality_id}/{page}"
        page_status, _ = fetch(page_url)
        if page_status == 200:
            results["passed"].append(f"{page.capitalize()} page loads (HTTP 200)")
        else:
            results["warnings"].append(f"{page.capitalize()} page HTTP {page_status}")

    return results


def generate_check_md(results: dict) -> str:
    """Generate check.md content from results dict."""
    lines = []
    mid = results["modality_id"]
    ts = results["checked_at"]

    # Header
    lines.append(f"# Quality Check: `{mid}`")
    lines.append(f"")
    lines.append(f"**Checked:** {ts}")
    lines.append(f"**URL:** {BASE_URL}/benchmark/{mid}")
    lines.append(f"")

    # Summary
    n_err = len(results["errors"])
    n_warn = len(results["warnings"])
    n_pass = len(results["passed"])

    if n_err == 0 and n_warn == 0:
        status = "PASS"
    elif n_err == 0:
        status = "WARN"
    else:
        status = "FAIL"

    lines.append(f"## Status: {status}")
    lines.append(f"")
    lines.append(f"| Category | Count |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Passed | {n_pass} |")
    lines.append(f"| Warnings | {n_warn} |")
    lines.append(f"| Errors | {n_err} |")
    lines.append(f"")

    # Errors
    if results["errors"]:
        lines.append(f"## Errors")
        lines.append(f"")
        for e in results["errors"]:
            lines.append(f"- [ ] {e}")
        lines.append(f"")

    # Warnings
    if results["warnings"]:
        lines.append(f"## Warnings")
        lines.append(f"")
        for w in results["warnings"]:
            lines.append(f"- [ ] {w}")
        lines.append(f"")

    # Passed
    if results["passed"]:
        lines.append(f"## Passed Checks")
        lines.append(f"")
        for p in results["passed"]:
            lines.append(f"- [x] {p}")
        lines.append(f"")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check modality quality")
    parser.add_argument("modality", nargs="?", help="Modality ID to check")
    parser.add_argument("--all", action="store_true", help="Check all modalities")
    parser.add_argument("--list-errors", action="store_true",
                        help="List modalities with existing errors")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results but don't write check.md")
    args = parser.parse_args()

    if args.list_errors:
        for d in sorted(LEARN_DIR.iterdir()):
            ck = d / "check.md"
            if ck.exists():
                content = ck.read_text()
                if "## Errors" in content:
                    print(f"FAIL  {d.name}")
                elif "## Warnings" in content:
                    print(f"WARN  {d.name}")
                else:
                    print(f"PASS  {d.name}")
        return

    if args.all:
        modalities = sorted(d.name for d in LEARN_DIR.iterdir() if d.is_dir()
                            and not d.name.startswith("."))
    elif args.modality:
        modalities = [args.modality]
    else:
        parser.error("Provide a modality ID or --all")
        return

    for mid in modalities:
        print(f"\n{'='*60}")
        print(f"Checking: {mid}")
        print(f"{'='*60}")

        results = check_modality(mid)
        md_content = generate_check_md(results)

        n_err = len(results["errors"])
        n_warn = len(results["warnings"])
        n_pass = len(results["passed"])

        print(f"  Passed: {n_pass}  Warnings: {n_warn}  Errors: {n_err}")

        if results["errors"]:
            for e in results["errors"]:
                print(f"  ERROR: {e}")
        if results["warnings"]:
            for w in results["warnings"]:
                print(f"  WARN:  {w}")

        if not args.dry_run:
            out_path = LEARN_DIR / mid / "check.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md_content)
            print(f"  Wrote: {out_path}")

    print(f"\nDone. Checked {len(modalities)} modalities.")


if __name__ == "__main__":
    main()
