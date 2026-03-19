#!/usr/bin/env python3
"""Thorough quality check for a single modality on the live benchmark site.

Usage:
    python3 scripts/check_modality.py <modality_id>
    python3 scripts/check_modality.py --all
    python3 scripts/check_modality.py --list-errors

Writes/revises check.md in benchmarks/learn/<modality_id>/check.md
"""
import argparse
import re
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://pwm.platformai.org"
REPO_ROOT = Path(__file__).resolve().parent.parent
LEARN_DIR = REPO_ROOT / "benchmarks" / "learn"

# Parent modalities that have sub-variants instead of own page
PARENT_MODALITIES = {
    "spc": ["spc_block", "spc_kronecker"],
}


def fetch(url: str, timeout: int = 15):
    """Fetch URL, return (status_code, body_text)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PWM-Checker/2.0"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return 0, str(e)


def check_url(url: str, timeout: int = 10):
    """Check if URL is accessible via GET with Range header. Returns status code."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "PWM-Checker/2.0",
            "Range": "bytes=0-100",
        })
        resp = urllib.request.urlopen(req, timeout=timeout)
        resp.read()
        return resp.status  # 200 or 206
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def check_modality(modality_id: str) -> dict:
    """Run comprehensive checks for a modality."""
    results = {
        "modality_id": modality_id,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "errors": [],
        "warnings": [],
        "info": [],
        "passed": [],
    }

    # Handle parent modalities
    if modality_id in PARENT_MODALITIES:
        subs = PARENT_MODALITIES[modality_id]
        results["info"].append(f"Parent modality — sub-variants: {', '.join(subs)}")
        for sv in subs:
            st, _ = fetch(f"{BASE_URL}/benchmark/{sv}")
            if st == 200:
                results["passed"].append(f"Sub-variant /benchmark/{sv} loads (HTTP 200)")
            else:
                results["errors"].append(f"Sub-variant /benchmark/{sv} HTTP {st}")
        # Still check learn materials
        _check_learn_materials(modality_id, results)
        return results

    # ── 1. Main page ──
    status, body = fetch(f"{BASE_URL}/benchmark/{modality_id}")
    page_size = len(body)
    results["info"].append(f"Page size: {page_size:,} bytes")

    if status != 200:
        results["errors"].append(f"Main page HTTP {status}")
        _check_learn_materials(modality_id, results)
        return results
    results["passed"].append(f"Main page loads (HTTP {status})")

    # ── 2. Page title ──
    title_m = re.search(r"<title>(.+?)</title>", body)
    if title_m:
        title = title_m.group(1).strip()
        if "Error" in title or "404" in title or "500" in title:
            results["errors"].append(f"Page title indicates error: {title}")
        else:
            results["passed"].append(f"Title: {title}")
    else:
        results["warnings"].append("No <title> tag found")

    # ── 3. Description ──
    desc_m = re.search(r'<p class="mt-1 text-sm text-gray-500">([^<]+)</p>', body)
    if desc_m:
        desc = desc_m.group(1).strip()
        results["passed"].append(f"Description: {desc[:80]}")
    else:
        results["warnings"].append("No description subtitle found")

    # ── 4. Spec notation ──
    spec_m = re.search(r'font-mono text-indigo[^>]*>([^<]+)<', body)
    if spec_m:
        spec = spec_m.group(1).strip()
        if len(spec) < 3:
            results["warnings"].append(f"Spec notation very short: '{spec}'")
        else:
            results["passed"].append(f"Spec notation: {spec[:80]}")
    else:
        results["warnings"].append("No spec notation found")

    # ── 5. Leaderboard ──
    if "Challenge Leaderboard" in body:
        results["passed"].append("Challenge Leaderboard section present")

        # Count entries (method names)
        methods = re.findall(
            r'class="px-4 py-3 font-semibold text-gray-800">([^<]+)<', body)
        if methods:
            results["passed"].append(f"Leaderboard: {len(methods)} entries")
            results["info"].append(f"Methods: {', '.join(methods[:4])}")
        else:
            results["warnings"].append("Leaderboard section present but no entries parsed")

        # Overall scores (must have decimal point to exclude rank numbers)
        overall_scores = re.findall(
            r'font-bold text-amber-800">(\d+\.\d+)</td>', body)
        for s in overall_scores:
            val = float(s)
            if val > 1.0:
                results["warnings"].append(f"Overall score {val} > 1.0 (outside [0,1] range)")

        # Leaderboard PSNR (in score cells: "XX.XX dB / 0.XXX")
        lb_psnr = re.findall(
            r'text-gray-400[^>]*>\s*([\d.]+)\s*dB\s*/\s*([\d.]+)', body)
        for psnr_s, ssim_s in lb_psnr:
            psnr = float(psnr_s)
            ssim = float(ssim_s)
            if psnr > 60:
                results["warnings"].append(
                    f"Leaderboard PSNR {psnr} dB > 60 (unrealistically high)")
            if ssim > 1.0:
                results["warnings"].append(
                    f"Leaderboard SSIM {ssim} > 1.0 (invalid)")

        # Check source citations
        sources = re.findall(r'text-xs text-gray-500">([^<]+)</td>', body)
        n_empty = sum(1 for s in sources if not s.strip() or s.strip() == "—")
        if n_empty > 0:
            results["info"].append(f"{n_empty} leaderboard entries missing source citation")
    else:
        results["errors"].append("No Challenge Leaderboard section found")

    # ── 6. Data Preview / Gallery ──
    if "Data Preview" in body or "Gallery" in body:
        results["passed"].append("Data Preview / Gallery section present")
    else:
        results["warnings"].append("No Data Preview / Gallery section")

    # ── 7. Verify ALL gallery images (4 scenes × gt + measurement + recon) ──
    gcs_refs = re.findall(r'/gcs/img/benchmark_gallery/[^"\']+\.png', body)
    unique_refs = sorted(set(gcs_refs))
    n_ok = 0
    n_fail = 0
    failed_imgs = []
    for ref in unique_refs:
        url = f"{BASE_URL}{ref}"
        st = check_url(url)
        if st in (200, 206):
            n_ok += 1
        else:
            n_fail += 1
            failed_imgs.append(f"{ref} → HTTP {st}")
    if n_ok > 0:
        results["passed"].append(f"Gallery images: {n_ok}/{n_ok + n_fail} load OK")
    if n_fail > 0:
        for fi in failed_imgs[:5]:  # cap at 5
            results["errors"].append(f"Gallery image broken: {fi}")
        if n_fail > 5:
            results["errors"].append(f"... and {n_fail - 5} more broken images")

    # ── 8. Challenge tier pages ──
    for tier in ["public", "dev"]:
        tier_url = f"{BASE_URL}/benchmark/{modality_id}/challenge/{tier}"
        tier_st, tier_body = fetch(tier_url)
        if tier_st == 200:
            results["passed"].append(f"Challenge {tier} page loads (HTTP 200)")
            if ".h5" in tier_body:
                results["passed"].append(f"Challenge {tier} has HDF5 reference")
            else:
                results["warnings"].append(
                    f"Challenge {tier} page has no HDF5 reference")
        else:
            results["errors"].append(f"Challenge {tier} page HTTP {tier_st}")

    # ── 9. Challenge HDF5 on GCS ──
    for tier in ["public", "dev"]:
        h5_url = (f"{BASE_URL}/gcs/challenge-data/v1.0/"
                  f"{modality_id}_challenge_{tier}.h5")
        h5_st = check_url(h5_url)
        if h5_st in (200, 206):
            results["passed"].append(f"Challenge {tier} HDF5 on GCS OK")
        else:
            results["errors"].append(f"Challenge {tier} HDF5 HTTP {h5_st}")

    # ── 10. Compete & Contribute pages ──
    for page in ["compete", "contribute"]:
        pg_url = f"{BASE_URL}/benchmark/{modality_id}/{page}"
        pg_st, _ = fetch(pg_url)
        if pg_st == 200:
            results["passed"].append(f"{page.capitalize()} page loads (HTTP 200)")
        else:
            results["warnings"].append(f"{page.capitalize()} page HTTP {pg_st}")

    # ── 11. Forward model equation ──
    if re.search(r'y\s*=\s*[HAΦ]', body) or "forward" in body.lower():
        results["passed"].append("Forward model reference found")
    else:
        results["info"].append("No explicit forward model equation on page")

    # ── 12. Learning materials ──
    _check_learn_materials(modality_id, results)

    return results


def _check_learn_materials(modality_id, results):
    """Check learning materials exist and have content."""
    learn_path = LEARN_DIR / modality_id
    if not learn_path.exists():
        results["errors"].append("Learning materials directory missing")
        return

    results["passed"].append("Learning materials directory exists")
    expected = [
        "README.md",
        "01_physics_fundamentals.md",
        "02_forward_model.md",
        "03_reconstruction_algorithms.md",
        "04_pwm_benchmark.md",
        "05_hands_on_tutorial.md",
    ]
    for f in expected:
        fp = learn_path / f
        if fp.exists():
            size = fp.stat().st_size
            if size < 100:
                results["warnings"].append(f"Learn file {f} very small ({size} B)")
            else:
                results["passed"].append(f"Learn: {f} ({size:,} B)")
        else:
            results["errors"].append(f"Learn file {f} MISSING")


def generate_check_md(results: dict) -> str:
    """Generate check.md content from results dict."""
    lines = []
    mid = results["modality_id"]
    ts = results["checked_at"]

    n_err = len(results["errors"])
    n_warn = len(results["warnings"])
    n_info = len(results["info"])
    n_pass = len(results["passed"])

    if n_err == 0 and n_warn == 0:
        status_icon = "PASS"
    elif n_err == 0:
        status_icon = "WARN"
    else:
        status_icon = "FAIL"

    lines.append(f"# Benchmark QA Check — {mid}")
    lines.append(f"")
    lines.append(f"**URL:** {BASE_URL}/benchmark/{mid}")
    lines.append(f"**Check Date:** {ts}")
    lines.append(f"**Status:** {status_icon}")
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Severity | Count |")
    lines.append(f"|----------|-------|")
    lines.append(f"| ERROR | {n_err} |")
    lines.append(f"| WARNING | {n_warn} |")
    lines.append(f"| INFO | {n_info} |")
    lines.append(f"| PASSED | {n_pass} |")
    lines.append(f"")

    if results["errors"]:
        lines.append(f"## Errors")
        lines.append(f"")
        for e in results["errors"]:
            lines.append(f"- [ ] {e}")
        lines.append(f"")

    if results["warnings"]:
        lines.append(f"## Warnings")
        lines.append(f"")
        for w in results["warnings"]:
            lines.append(f"- [ ] {w}")
        lines.append(f"")

    if results["info"]:
        lines.append(f"## Info")
        lines.append(f"")
        for i in results["info"]:
            lines.append(f"- {i}")
        lines.append(f"")

    if results["passed"]:
        lines.append(f"## Passed Checks")
        lines.append(f"")
        for p in results["passed"]:
            lines.append(f"- [x] {p}")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Generated by `scripts/check_modality.py` v2*")
    return "\n".join(lines)


def pull_and_push():
    """Pull latest from remote, push our changes."""
    import subprocess
    for attempt in range(8):
        # Stash local unstaged changes
        subprocess.run(["git", "stash"], capture_output=True)
        # Pull rebase
        r = subprocess.run(
            ["git", "pull", "--rebase", "origin", "master"],
            capture_output=True, text=True)
        # If conflicts, accept theirs for check.md files and continue
        if r.returncode != 0:
            # Get conflicted files
            cr = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                capture_output=True, text=True)
            conflicts = [f.strip() for f in cr.stdout.strip().split("\n") if f.strip()]
            if conflicts:
                subprocess.run(["git", "checkout", "--theirs"] + conflicts,
                               capture_output=True)
                subprocess.run(["git", "add"] + conflicts, capture_output=True)
                subprocess.run(["git", "rebase", "--continue"],
                               capture_output=True, env={**dict(__import__('os').environ),
                                                          "GIT_EDITOR": "true"})
        # Pop stash
        subprocess.run(["git", "stash", "pop"], capture_output=True)
        # Try push
        r = subprocess.run(
            ["git", "push", "origin", "master"],
            capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  Push OK (attempt {attempt + 1})")
            return True
        import time
        time.sleep(1)
    print("  Push failed after 8 attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description="Check modality quality")
    parser.add_argument("modality", nargs="?", help="Modality ID to check")
    parser.add_argument("--all", action="store_true", help="Check all modalities")
    parser.add_argument("--list-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Git commit + push after each modality")
    args = parser.parse_args()

    if args.list_errors:
        for d in sorted(LEARN_DIR.iterdir()):
            if not d.is_dir():
                continue
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
        modalities = sorted(d.name for d in LEARN_DIR.iterdir()
                            if d.is_dir() and not d.name.startswith("."))
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

        tag = "FAIL" if n_err else ("WARN" if n_warn else "PASS")
        print(f"  [{tag}] Passed:{n_pass} Warn:{n_warn} Err:{n_err}")

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

        if args.push and not args.dry_run:
            import subprocess
            subprocess.run(["git", "add", str(LEARN_DIR / mid / "check.md")],
                           capture_output=True)
            msg = f"check({mid}): {tag} — {n_pass}P/{n_warn}W/{n_err}E"
            subprocess.run(
                ["git", "commit", "-m", msg + "\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"],
                capture_output=True)
            pull_and_push()

    print(f"\nDone. Checked {len(modalities)} modalities.")


if __name__ == "__main__":
    main()
