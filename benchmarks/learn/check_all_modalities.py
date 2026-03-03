#!/usr/bin/env python3
"""
Fetch every modality benchmark page from https://pwm.platformai.org/benchmark/<slug>
and generate a check.md error report in each benchmarks/learn/<folder>/ directory.

Usage:
    python benchmarks/learn/check_all_modalities.py [--modality mri] [--workers 8]
"""

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # Physics_World_Model
LEARN_DIR = ROOT / "benchmarks" / "learn"
MODALITIES_YAML = ROOT / "packages" / "pwm_core" / "contrib" / "modalities.yaml"
SOLVER_YAML = ROOT / "packages" / "pwm_core" / "contrib" / "solver_registry.yaml"
BASE_URL = "https://pwm.platformai.org/benchmark"

# ---------------------------------------------------------------------------
# Folder-name → URL-slug mapping (for the few that differ)
# ---------------------------------------------------------------------------
FOLDER_TO_SLUG = {
    "cassi": "sd_cassi",
    "spc": "spc_block",          # spc folder maps to spc_block on site
    "MRI": "mri",
}
# Most folder names == URL slugs; only override exceptions above.


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def slug_for_folder(folder: str) -> str:
    return FOLDER_TO_SLUG.get(folder, folder.lower())


# ---------------------------------------------------------------------------
# HTML Fetching
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "PWM-BenchmarkChecker/1.0 (internal QA)",
    "Accept": "text/html",
})


def fetch_page(slug: str) -> tuple[int, str]:
    """Return (status_code, html_text)."""
    url = f"{BASE_URL}/{slug}"
    try:
        resp = SESSION.get(url, timeout=30)
        return resp.status_code, resp.text
    except requests.RequestException as exc:
        return 0, str(exc)


# ---------------------------------------------------------------------------
# Checker functions — each returns a list of (severity, message) tuples.
# severity: CRITICAL | ERROR | WARNING | INFO
# ---------------------------------------------------------------------------

def check_http_status(status_code: int, slug: str) -> list[tuple[str, str]]:
    issues = []
    if status_code == 0:
        issues.append(("CRITICAL", f"Connection failed for /benchmark/{slug}"))
    elif status_code == 404:
        issues.append(("CRITICAL", f"Page not found (404) at /benchmark/{slug}"))
    elif status_code >= 500:
        issues.append(("CRITICAL", f"Server error ({status_code}) at /benchmark/{slug}"))
    elif status_code != 200:
        issues.append(("ERROR", f"Unexpected HTTP status {status_code} at /benchmark/{slug}"))
    return issues


def check_page_structure(soup: BeautifulSoup, slug: str) -> list[tuple[str, str]]:
    """Check that essential page sections exist."""
    issues = []
    text = soup.get_text(separator=" ", strip=True).lower()

    # Check for essential sections
    expected_sections = {
        "leaderboard": ["leaderboard", "ranking", "score"],
        "forward_model": ["forward model", "forward operator", "measurement model", "y =", "y="],
        "spec_ranges": ["spec", "parameter", "range", "mismatch"],
        "gallery": ["gallery", "reconstruction", "scene"],
        "scoring": ["psnr", "ssim", "score", "metric"],
    }

    for section_name, keywords in expected_sections.items():
        found = any(kw in text for kw in keywords)
        if not found:
            issues.append(("ERROR", f"Missing '{section_name}' section — none of {keywords} found in page text"))

    # Check for empty/minimal page
    if len(text) < 500:
        issues.append(("CRITICAL", f"Page content suspiciously short ({len(text)} chars) — may be empty or broken"))
    elif len(text) < 2000:
        issues.append(("WARNING", f"Page content relatively short ({len(text)} chars) — may be incomplete"))

    return issues


def check_leaderboard_consistency(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Check leaderboard data for inconsistencies."""
    issues = []
    text = soup.get_text(separator="\n", strip=True)

    # Look for score values
    scores = re.findall(r'\b0\.\d{2,4}\b', text)
    if scores:
        score_vals = [float(s) for s in scores]
        # Check for scores outside reasonable range
        for sv in score_vals:
            if sv < 0.05:
                issues.append(("WARNING", f"Very low score value found: {sv:.4f} — may indicate broken solver"))
            elif sv > 0.99:
                issues.append(("WARNING", f"Suspiciously high score: {sv:.4f} — possible data leak or evaluation bug"))

        # Check if we have enough leaderboard entries
        if len(score_vals) < 3:
            issues.append(("WARNING", f"Only {len(score_vals)} score values found — leaderboard may be incomplete"))
    else:
        issues.append(("ERROR", "No score values (0.XX pattern) found on page — leaderboard may be missing"))

    # Check for PSNR values
    psnr_vals = re.findall(r'(\d+\.?\d*)\s*dB', text)
    if psnr_vals:
        for pv in psnr_vals:
            val = float(pv)
            if val < 5:
                issues.append(("WARNING", f"Very low PSNR value: {val} dB — may indicate failed reconstruction"))
            elif val > 60:
                issues.append(("WARNING", f"Unusually high PSNR: {val} dB — verify this is correct"))
    else:
        issues.append(("INFO", "No PSNR (dB) values found in page text"))

    # Look for tier-specific leaderboards
    tier_keywords = ["public", "dev", "hidden"]
    tier_found = sum(1 for t in tier_keywords if t in text.lower())
    if tier_found < 2:
        issues.append(("WARNING", f"Only {tier_found}/3 tier keywords found — multi-tier leaderboard may be incomplete"))

    return issues


def check_forward_model(soup: BeautifulSoup, modality_cfg: dict | None) -> list[tuple[str, str]]:
    """Check forward model notation and consistency with YAML config."""
    issues = []
    text = soup.get_text(separator=" ", strip=True)

    # Check for forward model equation
    has_equation = bool(re.search(r'y\s*=', text))
    if not has_equation:
        issues.append(("WARNING", "No explicit forward model equation (y = ...) found on page"))

    # Check notation consistency — look for multiple operator symbols for same concept
    operator_patterns = [
        (r'\bH\b.*forward', r'\bA\b.*forward', "Forward operator denoted as both H and A"),
        (r'\bH\b.*operator', r'\bR\b.*operator', "Operator denoted as both H and R"),
    ]
    for pat1, pat2, msg in operator_patterns:
        if re.search(pat1, text, re.IGNORECASE) and re.search(pat2, text, re.IGNORECASE):
            issues.append(("WARNING", f"Notation inconsistency: {msg}"))

    # Cross-check with modalities.yaml
    if modality_cfg:
        yaml_eq = modality_cfg.get("forward_model_equation", "")
        if yaml_eq:
            # Check if key symbols from YAML equation appear on page
            yaml_symbols = re.findall(r'[A-Z]\w*', yaml_eq)
            missing_syms = []
            for sym in set(yaml_symbols):
                if len(sym) > 1 and sym.lower() not in text.lower():
                    missing_syms.append(sym)
            if missing_syms:
                issues.append(("INFO",
                    f"YAML forward model uses symbols {missing_syms} not found on page "
                    f"(YAML eq: {yaml_eq})"))

    return issues


def check_spec_ranges(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Check spec/parameter ranges for physical validity."""
    issues = []
    text = soup.get_text(separator="\n", strip=True)

    # Look for numeric ranges like "min to max" or "min – max" or "[min, max]"
    ranges = re.findall(
        r'(-?\d+\.?\d*)\s*(?:to|–|—|-|\.\.)\s*(-?\d+\.?\d*)', text
    )

    for lo_str, hi_str in ranges:
        lo, hi = float(lo_str), float(hi_str)
        if lo > hi:
            issues.append(("ERROR",
                f"Inverted range found: {lo} to {hi} (low > high)"))
        if lo == hi:
            issues.append(("WARNING",
                f"Degenerate range: {lo} to {hi} (zero width — no variation)"))

    # Check for negative physical quantities that should be positive
    negative_physics = re.findall(
        r'(-\d+\.?\d*)\s*(nm|mm|cm|m|Hz|MHz|GHz|keV|MeV|mA|kV|'
        r'pixel|px|voxel|degree|rad|fps|μm|µm)', text
    )
    for val, unit in negative_physics:
        fval = float(val)
        if fval < 0 and unit in ("nm", "mm", "cm", "m", "Hz", "MHz", "GHz",
                                  "keV", "MeV", "mA", "kV", "pixel", "px",
                                  "voxel", "fps", "μm", "µm"):
            issues.append(("WARNING",
                f"Negative physical quantity: {val} {unit} — verify sign is intended"))

    return issues


def check_images_and_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Check for broken image references and links."""
    issues = []

    # Check images
    imgs = soup.find_all("img")
    for img in imgs:
        src = img.get("src", "")
        alt = img.get("alt", "")
        if not src:
            issues.append(("ERROR", "Image tag with empty src attribute"))
        elif src.startswith("data:"):
            pass  # inline base64, fine
        elif not alt and not src.endswith((".svg", ".ico")):
            issues.append(("INFO", f"Image missing alt text: {src[:80]}"))

    # Check for broken internal links
    links = soup.find_all("a", href=True)
    for link in links:
        href = link["href"]
        if href.startswith("/benchmark/") and not href.strip("/").split("/")[-1]:
            issues.append(("WARNING", f"Empty benchmark link: {href}"))
        if href == "#" or href == "":
            issues.append(("INFO", f"Placeholder link found: '{link.get_text(strip=True)[:40]}'"))

    # Check for GCS image paths that might fail
    gcs_refs = re.findall(r'/gcs/[^\s"\']+', soup.decode())
    for ref in gcs_refs:
        issues.append(("INFO", f"GCS image reference: {ref[:80]} — verify it loads"))

    return issues


def check_scoring_formula(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Check scoring formula completeness."""
    issues = []
    text = soup.get_text(separator=" ", strip=True)

    # Look for scoring formula
    has_scoring = bool(re.search(r'score|scoring|composite|metric.*formula', text, re.IGNORECASE))
    if not has_scoring:
        issues.append(("WARNING", "No explicit scoring formula section found"))

    # Check if PSNR normalization is defined
    if "psnr_norm" in text.lower() or "normalized psnr" in text.lower():
        if "min" not in text.lower() or "max" not in text.lower():
            issues.append(("WARNING",
                "PSNR normalization mentioned but min/max bounds not defined"))

    # Check for undefined variables in formulas
    formula_match = re.search(r'(?:score|Score)\s*=\s*([^\n.]{10,80})', text)
    if formula_match:
        formula = formula_match.group(1)
        # Look for variables that might be undefined
        vars_in_formula = re.findall(r'[A-Za-z_]\w*', formula)
        common_defined = {"psnr", "ssim", "score", "norm", "max", "min", "val",
                          "mean", "std", "sum", "abs", "log", "exp"}
        undefined = [v for v in vars_in_formula
                     if v.lower() not in common_defined and len(v) > 1]
        if undefined:
            issues.append(("INFO",
                f"Formula variables that may need definition: {undefined}"))

    return issues


def check_algorithm_names(soup: BeautifulSoup, solver_cfg: dict | None) -> list[tuple[str, str]]:
    """Cross-check algorithm names between page and solver registry."""
    issues = []
    text = soup.get_text(separator=" ", strip=True)

    if solver_cfg:
        yaml_names = set()
        for tier_key, tier_val in solver_cfg.items():
            if isinstance(tier_val, dict):
                name = tier_val.get("name", "")
                if name:
                    yaml_names.add(name)

        for name in yaml_names:
            # Check if the solver name (or close variant) appears on the page
            name_lower = name.lower().replace("-", " ").replace("_", " ")
            text_lower = text.lower()
            # Try exact and fuzzy match
            if name_lower not in text_lower:
                # Try first word only
                first_word = name_lower.split()[0] if name_lower.split() else ""
                if first_word and first_word not in text_lower:
                    issues.append(("INFO",
                        f"Solver '{name}' from registry not found on page"))

    return issues


def check_dataset_info(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Check dataset/submission documentation."""
    issues = []
    text = soup.get_text(separator=" ", strip=True).lower()

    # Check for dataset format info
    format_keywords = ["hdf5", "h5", "npy", "npz", "tiff", "png", "csv",
                       "json", "format", "shape"]
    has_format = any(kw in text for kw in format_keywords)
    if not has_format:
        issues.append(("WARNING", "No dataset format specification (HDF5/npy/shape) found"))

    # Check for sample count
    sample_patterns = re.findall(r'(\d+)\s*(?:sample|scene|phantom|image|volume)s?', text)
    if not sample_patterns:
        issues.append(("INFO", "No explicit sample/scene count mentioned"))

    # Check for submission instructions
    submission_keywords = ["submit", "docker", "upload", "evaluation", "submission"]
    has_submission = any(kw in text for kw in submission_keywords)
    if not has_submission:
        issues.append(("INFO", "No submission instructions found"))

    return issues


def check_physics_consistency(soup: BeautifulSoup, modality_cfg: dict | None) -> list[tuple[str, str]]:
    """Cross-check physics parameters between page and YAML config."""
    issues = []
    if not modality_cfg:
        return issues

    text = soup.get_text(separator=" ", strip=True)

    # Check signal dimensions
    sig_dims = modality_cfg.get("signal_dims", {})
    if sig_dims:
        x_shape = sig_dims.get("x", [])
        y_shape = sig_dims.get("y", [])
        if x_shape:
            shape_str = "×".join(str(d) for d in x_shape)
            alt_shape_str = "x".join(str(d) for d in x_shape)
            if shape_str not in text and alt_shape_str not in text:
                # Try individual dims
                all_found = all(str(d) in text for d in x_shape)
                if not all_found:
                    issues.append(("INFO",
                        f"Signal shape {x_shape} from YAML not clearly shown on page"))

    # Check category match
    yaml_category = modality_cfg.get("category", "")
    if yaml_category:
        if yaml_category.lower().replace("_", " ") not in text.lower():
            # Not critical — category might use different wording
            pass

    # Check wavelength range if applicable
    wl_range = modality_cfg.get("wavelength_range_nm")
    if wl_range and len(wl_range) == 2:
        lo, hi = wl_range
        if str(lo) not in text and str(hi) not in text:
            issues.append(("INFO",
                f"Wavelength range {lo}–{hi} nm from YAML not found on page"))

    return issues


def check_cross_tier_degradation(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Check that hidden tier shows worse metrics than public (expected behavior)."""
    issues = []
    text = soup.get_text(separator="\n", strip=True)

    # Try to find PSNR values associated with tiers
    # This is heuristic — look for patterns like "Public: XX.X dB" or tabular data
    tier_psnr = {}
    for tier in ["public", "dev", "hidden"]:
        # Look for PSNR near tier name
        pattern = rf'{tier}[^.]*?(\d{{2,3}}\.?\d*)\s*(?:dB|PSNR)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            tier_psnr[tier] = float(match.group(1))

    if len(tier_psnr) >= 2:
        if "public" in tier_psnr and "hidden" in tier_psnr:
            if tier_psnr["hidden"] > tier_psnr["public"] + 3:
                issues.append(("WARNING",
                    f"Hidden tier PSNR ({tier_psnr['hidden']:.1f}) higher than "
                    f"Public ({tier_psnr['public']:.1f}) — expected degradation "
                    f"with stronger mismatch"))
        if "public" in tier_psnr and "dev" in tier_psnr:
            if tier_psnr["dev"] > tier_psnr["public"] + 3:
                issues.append(("WARNING",
                    f"Dev tier PSNR ({tier_psnr['dev']:.1f}) higher than "
                    f"Public ({tier_psnr['public']:.1f}) — unexpected"))

    return issues


# ---------------------------------------------------------------------------
# Main checker orchestration
# ---------------------------------------------------------------------------

def check_modality(
    folder: str,
    modalities_cfg: dict,
    solvers_cfg: dict,
) -> dict[str, Any]:
    """Run all checks for a single modality. Returns dict with issues."""
    slug = slug_for_folder(folder)
    modality_cfg = modalities_cfg.get(folder.lower()) or modalities_cfg.get(slug)
    solver_cfg = solvers_cfg.get(folder.lower()) or solvers_cfg.get(slug)

    result = {
        "folder": folder,
        "slug": slug,
        "url": f"{BASE_URL}/{slug}",
        "issues": [],
        "status_code": 0,
        "page_length": 0,
    }

    # Fetch
    status_code, html = fetch_page(slug)
    result["status_code"] = status_code
    result["page_length"] = len(html)

    # HTTP check
    result["issues"].extend(check_http_status(status_code, slug))
    if status_code != 200:
        return result

    soup = BeautifulSoup(html, "lxml")

    # Run all checkers
    checkers = [
        ("Page Structure", lambda: check_page_structure(soup, slug)),
        ("Leaderboard Consistency", lambda: check_leaderboard_consistency(soup)),
        ("Forward Model", lambda: check_forward_model(soup, modality_cfg)),
        ("Spec Ranges", lambda: check_spec_ranges(soup)),
        ("Images & Links", lambda: check_images_and_links(soup)),
        ("Scoring Formula", lambda: check_scoring_formula(soup)),
        ("Algorithm Names", lambda: check_algorithm_names(soup, solver_cfg)),
        ("Dataset Info", lambda: check_dataset_info(soup)),
        ("Physics Consistency", lambda: check_physics_consistency(soup, modality_cfg)),
        ("Cross-Tier Degradation", lambda: check_cross_tier_degradation(soup)),
    ]

    for checker_name, checker_fn in checkers:
        try:
            checker_issues = checker_fn()
            for severity, msg in checker_issues:
                result["issues"].append({
                    "checker": checker_name,
                    "severity": severity,
                    "message": msg,
                })
        except Exception as exc:
            result["issues"].append({
                "checker": checker_name,
                "severity": "ERROR",
                "message": f"Checker crashed: {exc}",
            })

    return result


def write_check_md(result: dict, learn_dir: Path) -> Path:
    """Write check.md into the modality's learn folder."""
    folder = result["folder"]
    out_dir = learn_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "check.md"

    lines = [
        f"# Benchmark QA Check — {folder}",
        "",
        f"**URL:** {result['url']}",
        f"**HTTP Status:** {result['status_code']}",
        f"**Page Size:** {result['page_length']:,} bytes",
        f"**Check Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
    ]

    issues = result["issues"]
    if not issues:
        lines.append("> No issues found. Page looks healthy.")
        lines.append("")
    else:
        # Count by severity
        counts = {}
        for iss in issues:
            sev = iss["severity"]
            counts[sev] = counts.get(sev, 0) + 1

        lines.append("## Summary")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
            if sev in counts:
                lines.append(f"| {sev} | {counts[sev]} |")
        lines.append("")

        # Group by checker
        by_checker: dict[str, list] = {}
        for iss in issues:
            checker = iss["checker"]
            if checker not in by_checker:
                by_checker[checker] = []
            by_checker[checker].append(iss)

        lines.append("## Issues by Category")
        lines.append("")

        for checker_name, checker_issues in by_checker.items():
            lines.append(f"### {checker_name}")
            lines.append("")
            for iss in checker_issues:
                sev = iss["severity"]
                msg = iss["message"]
                icon = {"CRITICAL": "🔴", "ERROR": "🟠", "WARNING": "🟡", "INFO": "🔵"}.get(sev, "⚪")
                lines.append(f"- {icon} **{sev}**: {msg}")
            lines.append("")

    lines.append("---")
    lines.append("*Auto-generated by `benchmarks/learn/check_all_modalities.py`*")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check all modality benchmark pages")
    parser.add_argument("--modality", type=str, default=None,
                        help="Check a single modality folder name")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent HTTP workers")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Also write full results to JSON file")
    args = parser.parse_args()

    # Load configs
    print("Loading modalities.yaml and solver_registry.yaml ...")
    modalities_cfg = load_yaml(MODALITIES_YAML).get("modalities", {})
    solvers_cfg = load_yaml(SOLVER_YAML)

    # Determine which folders to check
    if args.modality:
        folders = [args.modality]
    else:
        folders = sorted(
            d.name for d in LEARN_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    print(f"Checking {len(folders)} modalities with {args.workers} workers ...")
    print()

    all_results = []
    total_issues = {"CRITICAL": 0, "ERROR": 0, "WARNING": 0, "INFO": 0}
    checked = 0

    def process_one(folder: str) -> dict:
        return check_modality(folder, modalities_cfg, solvers_cfg)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_folder = {
            pool.submit(process_one, f): f for f in folders
        }
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "folder": folder,
                    "slug": slug_for_folder(folder),
                    "url": f"{BASE_URL}/{slug_for_folder(folder)}",
                    "issues": [{"checker": "Runner", "severity": "CRITICAL",
                                "message": f"Uncaught exception: {exc}"}],
                    "status_code": 0,
                    "page_length": 0,
                }

            all_results.append(result)
            checked += 1

            # Write check.md
            md_path = write_check_md(result, LEARN_DIR)

            # Tally
            n_issues = len(result["issues"])
            for iss in result["issues"]:
                sev = iss["severity"] if isinstance(iss, dict) else iss[0]
                total_issues[sev] = total_issues.get(sev, 0) + 1

            status = result["status_code"]
            crit = sum(1 for i in result["issues"]
                       if (i["severity"] if isinstance(i, dict) else i[0]) == "CRITICAL")
            errs = sum(1 for i in result["issues"]
                       if (i["severity"] if isinstance(i, dict) else i[0]) == "ERROR")
            warns = sum(1 for i in result["issues"]
                        if (i["severity"] if isinstance(i, dict) else i[0]) == "WARNING")

            tag = "OK" if crit == 0 and errs == 0 else "FAIL"
            print(f"  [{checked:3d}/{len(folders)}] {tag:4s}  HTTP {status}  "
                  f"C={crit} E={errs} W={warns}  {folder}")

    # Summary
    print()
    print("=" * 60)
    print(f"Checked: {len(all_results)} modalities")
    for sev in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
        print(f"  {sev:10s}: {total_issues.get(sev, 0)}")
    print(f"  check.md files written: {len(all_results)}")
    print("=" * 60)

    # Optionally dump JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Full results: {args.output_json}")

    # Exit code
    if total_issues.get("CRITICAL", 0) > 0:
        sys.exit(2)
    elif total_issues.get("ERROR", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
