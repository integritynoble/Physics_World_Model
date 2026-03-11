#!/usr/bin/env python3
"""Update check.md files with algorithm test results from GPU server."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEARN_DIR = ROOT / "benchmarks" / "learn"
RESULTS_PATH = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"

# Load test results
with open(RESULTS_PATH) as f:
    results = json.load(f)

ALGO_TEST_SECTION = """
---

## GPU Server Algorithm Test Results

**Test Date:** {timestamp}
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
{solver_rows}

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
"""

updated = 0
for mod_id, mod_data in results["modalities"].items():
    check_path = LEARN_DIR / mod_id / "check.md"
    if not check_path.exists():
        print(f"  {mod_id}: check.md not found, skipping")
        continue

    # Build solver rows
    rows = []
    for sname, sres in mod_data["solvers"].items():
        status = sres.get("status", "unknown")
        if status == "completed":
            psnr = sres.get("psnr_db", 0)
            ssim = sres.get("ssim", 0)
            t = sres.get("exec_time_sec", 0)
            rows.append(f"| {sname} | {psnr:.2f} | {ssim:.4f} | {t:.2f} | PASS |")
        else:
            rows.append(f"| {sname} | - | - | - | {status} |")

    section = ALGO_TEST_SECTION.format(
        timestamp=results["timestamp"][:19],
        solver_rows="\n".join(rows)
    )

    # Read existing check.md
    content = check_path.read_text(encoding="utf-8")

    # Remove old GPU test section if present
    marker = "## GPU Server Algorithm Test Results"
    if marker in content:
        idx = content.index(marker)
        # Find the --- before this section
        pre_idx = content.rfind("---", 0, idx)
        if pre_idx >= 0:
            content = content[:pre_idx].rstrip()

    # Append new section
    content = content.rstrip() + "\n" + section

    check_path.write_text(content, encoding="utf-8")
    n_pass = sum(1 for r in mod_data["solvers"].values() if r.get("status") == "completed")
    n_total = len(mod_data["solvers"])
    print(f"  {mod_id}: updated check.md ({n_pass}/{n_total} solvers passed)")
    updated += 1

print(f"\nUpdated {updated} check.md files")
