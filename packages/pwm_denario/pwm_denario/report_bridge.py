"""pwm_denario.report_bridge

Helper to embed PWM report artifacts into Denario's artifact system.
In this starter: just return paths and markdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def collect_report(runbundle_dir: str) -> Dict[str, Any]:
    rb = Path(runbundle_dir)
    report = rb / "artifacts" / "analysis" / "report.md"
    return {
        "runbundle_dir": str(rb),
        "report_md": report.read_text(encoding="utf-8") if report.exists() else "",
    }
