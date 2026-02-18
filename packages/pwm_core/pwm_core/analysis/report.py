"""pwm_core.analysis.report

Generates markdown report from results and diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime


def make_report_md(bundle: Dict[str, Any]) -> str:
    ts = datetime.utcnow().isoformat() + "Z"
    spec = bundle.get("spec", {})
    diag = bundle.get("diagnosis", {})
    metrics = bundle.get("metrics", {})
    actions = diag.get("suggested_actions", [])

    lines: List[str] = []
    lines.append(f"# PWM Report\n\nGenerated: `{ts}`\n")
    lines.append("## ExperimentSpec\n")
    lines.append("```json\n" + _json_dump(spec) + "\n```\n")

    lines.append("## Metrics\n")
    lines.append("```json\n" + _json_dump(metrics) + "\n```\n")

    lines.append("## Diagnosis\n")
    lines.append("```json\n" + _json_dump(diag) + "\n```\n")

    lines.append("## Suggested Actions\n")
    if isinstance(actions, list) and actions:
        for a in actions:
            lines.append(f"- **{a.get('knob','')}** {a.get('op','')} {a.get('val','')}: {a.get('rationale','')}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def _json_dump(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)
