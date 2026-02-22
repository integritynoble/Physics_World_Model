"""pwm_core.analysis.report

Generates markdown reports from harness results and diagnosis.

Two entry points:
- make_report_md(bundle): legacy generic JSON-dump report
- make_triad_report_md(result): rich TriadReport from a HarnessResult
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pwm_core.targeting.harness import HarnessResult

# Gate labels for display
_GATE_LABELS = {
    "gate_1_sampling": "Gate 1 — Sampling / Measurement Budget",
    "gate_2_noise": "Gate 2 — Noise Model Accuracy",
    "gate_3_mismatch": "Gate 3 — Operator Mismatch",
}


def make_report_md(bundle: Dict[str, Any]) -> str:
    """Legacy generic report from an arbitrary results bundle."""
    ts = datetime.now(timezone.utc).isoformat()
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
            lines.append(
                f"- **{a.get('knob', '')}** {a.get('op', '')} "
                f"{a.get('val', '')}: {a.get('rationale', '')}"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def make_triad_report_md(result: "HarnessResult") -> str:
    """Generate a TriadReport in Markdown from a HarnessResult.

    Includes:
    - Evaluation metadata
    - 4-Scenario PSNR summary
    - Core scoring metrics (rho, oracle gap, RoIC, OFS)
    - Triad gate attribution with evidence and recommended action
    - Safety brake and gaming penalty summary

    Parameters
    ----------
    result : HarnessResult
        Output of ``Harness.run()``.

    Returns
    -------
    str
        Markdown-formatted TriadReport.
    """
    ts = datetime.now(timezone.utc).isoformat()
    agg = result.aggregate
    ga = result.gate_attribution

    lines: List[str] = []

    # --- Header ---
    lines += [
        f"# TriadReport: {result.modality}",
        f"",
        f"**Generated:** `{ts}`",
        f"**Modality:** `{result.modality}`  "
        f"**Solver:** `{result.solver}`  "
        f"**Track:** `{result.track}`",
        f"**Scenes:** {result.n_scenes}  "
        f"**Severity:** `{result.severity}`  "
        f"**Sandbox:** {result.sandbox}",
        f"",
    ]

    # --- Scoring Summary ---
    lines += [
        "## Scoring Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Recovery ratio (ρ) | `{agg.rho:.4f}` |",
        f"| Oracle gap | `{agg.oracle_gap:.2f}` dB |",
        f"| RoIC (dB/GPU-hr) | `{agg.roic:.2f}` |",
        f"| OFS | `{agg.ofs:.4f}` |",
        f"| Final score | `{agg.final_score:.4f}` |",
        f"| Total time | `{result.timing_s:.2f}` s |",
        "",
    ]

    # --- 4-Scenario PSNR ---
    if result.per_scene:
        first = result.per_scene[0]
        sr = first.scenario_results
        lines += [
            "## 4-Scenario PSNR (Scene 1)",
            "",
            "| Scenario | PSNR (dB) | SSIM | Runtime (s) |",
            "|----------|-----------|------|-------------|",
        ]
        for label, sid in [
            ("I — Ideal", "I"),
            ("II — Assumed (mismatch)", "II"),
            ("III — Calibrated", "III"),
            ("IV — Oracle", "IV"),
        ]:
            s = sr.get(sid)
            if s:
                lines.append(
                    f"| {label} | `{s.psnr:.2f}` | `{s.ssim:.4f}` | `{s.runtime_s:.3f}` |"
                )
        lines.append("")

    # --- Triad Gate Attribution ---
    lines += ["## Triad Gate Attribution", ""]

    if ga is not None:
        dom_label = _GATE_LABELS.get(ga.dominant_gate, ga.dominant_gate)
        lines += [
            f"**Dominant gate:** {dom_label}",
            f"",
            f"**Confidence:** `{ga.confidence:.3f}` "
            + (
                "(high — clear bottleneck)" if ga.confidence > 0.6
                else "(medium — moderate evidence)" if ga.confidence > 0.3
                else "(low — near-ideal performance or insufficient mismatch gap)"
            ),
            f"",
            "| Gate | Fraction | Label |",
            "|------|----------|-------|",
            f"| Gate 1 | `{ga.gate_1_sampling:.3f}` | Sampling / Measurement Budget |",
            f"| Gate 2 | `{ga.gate_2_noise:.3f}` | Noise Model Accuracy |",
            f"| Gate 3 | `{ga.gate_3_mismatch:.3f}` | Operator Mismatch |",
            "",
            "### Evidence",
            "",
        ]
        # Use first-scene evidence if aggregate evidence lacks PSNR detail
        ev = ga.evidence
        if "psnr_i" not in ev and result.per_scene:
            first_ga = result.per_scene[0].gate_attribution
            if first_ga is not None:
                ev = first_ga.evidence
        if "psnr_i" in ev:
            lines += [
                f"- PSNR_I (Ideal): `{ev['psnr_i']:.2f}` dB",
                f"- PSNR_II (Assumed): `{ev['psnr_ii']:.2f}` dB",
                f"- PSNR_III (Calibrated): `{ev['psnr_iii']:.2f}` dB",
                f"- PSNR_IV (Oracle): `{ev['psnr_iv']:.2f}` dB",
                f"- Total gap (I − II): `{ev.get('total_gap_db', 0):.2f}` dB",
                f"- Calibration gain (III − II): `{ev.get('calibration_gain_db', 0):.2f}` dB",
                f"- Oracle gap (I − III): `{ev.get('oracle_gap_db', 0):.2f}` dB",
                f"- Mismatch magnitude: `{ev.get('mismatch_magnitude', 0):.4f}`",
                "",
            ]

        lines += [
            "### Recommended Action",
            "",
            f"> {ga.recommended_action}",
            "",
        ]
    else:
        lines += [
            "_Gate attribution not available for this run._",
            "",
        ]

    # --- Safety Brakes ---
    if agg.safety_violations:
        lines += ["## Safety Brakes Triggered", ""]
        for v in agg.safety_violations:
            lines.append(
                f"- **{v.condition}** ({v.threshold}) → action: `{v.action}` "
                f"[actual: `{v.actual_value:.4f}`]"
            )
        lines.append("")

    if agg.disqualified:
        lines += [
            "## ⚠ Disqualified",
            "",
            f"> {agg.disqualification_reason}",
            "",
        ]

    # --- Gaming Penalties ---
    if agg.penalties:
        lines += ["## Gaming Penalties Applied", ""]
        for p in agg.penalties:
            lines.append(
                f"- **{p.check}**: −{p.penalty_fraction * 100:.0f}% — {p.reason}"
            )
        lines.append("")

    lines += ["---", f"*Generated by pwm_core `make_triad_report_md`*", ""]
    return "\n".join(lines)


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
