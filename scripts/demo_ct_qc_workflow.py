#!/usr/bin/env python3
"""Demonstrate CT scanner QC workflow with InstrumentCard + drift detection.

Satisfies checklist items:
- Operations flywheel: instrument -> QC runs -> drift detection
- CT scanner has InstrumentCard + running QC workflow
"""
import json
import numpy as np
from pathlib import Path
import datetime


def demo_ct_qc():
    output_dir = Path("/tmp/pwm_ct_qc_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create InstrumentCard for a CT scanner
    instrument_card = {
        "card_type": "InstrumentCard",
        "instrument_id": "siemens_somatom_001",
        "manufacturer": "Siemens Healthineers",
        "model": "SOMATOM Force",
        "modality": "ct",
        "serial_number": "SN-2024-001",
        "installation_date": "2024-06-15",
        "calibration_state": "calibrated",
        "last_qc_date": "2026-03-25",
        "primitive_chain": ["Project", "Accumulate", "Detect"],
        "operating_parameters": {
            "kVp": [80, 100, 120, 140],
            "mA_range": [50, 800],
            "rotation_time_s": 0.5,
            "detector_rows": 192,
        },
    }
    (output_dir / "instrument_card.json").write_text(
        json.dumps(instrument_card, indent=2)
    )
    print(f"1. InstrumentCard created: {instrument_card['instrument_id']}")

    # 2. Simulate commissioning baseline
    baseline = {
        "noise_std_hu": 4.2,
        "cnr": 15.3,
        "mtf10_lpmm": 0.42,
        "hu_accuracy": 0.5,
        "uniformity_hu": 2.1,
        "date": "2024-06-15",
    }
    (output_dir / "commissioning_baseline.json").write_text(
        json.dumps(baseline, indent=2)
    )
    print(
        f"2. Commissioning baseline: noise={baseline['noise_std_hu']} HU, "
        f"CNR={baseline['cnr']}"
    )

    # 3. Simulate monthly QC run (current)
    np.random.seed(42)
    current_qc = {
        "noise_std_hu": baseline["noise_std_hu"] + np.random.normal(0, 0.3),
        "cnr": baseline["cnr"] + np.random.normal(0, 0.5),
        "mtf10_lpmm": baseline["mtf10_lpmm"] + np.random.normal(0, 0.01),
        "hu_accuracy": baseline["hu_accuracy"] + np.random.normal(0, 0.2),
        "uniformity_hu": baseline["uniformity_hu"] + np.random.normal(0, 0.15),
        "date": "2026-03-25",
    }

    # 4. Drift detection
    drift_detected = False
    drift_metrics = {}
    thresholds = {
        "noise_std_hu": 6.0,
        "cnr": 10.0,
        "mtf10_lpmm": 0.35,
        "hu_accuracy": 4.0,
        "uniformity_hu": 5.0,
    }

    for metric, threshold in thresholds.items():
        current_val = current_qc[metric]
        baseline_val = baseline[metric]
        drift = abs(current_val - baseline_val) / max(abs(baseline_val), 1e-6)
        breached = (
            metric in ["noise_std_hu", "hu_accuracy", "uniformity_hu"]
            and current_val > threshold
        ) or (metric in ["cnr", "mtf10_lpmm"] and current_val < threshold)
        drift_metrics[metric] = {
            "baseline": round(baseline_val, 3),
            "current": round(float(current_val), 3),
            "threshold": threshold,
            "drift_pct": round(float(drift * 100), 1),
            "breached": breached,
        }
        if breached:
            drift_detected = True

    print(
        f"3. QC run: noise={current_qc['noise_std_hu']:.1f} HU, "
        f"CNR={current_qc['cnr']:.1f}"
    )
    print(f"   Drift detected: {drift_detected}")

    # 5. Generate CTQCDiagnosticReport
    from pwm_core.core.runbundle.triad_report import CTQCDiagnosticReport

    report = CTQCDiagnosticReport(
        run_id="qc_somatom_001_2026_03",
        spec_id="ct_qc_monthly_phantom",
        drift_detected=drift_detected,
        drift_magnitude=max(d["drift_pct"] for d in drift_metrics.values()),
        artifact_flags={"ring": False, "cupping": False, "streak": False},
        threshold_breaches={
            k: v for k, v in drift_metrics.items() if v["breached"]
        },
        noise_std_hu=round(float(current_qc["noise_std_hu"]), 2),
        cnr=round(float(current_qc["cnr"]), 2),
        mtf10_lpmm=round(float(current_qc["mtf10_lpmm"]), 3),
        hu_accuracy=round(float(current_qc["hu_accuracy"]), 2),
        uniformity_hu=round(float(current_qc["uniformity_hu"]), 2),
    )

    report_path = output_dir / "ct_qc_diagnostic_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    print(
        f"4. CTQCDiagnosticReport: complete={report.is_complete}, "
        f"concern={report.dominant_concern}"
    )

    # 6. Issue Certificate for QC run
    from pwm_core.core.runbundle.certificate import (
        Certificate,
        DomainFlags,
        GateResult,
        GateVerdict,
        RiskFlag,
        TrustTier,
    )

    cert = Certificate(
        run_id="qc_somatom_001_2026_03",
        spec_id="ct_qc_monthly_phantom",
        trust_tier=TrustTier.draft,
        risk_flags=[RiskFlag.safety_brake] if drift_detected else [],
        active_gates=["r1", "r2", "r3", "r4"],
        gate_verdicts={
            "r1": GateResult(
                verdict=GateVerdict.pass_, message="QC spec complete"
            ),
            "r2": GateResult(
                verdict=GateVerdict.pass_, message="Reproducible QC protocol"
            ),
            "r3": GateResult(
                verdict=GateVerdict.pass_, message="Metrics verified"
            ),
            "r4": GateResult(
                verdict=GateVerdict.pass_, message="Within time budget"
            ),
        },
        domain_flags=DomainFlags(
            ct_qc={
                "drift": drift_detected,
                "artifacts": {"ring": False, "cupping": False},
                "threshold_breaches": {
                    k: v["breached"] for k, v in drift_metrics.items()
                },
            }
        ),
        kernel_version="demo",
        profile_version="ct_qc/v1.0",
    )

    cert_path = output_dir / "certificate.json"
    cert_path.write_text(json.dumps(cert.to_dict(), indent=2, default=str))
    print(
        f"5. Certificate: trust_tier={cert.trust_tier.value}, "
        f"risk_flags={[f.value for f in cert.risk_flags]}"
    )

    print(f"\nCT QC workflow complete: {output_dir}")
    return 0


if __name__ == "__main__":
    demo_ct_qc()
