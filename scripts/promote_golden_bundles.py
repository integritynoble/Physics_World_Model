#!/usr/bin/env python3
"""Promote golden reference bundles from Draft to Certified tier.

Reads certificate.json from each golden bundle, verifies all R-gates pass,
and re-issues with trust_tier=certified + reviewer signoff.
"""
import json
import sys
import datetime
from pathlib import Path

GOLDEN_DIR = Path("/tmp/golden_bundles")

MODALITIES = [
    "ct", "mri", "pet", "spect", "ultrasound", "oct",
    "mammography", "cbct", "fundus", "endoscopy", "fmri", "diffusion_mri",
]


def promote_bundle(modality: str) -> dict:
    """Promote a single bundle from draft to certified."""
    mod_dir = GOLDEN_DIR / modality
    if not mod_dir.exists():
        return {"modality": modality, "success": False, "error": "Directory not found"}

    # Find the bundle directory
    bundles = list(mod_dir.glob("run_*"))
    if not bundles:
        return {"modality": modality, "success": False, "error": "No bundle found"}

    bundle_dir = bundles[0]
    cert_path = bundle_dir / "certificate.json"
    if not cert_path.exists():
        return {"modality": modality, "success": False, "error": "No certificate.json"}

    cert = json.loads(cert_path.read_text())

    # Verify all R-gates pass
    gate_verdicts = cert.get("gate_verdicts", {})
    all_pass = all(
        (v.get("verdict") if isinstance(v, dict) else v) == "pass"
        for v in gate_verdicts.values()
    )

    if not all_pass:
        return {
            "modality": modality,
            "success": False,
            "error": f"Not all gates pass: {gate_verdicts}",
        }

    # Check no safety_brake
    risk_flags = cert.get("risk_flags", [])
    has_brake = any(
        (f if isinstance(f, str) else f.get("value", str(f))) == "safety_brake"
        for f in risk_flags
    )
    if has_brake:
        return {"modality": modality, "success": False, "error": "safety_brake is set"}

    # Promote to Certified
    cert["trust_tier"] = "certified"
    cert["promoted_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cert["promotion_reviewer"] = "pwm-golden-bundle-authority"
    cert["promotion_notes"] = (
        "Promoted from Draft to Certified: all R1-R4 gates pass, "
        "no safety_brake, golden reference bundle for Priority-1 modality."
    )

    # Write updated certificate
    cert_path.write_text(json.dumps(cert, indent=2, default=str))

    return {
        "modality": modality,
        "success": True,
        "bundle": str(bundle_dir),
        "trust_tier": "certified",
    }


def main():
    print("Golden Bundle Promotion: Draft -> Certified")
    print("=" * 50)

    results = []
    for mod in MODALITIES:
        info = promote_bundle(mod)
        status = "OK" if info["success"] else "FAIL"
        detail = info.get("error", info.get("trust_tier", ""))
        print(f"  {mod:20s} [{status}]  {detail}")
        results.append(info)

    success = sum(1 for r in results if r["success"])
    print(f"\nPromoted: {success}/{len(results)}")

    # Re-upload to GCS
    if success > 0 and "--upload" in sys.argv:
        import subprocess

        for r in results:
            if r["success"]:
                bundle = r["bundle"]
                cert = Path(bundle) / "certificate.json"
                dest = f"gs://pwm-benchmark-datasets/golden_bundles/{r['modality']}/"
                # Upload just the updated certificate
                subprocess.run(
                    [
                        "gsutil",
                        "cp",
                        str(cert),
                        f"{dest}{Path(bundle).name}/certificate.json",
                    ],
                    check=False,
                )
                print(f"  Uploaded: {r['modality']}")

    return 0 if success == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
