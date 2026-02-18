"""Validate pwm/reports/scoreboard.yaml contract.

If scoreboard.yaml does not exist the tests are skipped (not failed).
"""

import pytest
import yaml
from pathlib import Path

SCOREBOARD_PATH = (
    Path(__file__).resolve().parent.parent / "pwm" / "reports" / "scoreboard.yaml"
)
REPORT_DIR = SCOREBOARD_PATH.parent

VALID_STATUSES = {"DONE", "PARTIAL", "PENDING"}
EXPECTED_MODALITY_COUNT = 64


def _load_scoreboard():
    """Load and return the scoreboard dict, or None if file missing."""
    if not SCOREBOARD_PATH.exists():
        return None
    with open(SCOREBOARD_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Skipif decorator — skip entire module when scoreboard is absent
# ---------------------------------------------------------------------------

_scoreboard = _load_scoreboard()

pytestmark = pytest.mark.skipif(
    _scoreboard is None,
    reason="pwm/reports/scoreboard.yaml does not exist — skipping scoreboard tests",
)


class TestScoreboard:
    """Contract tests for scoreboard.yaml."""

    def _data(self):
        # Re-use the module-level load (cached at import time).
        assert _scoreboard is not None, "scoreboard.yaml not loaded"
        return _scoreboard

    # 1. All 64 modalities present ----------------------------------------

    def test_all_modalities_present(self):
        data = self._data()
        modalities = data.get("modalities", data)
        # If top-level is a dict of modalities directly
        if isinstance(modalities, dict) and "modalities" in modalities:
            modalities = modalities["modalities"]
        # Count entries — each key is a modality
        count = len(modalities)
        assert count == EXPECTED_MODALITY_COUNT, (
            f"Expected {EXPECTED_MODALITY_COUNT} modalities, found {count}"
        )

    # 2. Status values valid ----------------------------------------------

    def test_status_values(self):
        data = self._data()
        modalities = data.get("modalities", data)
        if isinstance(modalities, dict) and "modalities" in modalities:
            modalities = modalities["modalities"]

        for key, entry in modalities.items():
            if isinstance(entry, dict):
                status = entry.get("status", entry.get("Status"))
            elif isinstance(entry, str):
                status = entry
            else:
                pytest.fail(f"Unexpected entry type for '{key}': {type(entry)}")
                return
            assert status in VALID_STATUSES, (
                f"Modality '{key}' has invalid status '{status}'. "
                f"Must be one of {VALID_STATUSES}"
            )

    # 3. Counts add up to 64 ----------------------------------------------

    def test_counts_add_up(self):
        data = self._data()
        modalities = data.get("modalities", data)
        if isinstance(modalities, dict) and "modalities" in modalities:
            modalities = modalities["modalities"]

        counts = {"DONE": 0, "PARTIAL": 0, "PENDING": 0}
        for key, entry in modalities.items():
            if isinstance(entry, dict):
                status = entry.get("status", entry.get("Status", "PENDING"))
            elif isinstance(entry, str):
                status = entry
            else:
                status = "PENDING"
            if status in counts:
                counts[status] += 1

        total = counts["DONE"] + counts["PARTIAL"] + counts["PENDING"]
        assert total == EXPECTED_MODALITY_COUNT, (
            f"done({counts['DONE']}) + partial({counts['PARTIAL']}) + "
            f"pending({counts['PENDING']}) = {total}, expected {EXPECTED_MODALITY_COUNT}"
        )

    # 4. Every DONE modality has a report file ----------------------------

    def test_done_modalities_have_reports(self):
        data = self._data()
        modalities = data.get("modalities", data)
        if isinstance(modalities, dict) and "modalities" in modalities:
            modalities = modalities["modalities"]

        missing = []
        for key, entry in modalities.items():
            if isinstance(entry, dict):
                status = entry.get("status", entry.get("Status"))
            elif isinstance(entry, str):
                status = entry
            else:
                continue
            if status == "DONE":
                report_path = REPORT_DIR / f"{key}.md"
                if not report_path.exists():
                    missing.append(key)

        assert not missing, (
            f"DONE modalities missing report files: {missing}"
        )

    # 5. Metrics are numeric when present ---------------------------------

    def test_metrics_numeric(self):
        data = self._data()
        modalities = data.get("modalities", data)
        if isinstance(modalities, dict) and "modalities" in modalities:
            modalities = modalities["modalities"]

        METRIC_KEYS = {"psnr", "ssim", "nll", "score", "accuracy", "mae", "mse"}
        bad = []
        for key, entry in modalities.items():
            if not isinstance(entry, dict):
                continue
            for field, value in entry.items():
                if field.lower() in METRIC_KEYS and value is not None:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        bad.append(f"{key}.{field}={value!r}")

        assert not bad, (
            f"Non-numeric metric values found: {bad}"
        )
