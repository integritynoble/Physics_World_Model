"""Unit tests for 4-layer threshold resolution.

The threshold resolver merges four configuration layers with increasing
precedence: standard_default < scanner_model < protocol < site_override.
Each test verifies a specific resolution or evaluation scenario.

The actual ThresholdResolver uses a YAML schema with ``metrics:`` as the
top-level key, and each metric has sub-keys for each layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pwm_core.clinical.common.threshold_resolver import (
    ResolvedThreshold,
    ThresholdResolver,
    ThresholdResult,
    ThresholdType,
)


# ===================================================================
# Fixture: threshold YAML with proper schema
# ===================================================================

_THRESHOLD_YAML_CONTENT = """\
metrics:
  ct_number_water:
    standard_default:
      threshold_type: ABSOLUTE_RANGE
      pass_range: [-7.0, 7.0]
      unit: HU
    protocol:
      "ACR CT Adult Abdomen 120kVp":
        threshold_type: RELATIVE_TO_BASELINE_SIGMA
        tolerance_from_baseline_sigma: 2.0
        unit: HU

  ct_number_bone:
    standard_default:
      threshold_type: ABSOLUTE_RANGE
      pass_range: [850, 970]
      unit: HU

  ct_number_air:
    standard_default:
      threshold_type: ABSOLUTE_RANGE
      pass_range: [-1005, -970]
      unit: HU

  noise_std:
    standard_default:
      threshold_type: ABSOLUTE_RANGE
      pass_range: [0, 12.0]
      unit: HU
    scanner_model:
      "Siemens SOMATOM Force":
        pass_range: [0, 9.0]
    site_override:
      pass_range: [0, 8.0]

  uniformity:
    standard_default:
      threshold_type: ABSOLUTE_RANGE
      pass_range: [0, 5.0]
      unit: HU
    scanner_model:
      "Siemens SOMATOM Force":
        pass_range: [0, 3.5]

  geometric_accuracy:
    standard_default:
      threshold_type: ABSOLUTE_RANGE
      pass_range: [0, 2.0]
      unit: mm
"""


@pytest.fixture()
def threshold_yaml(tmp_path: Path) -> Path:
    """Create a temporary threshold YAML file with proper schema."""
    path = tmp_path / "thresholds.yaml"
    path.write_text(_THRESHOLD_YAML_CONTENT, encoding="utf-8")
    return path


# ===================================================================
# Layer resolution tests
# ===================================================================


class TestStandardDefaultOnly:
    """Resolve with no scanner or protocol overrides."""

    def test_standard_default_only(self, threshold_yaml: Path) -> None:
        """With only the standard layer, the default threshold should apply.

        For ``ct_number_water``, the standard range is ``[-7.0, 7.0]``.
        """
        resolver = ThresholdResolver(threshold_yaml)
        threshold = resolver.resolve(metric_name="ct_number_water")

        assert threshold.pass_range == (-7.0, 7.0)
        assert threshold.resolved_layer == "standard_default"
        assert threshold.unit == "HU"


class TestScannerModelOverride:
    """Scanner-model layer should override standard for matching metrics."""

    def test_scanner_model_override(self, threshold_yaml: Path) -> None:
        """Siemens SOMATOM Force has a tighter noise_std limit of 9.0 HU.

        The standard layer allows up to 12.0 HU, but the scanner-model
        layer tightens this to 9.0 HU.
        """
        resolver = ThresholdResolver(threshold_yaml)
        threshold = resolver.resolve(
            metric_name="noise_std",
            scanner_model="Siemens SOMATOM Force",
        )
        # Site override is always applied if present in the YAML,
        # so the effective range is [0, 8.0] from site_override.
        # But if we want to test scanner_model alone, we need a metric
        # without a site_override.
        threshold_unif = resolver.resolve(
            metric_name="uniformity",
            scanner_model="Siemens SOMATOM Force",
        )
        assert threshold_unif.pass_range == (0, 3.5)
        assert threshold_unif.resolved_layer == "scanner_model"


class TestProtocolOverride:
    """Protocol layer should override standard for matching metrics."""

    def test_protocol_override(self, threshold_yaml: Path) -> None:
        """The ACR CT Adult Abdomen protocol specifies a sigma-based
        threshold for ct_number_water.
        """
        resolver = ThresholdResolver(threshold_yaml)
        threshold = resolver.resolve(
            metric_name="ct_number_water",
            protocol="ACR CT Adult Abdomen 120kVp",
        )
        assert threshold.threshold_type == ThresholdType.RELATIVE_TO_BASELINE_SIGMA
        assert threshold.tolerance_sigma == 2.0
        assert threshold.resolved_layer == "protocol"


class TestSiteOverride:
    """Site-override layer should be the tightest override."""

    def test_site_override(self, threshold_yaml: Path) -> None:
        """noise_std has a site_override with pass_range [0, 8.0].

        This applies unconditionally (the site_override layer in the
        YAML is always merged if present).
        """
        resolver = ThresholdResolver(threshold_yaml)
        threshold = resolver.resolve(metric_name="noise_std")
        assert threshold.pass_range == (0, 8.0)
        assert threshold.resolved_layer == "site_override"


class TestLayerPrecedence:
    """Site > protocol > scanner > standard when all layers apply."""

    def test_layer_precedence(self, threshold_yaml: Path) -> None:
        """For noise_std, even with scanner_model specified, the site_override
        (tightest) should win because it comes last in the layer chain.

        Standard: [0, 12.0]
        Scanner:  [0, 9.0]
        Site:     [0, 8.0]
        """
        resolver = ThresholdResolver(threshold_yaml)
        threshold = resolver.resolve(
            metric_name="noise_std",
            scanner_model="Siemens SOMATOM Force",
        )
        assert threshold.pass_range == (0, 8.0)
        assert threshold.resolved_layer == "site_override"


# ===================================================================
# Evaluation tests
# ===================================================================


class TestEvaluatePass:
    """Value well within the resolved range should yield PASS."""

    def test_evaluate_pass(self, threshold_yaml: Path) -> None:
        """A water CT number of 0.5 HU is well within [-7.0, 7.0]."""
        resolver = ThresholdResolver(threshold_yaml)
        result = resolver.evaluate(
            metric_name="ct_number_water",
            value=0.5,
        )
        assert result.status == "PASS"


class TestEvaluateWarning:
    """Value near the boundary should yield WARNING."""

    def test_evaluate_warning(self, threshold_yaml: Path) -> None:
        """A water CT number of 6.5 HU is near the upper limit of [-7.0, 7.0].

        With a 10% warning margin on range width 14 HU, the warning zone
        starts at 7.0 - 1.4 = 5.6 HU from the edge.
        """
        resolver = ThresholdResolver(threshold_yaml)
        result = resolver.evaluate(
            metric_name="ct_number_water",
            value=6.5,
        )
        assert result.status == "WARNING", (
            f"Expected WARNING near boundary, got {result.status}"
        )


class TestEvaluateFail:
    """Value outside the range should yield FAIL."""

    def test_evaluate_fail(self, threshold_yaml: Path) -> None:
        """A water CT number of 10.0 HU exceeds the [-7.0, 7.0] range."""
        resolver = ThresholdResolver(threshold_yaml)
        result = resolver.evaluate(
            metric_name="ct_number_water",
            value=10.0,
        )
        assert result.status == "FAIL"


class TestRelativeThreshold:
    """Baseline-relative sigma threshold evaluation."""

    def test_relative_threshold(self, threshold_yaml: Path) -> None:
        """The ACR CT Adult Abdomen protocol defines ct_number_water with
        tolerance_from_baseline_sigma = 2.0.

        If baseline_value is 0.5, the tolerance band is [0.5 - 2.0, 0.5 + 2.0].
        A value of 0.8 should PASS (deviation = 0.3 < 2.0).
        """
        resolver = ThresholdResolver(threshold_yaml)
        result = resolver.evaluate(
            metric_name="ct_number_water",
            value=0.8,
            baseline_value=0.5,
            protocol="ACR CT Adult Abdomen 120kVp",
        )
        assert result.status == "PASS", f"Expected PASS, got {result.status}"


class TestResolveAll:
    """resolve_all should handle multiple metrics at once."""

    def test_resolve_all(self, threshold_yaml: Path) -> None:
        """Resolve thresholds for all standard-layer metrics in one call."""
        resolver = ThresholdResolver(threshold_yaml)

        metrics_to_resolve = [
            "ct_number_water",
            "ct_number_bone",
            "noise_std",
            "uniformity",
            "geometric_accuracy",
        ]
        results = resolver.resolve_all(metric_names=metrics_to_resolve)

        assert set(results.keys()) == set(metrics_to_resolve)
        for name, thresh in results.items():
            assert isinstance(thresh, ResolvedThreshold), (
                f"Expected ResolvedThreshold for '{name}', got {type(thresh)}"
            )
