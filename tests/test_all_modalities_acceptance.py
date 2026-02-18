"""Staged modality acceptance tests (Tier A/B/C).

Phase 5 acceptance gate: every modality template must compile, and
Tier A/B must produce finite forward output with appropriate shape.

Tests are parametrized from acceptance_thresholds.yaml and skip
gracefully if a template does not yet exist for a given modality.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import yaml

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.core.metric_registry import build_metric


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THRESHOLDS_PATH = os.path.join(
    _ROOT, "packages", "pwm_core", "contrib", "acceptance_thresholds.yaml"
)
_TEMPLATES_PATH = os.path.join(
    _ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml"
)


def _load_thresholds() -> Dict[str, Any]:
    with open(_THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


def _load_templates() -> Dict[str, Any]:
    with open(_TEMPLATES_PATH) as f:
        return yaml.safe_load(f)["templates"]


THRESHOLDS = _load_thresholds()
TEMPLATES = _load_templates()


# ---------------------------------------------------------------------------
# Modality-to-template key mapping
# ---------------------------------------------------------------------------
# Some acceptance threshold keys do not map 1:1 to template keys.
# This mapping resolves the discrepancies.

_MODALITY_TO_TEMPLATE: Dict[str, str] = {
    "widefield_low_dose": "widefield_lowdose_graph_v2",
    "confocal_live": "confocal_livecell_graph_v2",
    "confocal_3d": "confocal_3d_graph_v2",
    "light_sheet": "lightsheet_graph_v2",
    "panorama_multifocal": "panorama_graph_v2",
}


def _template_key(modality: str) -> str:
    """Resolve a modality name from the thresholds file to a v2 template key."""
    if modality in _MODALITY_TO_TEMPLATE:
        return _MODALITY_TO_TEMPLATE[modality]
    return f"{modality}_graph_v2"


def _has_template(modality: str) -> bool:
    """Check whether a v2 template exists for the given modality."""
    return _template_key(modality) in TEMPLATES


def _compile_template(modality: str):
    """Compile a v2 template for the given modality and return the GraphOperator."""
    key = _template_key(modality)
    tpl = dict(TEMPLATES[key])
    tpl.pop("description", None)
    spec = OperatorGraphSpec.model_validate({"graph_id": key, **tpl})
    compiler = GraphCompiler()
    return compiler.compile(spec), tpl


def _get_x_shape(tpl: Dict[str, Any]) -> Tuple[int, ...]:
    """Extract input shape from template metadata, defaulting to (32, 32)."""
    meta = tpl.get("metadata", {})
    raw = meta.get("x_shape", [32, 32])
    if isinstance(raw, list):
        return tuple(raw)
    return (32, 32)


# ---------------------------------------------------------------------------
# Tier lists
# ---------------------------------------------------------------------------

TIER_A = sorted([k for k, v in THRESHOLDS.items() if v.get("tier") == "A"])
TIER_B = sorted([k for k, v in THRESHOLDS.items() if v.get("tier") == "B"])
TIER_C = sorted([k for k, v in THRESHOLDS.items() if v.get("tier") == "C"])


# ---------------------------------------------------------------------------
# Tier A tests -- 8 must-pass modalities with full S1.1-S1.4
# ---------------------------------------------------------------------------


class TestTierA:
    """Tier A: 8 must-pass modalities with full template + forward checks."""

    @pytest.mark.parametrize("modality", TIER_A)
    def test_template_compiles(self, modality: str) -> None:
        """S0: Template exists and compiles without error."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, _tpl = _compile_template(modality)
        assert graph is not None
        assert graph.graph_id == _template_key(modality)

    @pytest.mark.parametrize("modality", TIER_A)
    def test_forward_sanity(self, modality: str) -> None:
        """S1.1: Forward produces finite, non-NaN output."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, tpl = _compile_template(modality)
        rng = np.random.RandomState(42)
        x_shape = _get_x_shape(tpl)
        x = rng.rand(*x_shape).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert np.isfinite(y).all(), f"{modality}: forward produced NaN/Inf"

    @pytest.mark.parametrize("modality", TIER_A)
    def test_forward_shape_nonzero(self, modality: str) -> None:
        """S1.2: Forward output is non-empty and not all zeros."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, tpl = _compile_template(modality)
        rng = np.random.RandomState(42)
        x_shape = _get_x_shape(tpl)
        x = rng.rand(*x_shape).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.size > 0, f"{modality}: forward output is empty"
        # For noise models that add random perturbation, output should not be flat zero
        assert np.any(y != 0), f"{modality}: forward output is all zeros"

    @pytest.mark.parametrize("modality", TIER_A)
    def test_threshold_entry_valid(self, modality: str) -> None:
        """S1.3: Threshold entry has required fields."""
        entry = THRESHOLDS[modality]
        assert entry["tier"] == "A"
        assert "primary_metric" in entry
        assert "quick_threshold" in entry
        assert "full_threshold" in entry
        # primary_metric must be a known metric
        metric = build_metric(entry["primary_metric"])
        assert metric is not None

    @pytest.mark.parametrize("modality", TIER_A)
    def test_deterministic_forward(self, modality: str) -> None:
        """S1.4: Forward is deterministic with fixed seed (same input -> same output)."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph1, tpl = _compile_template(modality)
        graph2, _ = _compile_template(modality)
        rng = np.random.RandomState(42)
        x_shape = _get_x_shape(tpl)
        x = rng.rand(*x_shape).astype(np.float64) * 0.5 + 0.1
        y1 = graph1.forward(x)
        y2 = graph2.forward(x)
        np.testing.assert_array_equal(
            y1, y2, err_msg=f"{modality}: forward not deterministic with same seed"
        )


# ---------------------------------------------------------------------------
# Tier B tests -- 4 modalities with S1.1-S1.2
# ---------------------------------------------------------------------------


class TestTierB:
    """Tier B: 4 modalities that must pass before release."""

    @pytest.mark.parametrize("modality", TIER_B)
    def test_template_compiles(self, modality: str) -> None:
        """S0: Template exists and compiles."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, _tpl = _compile_template(modality)
        assert graph is not None

    @pytest.mark.parametrize("modality", TIER_B)
    def test_forward_sanity(self, modality: str) -> None:
        """S1.1: Forward produces finite, non-NaN output."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, tpl = _compile_template(modality)
        rng = np.random.RandomState(42)
        x_shape = _get_x_shape(tpl)
        x = rng.rand(*x_shape).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert np.isfinite(y).all(), f"{modality}: forward produced NaN/Inf"

    @pytest.mark.parametrize("modality", TIER_B)
    def test_forward_shape_nonzero(self, modality: str) -> None:
        """S1.2: Forward output is non-empty."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, tpl = _compile_template(modality)
        rng = np.random.RandomState(42)
        x_shape = _get_x_shape(tpl)
        x = rng.rand(*x_shape).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.size > 0, f"{modality}: forward output is empty"

    @pytest.mark.parametrize("modality", TIER_B)
    def test_threshold_entry_valid(self, modality: str) -> None:
        """Threshold entry has required fields."""
        entry = THRESHOLDS[modality]
        assert entry["tier"] == "B"
        assert "primary_metric" in entry
        assert "quick_threshold" in entry


# ---------------------------------------------------------------------------
# Tier C tests -- 22 smoke test modalities
# ---------------------------------------------------------------------------


class TestTierC:
    """Tier C: 22 smoke-test modalities -- template exists, forward doesn't crash."""

    @pytest.mark.parametrize("modality", TIER_C)
    def test_template_exists(self, modality: str) -> None:
        """Tier C: v2 template exists in graph_templates.yaml."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality} (expected at {_template_key(modality)})")
        tpl = TEMPLATES[_template_key(modality)]
        assert tpl is not None

    @pytest.mark.parametrize("modality", TIER_C)
    def test_template_compiles(self, modality: str) -> None:
        """Tier C: Template compiles without error."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, _tpl = _compile_template(modality)
        assert graph is not None

    @pytest.mark.parametrize("modality", TIER_C)
    def test_forward_no_crash(self, modality: str) -> None:
        """Tier C: Forward pass completes without exception."""
        if not _has_template(modality):
            pytest.skip(f"No v2 template for {modality}")
        graph, tpl = _compile_template(modality)
        rng = np.random.RandomState(42)
        x_shape = _get_x_shape(tpl)
        x = rng.rand(*x_shape).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        # Minimal check: output exists and is an array
        assert isinstance(y, np.ndarray)
        assert y.size > 0


# ---------------------------------------------------------------------------
# Cross-tier structural checks
# ---------------------------------------------------------------------------


class TestAcceptanceStructure:
    """Structural consistency checks for the acceptance thresholds."""

    def test_all_tiers_present(self) -> None:
        """All three tiers are present in the thresholds."""
        tiers = {v.get("tier") for v in THRESHOLDS.values()}
        assert "A" in tiers
        assert "B" in tiers
        assert "C" in tiers

    def test_tier_a_count(self) -> None:
        """Tier A has exactly 8 modalities."""
        assert len(TIER_A) == 8

    def test_tier_b_count(self) -> None:
        """Tier B has exactly 4 modalities."""
        assert len(TIER_B) == 4

    def test_tier_c_count(self) -> None:
        """Tier C has at least 20 modalities."""
        assert len(TIER_C) >= 20

    def test_no_duplicate_modalities(self) -> None:
        """No modality appears in multiple tiers."""
        all_mods = TIER_A + TIER_B + TIER_C
        assert len(all_mods) == len(set(all_mods))

    def test_primary_metrics_known(self) -> None:
        """All primary_metric values are in the metric registry."""
        for mod, entry in THRESHOLDS.items():
            metric_name = entry["primary_metric"]
            metric = build_metric(metric_name)
            assert metric is not None, f"{mod}: unknown metric '{metric_name}'"

    def test_quick_threshold_leq_full(self) -> None:
        """quick_threshold <= full_threshold for all Tier A/B entries."""
        for mod, entry in THRESHOLDS.items():
            if entry.get("tier") in ("A", "B"):
                qt = entry.get("quick_threshold", 0)
                ft = entry.get("full_threshold", 0)
                assert qt <= ft, (
                    f"{mod}: quick_threshold ({qt}) > full_threshold ({ft})"
                )
