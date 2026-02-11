"""Tests for pwm_core.api.prompt_parser (R9: typed ExperimentSpec output).

Verifies that parse_prompt() returns a validated ExperimentSpec and that
parse_prompt_raw() provides backward-compatible _ParsedFields access.
"""

from __future__ import annotations

import re

import pytest

from pwm_core.api.prompt_parser import (
    ParsedPrompt,
    _ParsedFields,
    parse_prompt,
    parse_prompt_raw,
)
from pwm_core.api.types import (
    ExperimentSpec,
    InputMode,
    PhysicsState,
    TaskKind,
)


# ---------------------------------------------------------------------------
# 1. parse_prompt returns ExperimentSpec
# ---------------------------------------------------------------------------


class TestParsePromptReturnsSpec:
    """Core tests for the simulate-widefield-low-dose prompt."""

    @pytest.fixture()
    def spec(self) -> ExperimentSpec:
        return parse_prompt("simulate widefield low-dose")

    def test_returns_experiment_spec(self, spec: ExperimentSpec) -> None:
        """parse_prompt('simulate widefield low-dose') returns ExperimentSpec."""
        assert isinstance(spec, ExperimentSpec)

    def test_version(self, spec: ExperimentSpec) -> None:
        """spec.version == '0.2.1'."""
        assert spec.version == "0.2.1"

    def test_modality(self, spec: ExperimentSpec) -> None:
        """spec.states.physics.modality == 'widefield'."""
        assert spec.states.physics.modality == "widefield"

    def test_task_kind_simulate(self, spec: ExperimentSpec) -> None:
        """simulate mode -> TaskKind.simulate_recon_analyze."""
        assert spec.states.task.kind == TaskKind.simulate_recon_analyze

    def test_photon_budget(self, spec: ExperimentSpec) -> None:
        """low-dose -> photon_budget max_photons == 1000.0."""
        assert spec.states.budget is not None
        assert spec.states.budget.photon_budget == {"max_photons": 1000.0}

    def test_input_mode_simulate(self, spec: ExperimentSpec) -> None:
        """simulate mode -> InputMode.simulate."""
        assert spec.input.mode == InputMode.simulate

    def test_graph_template_id(self, spec: ExperimentSpec) -> None:
        """graph_template_id == 'widefield_graph_v2'."""
        assert spec.states.physics.graph_template_id == "widefield_graph_v2"

    def test_operator_id(self, spec: ExperimentSpec) -> None:
        """operator.parametric.operator_id == 'widefield_graph_v2'."""
        assert spec.input.operator is not None
        assert spec.input.operator.parametric is not None
        assert spec.input.operator.parametric.operator_id == "widefield_graph_v2"


# ---------------------------------------------------------------------------
# 2. CASSI + FISTA reconstruct
# ---------------------------------------------------------------------------


class TestCassiFista:
    """Tests for 'reconstruct CASSI using FISTA'."""

    @pytest.fixture()
    def spec(self) -> ExperimentSpec:
        return parse_prompt("reconstruct CASSI using FISTA")

    def test_modality_cassi(self, spec: ExperimentSpec) -> None:
        assert spec.states.physics.modality == "cassi"

    def test_task_kind_reconstruct(self, spec: ExperimentSpec) -> None:
        assert spec.states.task.kind == TaskKind.reconstruct_only

    def test_solver_fista(self, spec: ExperimentSpec) -> None:
        solver_ids = [s.id for s in spec.recon.portfolio.solvers]
        assert "fista" in solver_ids


# ---------------------------------------------------------------------------
# 3. Calibrate MRI
# ---------------------------------------------------------------------------


def test_calibrate_mri() -> None:
    """'calibrate MRI' -> calibrate_and_reconstruct."""
    spec = parse_prompt("calibrate MRI")
    assert spec.states.task.kind == TaskKind.calibrate_and_reconstruct


# ---------------------------------------------------------------------------
# 4. Round-trip serialization
# ---------------------------------------------------------------------------


def test_round_trip() -> None:
    """model_dump -> model_validate round-trip succeeds."""
    spec = parse_prompt("simulate widefield low-dose")
    data = spec.model_dump()
    restored = ExperimentSpec.model_validate(data)
    assert restored.version == spec.version
    assert restored.states.physics.modality == spec.states.physics.modality
    assert restored.id == spec.id


# ---------------------------------------------------------------------------
# 5. Backward compat: parse_prompt_raw
# ---------------------------------------------------------------------------


def test_parse_prompt_raw_returns_parsed_fields() -> None:
    """parse_prompt_raw returns _ParsedFields (backward compat)."""
    fields = parse_prompt_raw("simulate widefield")
    assert isinstance(fields, _ParsedFields)
    assert fields.modality == "widefield"


def test_parsed_prompt_alias() -> None:
    """ParsedPrompt alias still works."""
    assert ParsedPrompt is _ParsedFields


# ---------------------------------------------------------------------------
# 6. Unknown modality -> "unknown"
# ---------------------------------------------------------------------------


def test_unknown_modality() -> None:
    """Unknown modality prompt -> modality='unknown', no error."""
    spec = parse_prompt("reconstruct something")
    assert spec.states.physics.modality == "unknown"
    assert spec.states.physics.graph_template_id is None
    assert spec.input.operator is None


# ---------------------------------------------------------------------------
# 7. spec.id format
# ---------------------------------------------------------------------------


def test_spec_id_format() -> None:
    """spec.id starts with 'prompt_' followed by 8 hex chars."""
    spec = parse_prompt("simulate widefield")
    assert spec.id.startswith("prompt_")
    hex_part = spec.id[len("prompt_"):]
    assert len(hex_part) == 8
    assert re.fullmatch(r"[0-9a-f]{8}", hex_part) is not None


# ---------------------------------------------------------------------------
# 8. PhysicsState has graph_template_id field
# ---------------------------------------------------------------------------


def test_physics_state_has_graph_template_id() -> None:
    """PhysicsState model fields include graph_template_id."""
    assert "graph_template_id" in PhysicsState.model_fields


# ---------------------------------------------------------------------------
# 9. Input mode for non-simulate
# ---------------------------------------------------------------------------


def test_input_mode_measured_for_reconstruct() -> None:
    """Non-simulate mode -> InputMode.measured."""
    spec = parse_prompt("reconstruct CASSI")
    assert spec.input.mode == InputMode.measured


def test_input_mode_measured_for_calibrate() -> None:
    """Calibrate mode -> InputMode.measured."""
    spec = parse_prompt("calibrate MRI")
    assert spec.input.mode == InputMode.measured
