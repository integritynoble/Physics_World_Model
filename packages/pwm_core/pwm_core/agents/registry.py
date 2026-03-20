"""
pwm_core.agents.registry
~~~~~~~~~~~~~~~~~~~~~~~~~

RegistryBuilder: loads all YAML registries from ``contrib/``, validates each
through its Pydantic schema, and provides assertion helpers for LLM output
validation.

The builder is the single source of truth for every modality key, mismatch
family, solver identifier, and calibration table used by the agent system.
Invalid or hallucinated keys are caught immediately via ``assert_*`` helpers
before any heavy computation begins.

Usage::

    from pwm_core.agents.registry import RegistryBuilder

    registry = RegistryBuilder()              # uses default contrib/ path
    registry.assert_modality_exists("cassi")  # passes
    registry.assert_modality_exists("bogus")  # raises RegistryKeyError
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .contracts import StrictBaseModel
from .registry_schemas import (
    CalibrationEntry,
    CompressionDbFileYaml,
    MetricsDbFileYaml,
    MismatchDbFileYaml,
    MismatchModalityYaml,
    ModalitiesFileYaml,
    ModalityYaml,
    PhotonDbFileYaml,
    PhotonModelYaml,
)

logger = logging.getLogger(__name__)

# Default contrib/ directory relative to *this* file.
# Layout: packages/pwm_core/pwm_core/agents/registry.py
#          -> ../../contrib/
_DEFAULT_CONTRIB_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "contrib"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegistryKeyError(Exception):
    """Raised when a requested key does not exist in a registry."""


# ---------------------------------------------------------------------------
# RegistryBuilder
# ---------------------------------------------------------------------------


class RegistryBuilder:
    """Load, validate, and query all YAML registries under ``contrib/``.

    Parameters
    ----------
    contrib_dir : str, optional
        Filesystem path to the ``contrib/`` directory containing YAML
        registry files.  Defaults to the ``contrib/`` directory relative
        to this package (i.e. ``packages/pwm_core/contrib/``).
    """

    # Canonical filenames ------------------------------------------------
    _MODALITIES_FILE = "modalities.yaml"
    _MISMATCH_FILE = "mismatch_db.yaml"
    _COMPRESSION_FILE = "compression_db.yaml"
    _PHOTON_FILE = "photon_db.yaml"
    _METRICS_FILE = "metrics_db.yaml"
    _SOLVER_FILE = "solver_registry.yaml"

    def __init__(self, contrib_dir: str = _DEFAULT_CONTRIB_DIR) -> None:
        self._contrib_dir = contrib_dir
        logger.info("RegistryBuilder loading from %s", contrib_dir)

        # -- Load and validate each YAML file ----------------------------
        self._modalities: Optional[ModalitiesFileYaml] = self._load_validated(
            self._MODALITIES_FILE, ModalitiesFileYaml
        )
        self._mismatch: Optional[MismatchDbFileYaml] = self._load_validated(
            self._MISMATCH_FILE, MismatchDbFileYaml
        )
        self._compression: Optional[CompressionDbFileYaml] = self._load_validated(
            self._COMPRESSION_FILE, CompressionDbFileYaml
        )
        self._photon: Optional[PhotonDbFileYaml] = self._load_validated(
            self._PHOTON_FILE, PhotonDbFileYaml
        )
        self._metrics: Optional[MetricsDbFileYaml] = self._load_validated(
            self._METRICS_FILE, MetricsDbFileYaml
        )

        # Solver registry is a flat dict (not a Pydantic schema) so load
        # it separately.
        self._solver_raw: Optional[Dict[str, Any]] = self._load_raw_optional(
            self._SOLVER_FILE
        )

        # -- Cross-reference validation ----------------------------------
        self._validate_cross_references()

    # ===================================================================
    # Internal loading helpers
    # ===================================================================

    def _load_raw(self, filename: str) -> dict:
        """Read and parse a YAML file from the contrib directory.

        Parameters
        ----------
        filename : str
            Basename of the YAML file (e.g. ``"modalities.yaml"``).

        Returns
        -------
        dict
            Parsed YAML content.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        filepath = os.path.join(self._contrib_dir, filename)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data is None:
            return {}
        return data

    def _load_raw_optional(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a YAML file, returning ``None`` if it does not exist."""
        try:
            return self._load_raw(filename)
        except FileNotFoundError:
            logger.warning(
                "Registry file %s not found in %s -- continuing with "
                "empty data for this registry.",
                filename,
                self._contrib_dir,
            )
            return None

    def _load_validated(self, filename: str, schema_cls: type):
        """Load a YAML file and validate it through a Pydantic schema.

        Parameters
        ----------
        filename : str
            Basename of the YAML file.
        schema_cls : type
            Pydantic model class to validate against.

        Returns
        -------
        StrictBaseModel or None
            Validated model instance, or ``None`` if the file does not exist.
        """
        raw = self._load_raw_optional(filename)
        if raw is None:
            return None
        try:
            return schema_cls.model_validate(raw)
        except Exception:
            logger.error(
                "Validation failed for %s against %s",
                filename,
                schema_cls.__name__,
                exc_info=True,
            )
            raise

    # ===================================================================
    # Cross-reference validation
    # ===================================================================

    def _validate_cross_references(self) -> None:
        """Assert that every modality key used in sub-registries exists in
        ``modalities.yaml``.

        If ``modalities.yaml`` was not loaded (file missing), this check
        is skipped with a warning.
        """
        if self._modalities is None:
            logger.warning(
                "Skipping cross-reference validation: modalities.yaml "
                "not loaded."
            )
            return

        valid_keys = set(self._modalities.modalities.keys())

        # -- mismatch_db.yaml -------------------------------------------
        if self._mismatch is not None:
            for key in self._mismatch.modalities:
                if key not in valid_keys:
                    logger.warning(
                        "mismatch_db.yaml references unknown modality '%s'. "
                        "Valid keys: %s",
                        key,
                        sorted(valid_keys),
                    )

        # -- compression_db.yaml ----------------------------------------
        if self._compression is not None:
            for key in self._compression.calibration_tables:
                if key not in valid_keys:
                    logger.warning(
                        "compression_db.yaml references unknown modality '%s'. "
                        "Valid keys: %s",
                        key,
                        sorted(valid_keys),
                    )

        # -- photon_db.yaml ---------------------------------------------
        if self._photon is not None:
            for key in self._photon.modalities:
                if key not in valid_keys:
                    logger.warning(
                        "photon_db.yaml references unknown modality '%s'. "
                        "Valid keys: %s",
                        key,
                        sorted(valid_keys),
                    )

        # -- metrics_db.yaml --------------------------------------------
        if self._metrics is not None:
            for metric_set in self._metrics.metric_sets.values():
                for key in metric_set.modalities:
                    if key not in valid_keys:
                        logger.warning(
                            "metrics_db.yaml references unknown modality "
                            "'%s'. Valid keys: %s",
                            key,
                            sorted(valid_keys),
                        )

    # ===================================================================
    # Assertion helpers (for LLM output validation)
    # ===================================================================

    def assert_modality_exists(self, key: str) -> None:
        """Raise :class:`RegistryKeyError` if *key* is not a valid modality.

        Parameters
        ----------
        key : str
            Modality key to validate (e.g. ``"cassi"``).

        Raises
        ------
        RegistryKeyError
            With a message listing all valid modality keys.
        """
        valid = self.list_modalities()
        if key not in valid:
            raise RegistryKeyError(
                f"Unknown modality key '{key}'. "
                f"Valid keys are: {sorted(valid)}"
            )

    def assert_mismatch_family_exists(
        self, modality_key: str, family_id: str
    ) -> None:
        """Raise :class:`RegistryKeyError` if *family_id* is not a valid
        mismatch parameter for *modality_key*.

        Parameters
        ----------
        modality_key : str
            Modality key (validated first).
        family_id : str
            Mismatch parameter / family identifier to check.

        Raises
        ------
        RegistryKeyError
            If the modality or family does not exist.
        """
        self.assert_modality_exists(modality_key)

        if self._mismatch is None:
            raise RegistryKeyError(
                f"Mismatch database not loaded; cannot validate "
                f"family '{family_id}' for modality '{modality_key}'."
            )

        if modality_key not in self._mismatch.modalities:
            raise RegistryKeyError(
                f"No mismatch entry for modality '{modality_key}'. "
                f"Modalities with mismatch data: "
                f"{sorted(self._mismatch.modalities.keys())}"
            )

        valid_families = list(
            self._mismatch.modalities[modality_key].parameters.keys()
        )
        if family_id not in valid_families:
            raise RegistryKeyError(
                f"Unknown mismatch family '{family_id}' for modality "
                f"'{modality_key}'. Valid families: {sorted(valid_families)}"
            )

    def assert_solver_exists(self, solver_id: str) -> None:
        """Raise :class:`RegistryKeyError` if *solver_id* is not found in any
        solver registry entry.

        The solver registry maps modality keys to tier dicts, each containing
        a ``name`` field.  This method searches across all tiers for a matching
        solver ``name`` or module-level key.

        Parameters
        ----------
        solver_id : str
            Solver identifier to check (name or module path fragment).

        Raises
        ------
        RegistryKeyError
            If no matching solver is found.
        """
        if self._solver_raw is None:
            raise RegistryKeyError(
                f"Solver registry not loaded; cannot validate "
                f"solver '{solver_id}'."
            )

        # Collect all known solver names and module paths.
        known_solvers: set[str] = set()
        for _modality_key, tiers in self._solver_raw.items():
            if not isinstance(tiers, dict):
                continue
            for _tier_name, tier_info in tiers.items():
                if not isinstance(tier_info, dict):
                    continue
                if "name" in tier_info:
                    known_solvers.add(tier_info["name"])
                if "module" in tier_info:
                    known_solvers.add(tier_info["module"])
                if "function" in tier_info:
                    known_solvers.add(tier_info["function"])

        if solver_id not in known_solvers:
            raise RegistryKeyError(
                f"Unknown solver '{solver_id}'. "
                f"Known solvers: {sorted(known_solvers)}"
            )

    def assert_signal_prior_exists(self, prior_class: str) -> None:
        """Raise :class:`RegistryKeyError` if *prior_class* is not a valid
        signal prior.

        Checks against both the ``SignalPriorClass`` enum values in
        ``contracts.py`` and any ``signal_prior_class`` fields found in
        ``compression_db.yaml``.

        Parameters
        ----------
        prior_class : str
            Signal prior class identifier to validate.

        Raises
        ------
        RegistryKeyError
            If the prior class is not recognised.
        """
        # Collect valid signal prior classes from contracts enum.
        from .contracts import SignalPriorClass

        valid_priors: set[str] = {member.value for member in SignalPriorClass}

        # Also collect from compression_db calibration tables.
        if self._compression is not None:
            for cal_table in self._compression.calibration_tables.values():
                valid_priors.add(cal_table.signal_prior_class)

        if prior_class not in valid_priors:
            raise RegistryKeyError(
                f"Unknown signal prior class '{prior_class}'. "
                f"Valid classes: {sorted(valid_priors)}"
            )

    # ===================================================================
    # Query helpers
    # ===================================================================

    def all_keys(self) -> Dict[str, List[str]]:
        """Return all valid keys across registries, for LLM constraint.

        Returns
        -------
        dict[str, list[str]]
            Mapping from registry name to the list of valid keys within it.
            Keys included: ``"modalities"``, ``"mismatch_modalities"``,
            ``"compression_modalities"``, ``"photon_modalities"``,
            ``"metric_sets"``, ``"solvers"``.
        """
        result: Dict[str, List[str]] = {}

        # Modalities
        result["modalities"] = self.list_modalities()

        # Mismatch modalities
        if self._mismatch is not None:
            result["mismatch_modalities"] = sorted(
                self._mismatch.modalities.keys()
            )
        else:
            result["mismatch_modalities"] = []

        # Compression / calibration modalities
        if self._compression is not None:
            result["compression_modalities"] = sorted(
                self._compression.calibration_tables.keys()
            )
        else:
            result["compression_modalities"] = []

        # Photon modalities
        if self._photon is not None:
            result["photon_modalities"] = sorted(
                self._photon.modalities.keys()
            )
        else:
            result["photon_modalities"] = []

        # Metric set names
        if self._metrics is not None:
            result["metric_sets"] = sorted(
                self._metrics.metric_sets.keys()
            )
        else:
            result["metric_sets"] = []

        # Solver names (deduplicated)
        solvers: set[str] = set()
        if self._solver_raw is not None:
            for _modality_key, tiers in self._solver_raw.items():
                if not isinstance(tiers, dict):
                    continue
                for _tier, info in tiers.items():
                    if isinstance(info, dict) and "name" in info:
                        solvers.add(info["name"])
        result["solvers"] = sorted(solvers)

        return result

    def get_modality(self, key: str) -> ModalityYaml:
        """Return the :class:`ModalityYaml` for a given modality key.

        Parameters
        ----------
        key : str
            Modality key (e.g. ``"cassi"``).

        Returns
        -------
        ModalityYaml
            Validated modality definition.

        Raises
        ------
        RegistryKeyError
            If the key does not exist or ``modalities.yaml`` is not loaded.
        """
        if self._modalities is None:
            raise RegistryKeyError(
                "modalities.yaml not loaded; cannot retrieve modality "
                f"'{key}'."
            )
        self.assert_modality_exists(key)
        return self._modalities.modalities[key]

    def list_modalities(self) -> List[str]:
        """Return a sorted list of all valid modality keys.

        Returns
        -------
        list[str]
            Sorted modality keys, or an empty list if ``modalities.yaml``
            is not loaded.
        """
        if self._modalities is None:
            return []
        return sorted(self._modalities.modalities.keys())

    def get_mismatch_params(self, modality_key: str) -> MismatchModalityYaml:
        """Return the mismatch specification for a modality.

        Parameters
        ----------
        modality_key : str
            Modality key (e.g. ``"cassi"``).

        Returns
        -------
        MismatchModalityYaml
            Validated mismatch parameters, severity weights, and correction
            method.

        Raises
        ------
        RegistryKeyError
            If the modality is not found in the mismatch database.
        """
        if self._mismatch is None:
            raise RegistryKeyError(
                "mismatch_db.yaml not loaded; cannot retrieve mismatch "
                f"parameters for '{modality_key}'."
            )
        if modality_key not in self._mismatch.modalities:
            raise RegistryKeyError(
                f"No mismatch entry for modality '{modality_key}'. "
                f"Available: {sorted(self._mismatch.modalities.keys())}"
            )
        return self._mismatch.modalities[modality_key]

    def get_photon_model(self, modality_key: str) -> PhotonModelYaml:
        """Return the photon-budget model for a modality.

        Parameters
        ----------
        modality_key : str
            Modality key (e.g. ``"cassi"``).

        Returns
        -------
        PhotonModelYaml
            Validated photon model specification.

        Raises
        ------
        RegistryKeyError
            If the modality is not found in the photon database.
        """
        if self._photon is None:
            raise RegistryKeyError(
                "photon_db.yaml not loaded; cannot retrieve photon model "
                f"for '{modality_key}'."
            )
        if modality_key not in self._photon.modalities:
            raise RegistryKeyError(
                f"No photon model for modality '{modality_key}'. "
                f"Available: {sorted(self._photon.modalities.keys())}"
            )
        return self._photon.modalities[modality_key]

    def get_calibration_entries(
        self, modality_key: str
    ) -> List[CalibrationEntry]:
        """Return the calibration table entries for a modality.

        Parameters
        ----------
        modality_key : str
            Modality key (e.g. ``"cassi"``).

        Returns
        -------
        list[CalibrationEntry]
            Ordered calibration measurements.

        Raises
        ------
        RegistryKeyError
            If the modality is not found in the compression database.
        """
        if self._compression is None:
            raise RegistryKeyError(
                "compression_db.yaml not loaded; cannot retrieve "
                f"calibration entries for '{modality_key}'."
            )
        if modality_key not in self._compression.calibration_tables:
            raise RegistryKeyError(
                f"No calibration table for modality '{modality_key}'. "
                f"Available: "
                f"{sorted(self._compression.calibration_tables.keys())}"
            )
        return self._compression.calibration_tables[modality_key].entries

    def get_metrics_for_modality(self, modality_key: str) -> List[str]:
        """Return the list of applicable metric names for a modality.

        Searches all metric sets in ``metrics_db.yaml`` and collects the
        metrics from every set whose ``modalities`` list contains
        *modality_key*.

        Parameters
        ----------
        modality_key : str
            Modality key (e.g. ``"cassi"``).

        Returns
        -------
        list[str]
            Deduplicated, sorted list of metric identifiers (e.g.
            ``["psnr", "sam", "ssim"]``).  Returns an empty list if
            ``metrics_db.yaml`` is not loaded or the modality has no
            associated metrics.
        """
        if self._metrics is None:
            logger.warning(
                "metrics_db.yaml not loaded; returning empty metrics "
                "for '%s'.",
                modality_key,
            )
            return []

        metrics: set[str] = set()
        for metric_set in self._metrics.metric_sets.values():
            if modality_key in metric_set.modalities:
                metrics.update(metric_set.metrics)

        return sorted(metrics)

    # ===================================================================
    # Repr
    # ===================================================================

    def __repr__(self) -> str:
        n_mod = len(self.list_modalities())
        n_mis = (
            len(self._mismatch.modalities) if self._mismatch is not None else 0
        )
        n_pho = (
            len(self._photon.modalities) if self._photon is not None else 0
        )
        n_met = (
            len(self._metrics.metric_sets) if self._metrics is not None else 0
        )
        return (
            f"<RegistryBuilder modalities={n_mod} mismatch={n_mis} "
            f"photon={n_pho} metric_sets={n_met} "
            f"contrib_dir='{self._contrib_dir}'>"
        )
