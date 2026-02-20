"""Scored root-cause diagnosis engine for CT QC failures.

When a QA metric fails, computes artifact signatures, scores each mismatch
hypothesis against the evidence, and plans minimal disambiguation tests.

The engine loads a *mismatch library* (``clinical_ct_mismatch.yaml``) that
describes known scanner failure modes --- each with feature weights,
expected feature directions, and recommended diagnostic tests.  It then
scores every hypothesis against the observed artifact features and returns
a ranked :class:`DiagnosisReport`.

Scoring algorithm
-----------------
For each mismatch hypothesis *h*:

1. **Evidence score** -- for every feature whose direction matches the
   hypothesis expectation, add ``feature_value * weight``.
2. **Penalty** -- for every feature whose direction contradicts the
   hypothesis, subtract ``feature_value * weight``.
3. **Final score** = evidence_score - penalty.

Hypotheses are ranked by score.  Confidence is set based on the gap
between top-ranked candidates.  If the top two are within 20 %, a
disambiguation test is recommended.

Example YAML entry (``clinical_ct_mismatch.yaml``)::

    mismatches:
      - mismatch_id: "detector_gain_drift"
        name: "Detector Gain Drift"
        affected_metrics: ["noise_std", "uniformity"]
        feature_weights:
          ring_index: 0.40
          noise_ratio: 0.30
          hu_drift_index: 0.20
          cupping_index: 0.10
        expected_direction:
          ring_index: "high"
          noise_ratio: "high"
          hu_drift_index: "high"
          cupping_index: "low"
        diagnostic_test: "Run air calibration and compare detector channel response"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Optional YAML dependency
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MismatchHypothesis(BaseModel):
    """A scored mismatch hypothesis with supporting evidence.

    Attributes
    ----------
    mismatch_id : str
        Unique identifier from the mismatch library.
    name : str
        Human-readable name of the failure mode.
    score : float
        Aggregate evidence score (higher = more likely).
    evidence : dict[str, float]
        Feature-name to contribution mapping for supporting evidence.
    contradictions : dict[str, float]
        Feature-name to contribution mapping for contradicting evidence.
    confidence : Literal["high", "moderate", "low"]
        Confidence level based on score separation from next candidate.
    """

    mismatch_id: str
    name: str
    score: float
    evidence: dict[str, float]
    contradictions: dict[str, float]
    confidence: Literal["high", "moderate", "low"]


class DiagnosisReport(BaseModel):
    """Complete diagnosis output for a set of QC failures.

    Attributes
    ----------
    failed_metrics : list[str]
        Names of the metrics that triggered the diagnosis.
    artifact_features : dict[str, float]
        Observed artifact signature features.
    ranked_hypotheses : list[MismatchHypothesis]
        Hypotheses ranked by score (highest first).
    recommended_next_test : str | None
        Suggested diagnostic test to disambiguate top candidates,
        or ``None`` if the diagnosis is unambiguous.
    disambiguation_needed : bool
        ``True`` if the top candidates are too close in score.
    timestamp : str
        ISO-8601 timestamp of the diagnosis.
    """

    failed_metrics: list[str]
    artifact_features: dict[str, float]
    ranked_hypotheses: list[MismatchHypothesis]
    recommended_next_test: str | None
    disambiguation_needed: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Direction classification thresholds
# ---------------------------------------------------------------------------

# Features above this threshold are considered "high"; below negative
# of this value are considered "low"; in between is "neutral".
_DIRECTION_THRESHOLD_HIGH: float = 0.5
_DIRECTION_THRESHOLD_LOW: float = -0.5


# ---------------------------------------------------------------------------
# Diagnosis engine
# ---------------------------------------------------------------------------

class DiagnosisEngine:
    """Score root-cause hypotheses against observed artifact features.

    Parameters
    ----------
    mismatch_library_path : Path
        Path to ``clinical_ct_mismatch.yaml``.

    Examples
    --------
    >>> engine = DiagnosisEngine(Path("clinical_ct_mismatch.yaml"))
    >>> report = engine.diagnose(
    ...     failed_metrics=["noise_std", "uniformity"],
    ...     artifact_features={"ring_index": 2.5, "noise_ratio": 1.8, ...},
    ... )
    >>> print(report.ranked_hypotheses[0].name)
    'Detector Gain Drift'
    """

    # Disambiguation threshold: if top-2 scores are within this fraction,
    # recommend a disambiguation test.
    _DISAMBIGUATION_FRACTION: float = 0.20

    def __init__(self, mismatch_library_path: Path) -> None:
        self._library_path = Path(mismatch_library_path)
        self._mismatches: list[dict[str, Any]] = self._load_mismatch_library(
            self._library_path,
        )
        logger.info(
            "Loaded mismatch library with %d hypotheses from '%s'.",
            len(self._mismatches), self._library_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        failed_metrics: list[str],
        artifact_features: dict[str, float],
    ) -> DiagnosisReport:
        """Score all hypotheses and produce a ranked diagnosis report.

        Parameters
        ----------
        failed_metrics : list[str]
            Names of the QC metrics that failed (e.g.
            ``["noise_std", "uniformity"]``).
        artifact_features : dict[str, float]
            Observed artifact signature features (as produced by
            :func:`~pwm_core.clinical.ct.diagnosis_features.compute_all_features`).

        Returns
        -------
        DiagnosisReport
            Ranked diagnosis with confidence assessment and optional
            disambiguation recommendation.
        """
        # Score every hypothesis
        hypotheses: list[MismatchHypothesis] = []
        for mismatch in self._mismatches:
            hypothesis = self._score_hypothesis(
                mismatch, failed_metrics, artifact_features,
            )
            hypotheses.append(hypothesis)

        # Rank by score (descending)
        hypotheses.sort(key=lambda h: h.score, reverse=True)

        # Assign confidence levels based on score separation
        hypotheses = self._assign_confidence(hypotheses)

        # Check disambiguation
        disambiguation_needed = False
        recommended_next_test: str | None = None

        if len(hypotheses) >= 2:
            top_score = hypotheses[0].score
            second_score = hypotheses[1].score

            if top_score > 0 and second_score > 0:
                relative_gap = (top_score - second_score) / top_score
                if relative_gap < self._DISAMBIGUATION_FRACTION:
                    disambiguation_needed = True
                    # Recommend diagnostic test from the higher-ranked hypothesis
                    top_mismatch = self._find_mismatch_by_id(
                        hypotheses[0].mismatch_id,
                    )
                    if top_mismatch:
                        recommended_next_test = top_mismatch.get(
                            "diagnostic_test",
                        )

        return DiagnosisReport(
            failed_metrics=failed_metrics,
            artifact_features=artifact_features,
            ranked_hypotheses=hypotheses,
            recommended_next_test=recommended_next_test,
            disambiguation_needed=disambiguation_needed,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_mismatch_library(self, path: Path) -> list[dict[str, Any]]:
        """Load the mismatch hypothesis library from YAML.

        Parameters
        ----------
        path : Path
            Path to the YAML file.

        Returns
        -------
        list[dict]
            List of mismatch hypothesis definitions.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        FileNotFoundError
            If the YAML file does not exist.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for the diagnosis engine. "
                "Install it with: pip install pyyaml"
            )

        if not path.exists():
            logger.warning(
                "Mismatch library not found at '%s'; "
                "using empty hypothesis list.",
                path,
            )
            return []

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return []

        # Support both production key ("mismatch_types") and legacy key ("mismatches").
        mismatches = raw.get("mismatch_types", raw.get("mismatches", []))
        if not isinstance(mismatches, list):
            logger.warning(
                "Expected 'mismatch_types'/'mismatches' to be a list in '%s'; got %s.",
                path, type(mismatches).__name__,
            )
            return []

        return [self._normalize_mismatch(entry) for entry in mismatches]

    @staticmethod
    def _normalize_mismatch(entry: dict[str, Any]) -> dict[str, Any]:
        """Convert production YAML format to internal format.

        The production ``clinical_ct_mismatch.yaml`` uses a nested
        ``diagnostic_features`` mapping where each feature has
        ``expected_direction`` and ``weight`` sub-keys, plus ``id``
        instead of ``mismatch_id``.  The engine expects flat
        ``feature_weights`` and ``expected_direction`` dicts and a
        ``mismatch_id`` key.  This method bridges both formats so the
        engine works with legacy *and* production YAML files.

        Parameters
        ----------
        entry : dict
            A single mismatch entry from the YAML library.

        Returns
        -------
        dict
            A new dict with ``feature_weights``, ``expected_direction``,
            and ``mismatch_id`` added if they could be derived from the
            nested format.  The original *entry* is not mutated.
        """
        result = dict(entry)  # shallow copy to avoid mutating YAML data

        # Normalize nested diagnostic_features -> flat feature_weights / expected_direction
        if "diagnostic_features" in result and "feature_weights" not in result:
            features = result["diagnostic_features"]
            result["feature_weights"] = {
                k: v["weight"] for k, v in features.items()
            }
            result["expected_direction"] = {
                k: v["expected_direction"] for k, v in features.items()
            }

        # Normalize id -> mismatch_id
        if "id" in result and "mismatch_id" not in result:
            result["mismatch_id"] = result["id"]

        return result

    def _score_hypothesis(
        self,
        mismatch: dict[str, Any],
        failed_metrics: list[str],
        features: dict[str, float],
    ) -> MismatchHypothesis:
        """Score a single mismatch hypothesis against observed features.

        The scoring procedure:

        1. For each feature in the hypothesis ``feature_weights``:

           * If the observed feature direction matches
             ``expected_direction`` -> add ``feature_value * weight``
             to evidence.
           * If the direction contradicts -> add ``feature_value * weight``
             to penalty.

        2. Add a bonus for each ``affected_metric`` that appears in
           ``failed_metrics``.

        3. ``score = evidence_score - penalty + metric_bonus``.

        Parameters
        ----------
        mismatch : dict
            Hypothesis definition from the mismatch library.
        failed_metrics : list[str]
            Currently failing QC metrics.
        features : dict[str, float]
            Observed artifact features.

        Returns
        -------
        MismatchHypothesis
            Scored hypothesis (confidence set to ``"low"`` initially;
            reassigned by :meth:`_assign_confidence` after ranking).
        """
        mismatch_id: str = mismatch.get("mismatch_id", "unknown")
        name: str = mismatch.get("name", mismatch_id)
        feature_weights: dict[str, float] = mismatch.get("feature_weights", {})
        expected_direction: dict[str, str] = mismatch.get("expected_direction", {})
        affected_metrics: list[str] = mismatch.get("affected_metrics", [])

        evidence: dict[str, float] = {}
        contradictions: dict[str, float] = {}
        evidence_score: float = 0.0
        penalty: float = 0.0

        for feature_name, weight in feature_weights.items():
            if feature_name not in features:
                continue

            feature_value = features[feature_name]
            expected = expected_direction.get(feature_name, "high")
            observed_direction = self._classify_direction(feature_value)

            if self._directions_match(expected, observed_direction):
                contribution = abs(feature_value) * weight
                evidence[feature_name] = contribution
                evidence_score += contribution
            elif self._directions_contradict(expected, observed_direction):
                contribution = abs(feature_value) * weight
                contradictions[feature_name] = contribution
                penalty += contribution
            # If observed is "neutral", no contribution either way.

        # Bonus for matching affected metrics
        metric_bonus: float = 0.0
        for metric in affected_metrics:
            if metric in failed_metrics:
                metric_bonus += 0.5  # Each matching failed metric adds 0.5

        score = evidence_score - penalty + metric_bonus

        return MismatchHypothesis(
            mismatch_id=mismatch_id,
            name=name,
            score=score,
            evidence=evidence,
            contradictions=contradictions,
            confidence="low",  # Placeholder; reassigned after ranking
        )

    @staticmethod
    def _assign_confidence(
        hypotheses: list[MismatchHypothesis],
    ) -> list[MismatchHypothesis]:
        """Assign confidence levels based on score separation.

        Rules:

        * ``"high"`` -- top score > 2x the second score.
        * ``"moderate"`` -- top score > 1.2x the second score.
        * ``"low"`` -- otherwise.

        Only the top hypothesis gets elevated confidence; all others
        remain ``"low"``.

        Parameters
        ----------
        hypotheses : list[MismatchHypothesis]
            Hypotheses sorted by score descending.

        Returns
        -------
        list[MismatchHypothesis]
            Same list with confidence levels updated.
        """
        if not hypotheses:
            return hypotheses

        if len(hypotheses) == 1:
            updated = hypotheses[0].model_copy(update={"confidence": "high"})
            return [updated]

        top_score = hypotheses[0].score
        second_score = hypotheses[1].score

        if second_score <= 0:
            confidence: Literal["high", "moderate", "low"] = (
                "high" if top_score > 0 else "low"
            )
        elif top_score > 2.0 * second_score:
            confidence = "high"
        elif top_score > 1.2 * second_score:
            confidence = "moderate"
        else:
            confidence = "low"

        result: list[MismatchHypothesis] = []
        result.append(hypotheses[0].model_copy(update={"confidence": confidence}))
        for h in hypotheses[1:]:
            result.append(h.model_copy(update={"confidence": "low"}))

        return result

    @staticmethod
    def _classify_direction(value: float) -> str:
        """Classify a feature value as ``"high"``, ``"low"``, or ``"neutral"``.

        Parameters
        ----------
        value : float
            Observed feature value.

        Returns
        -------
        str
            ``"high"`` if value > threshold, ``"low"`` if value < negative
            threshold, ``"neutral"`` otherwise.
        """
        if value > _DIRECTION_THRESHOLD_HIGH:
            return "high"
        elif value < _DIRECTION_THRESHOLD_LOW:
            return "low"
        else:
            return "neutral"

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        """Map direction synonyms to canonical ``"high"``/``"low"`` values.

        The production YAML uses vocabulary like ``"elevated"`` and
        ``"decreased"`` while the engine's :meth:`_classify_direction`
        returns ``"high"``, ``"low"``, or ``"neutral"``.  This helper
        maps synonyms so that comparison works regardless of vocabulary.

        Parameters
        ----------
        direction : str
            Direction string from either the YAML or the classifier.

        Returns
        -------
        str
            Canonical direction (``"high"``, ``"low"``, or unchanged
            if no synonym mapping exists).
        """
        synonyms = {
            "elevated": "high",
            "decreased": "low",
            "increased": "high",
            "reduced": "low",
        }
        return synonyms.get(direction, direction)

    @classmethod
    def _directions_match(cls, expected: str, observed: str) -> bool:
        """Check if observed direction matches the hypothesis expectation.

        Both *expected* and *observed* are normalized through
        :meth:`_normalize_direction` before comparison so that YAML
        vocabulary (``"elevated"``/``"decreased"``) and classifier
        vocabulary (``"high"``/``"low"``) are treated as equivalent.

        Parameters
        ----------
        expected : str
            Expected direction (``"high"``, ``"low"``, ``"elevated"``,
            ``"decreased"``, etc.).
        observed : str
            Classified observed direction.

        Returns
        -------
        bool
            ``True`` if the observed direction matches the expected.
        """
        return cls._normalize_direction(expected) == cls._normalize_direction(observed)

    @classmethod
    def _directions_contradict(cls, expected: str, observed: str) -> bool:
        """Check if observed direction contradicts the hypothesis expectation.

        Both *expected* and *observed* are normalized through
        :meth:`_normalize_direction` before comparison.

        Parameters
        ----------
        expected : str
            Expected direction (``"high"``, ``"low"``, ``"elevated"``,
            ``"decreased"``, etc.).
        observed : str
            Classified observed direction.

        Returns
        -------
        bool
            ``True`` if the directions are opposite.
        """
        ne = cls._normalize_direction(expected)
        no = cls._normalize_direction(observed)
        return (
            (ne == "high" and no == "low")
            or (ne == "low" and no == "high")
        )

    def _find_mismatch_by_id(self, mismatch_id: str) -> dict[str, Any] | None:
        """Find a mismatch definition by its ID.

        Parameters
        ----------
        mismatch_id : str
            The mismatch identifier to search for.

        Returns
        -------
        dict | None
            The matching mismatch definition, or ``None``.
        """
        for m in self._mismatches:
            if m.get("mismatch_id") == mismatch_id:
                return m
        return None
