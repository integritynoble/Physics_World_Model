"""pwm_core.agents.negotiator

Agent Negotiator: resolves conflicts between Photon, Recoverability, and
Mismatch agent reports before reconstruction is authorised.

Design mantra
-------------
The negotiator is fully deterministic -- no LLM involvement.  It inspects
the three upstream reports and either approves reconstruction or raises
vetoes with actionable resolution suggestions.

Veto conditions
---------------
1. **Low photon + high compression** -- attempting aggressive CS recovery
   when the measurement is already photon-starved guarantees artefacts.
2. **Severe mismatch without correction** -- if the forward-model mismatch
   exceeds 0.7 severity and no correction is planned, reconstruction will
   converge to a wrong solution.
3. **All three marginal / insufficient** -- if every subsystem is
   individually borderline, the joint probability is too low to justify
   the computation cost.
"""

from __future__ import annotations

import logging
from typing import List

from .contracts import (
    PhotonReport,
    MismatchReport,
    RecoverabilityReport,
    VetoReason,
    NegotiationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Photon: quality tiers considered "low"
_LOW_PHOTON_TIERS = {"marginal", "insufficient"}

# Recoverability: verdicts considered "poor"
_POOR_RECOVERABILITY_VERDICTS = {"marginal", "insufficient"}

# Mismatch: severity above which correction is mandatory
_MISMATCH_SEVERITY_VETO_THRESHOLD = 0.7

# Joint probability below which reconstruction is not recommended
_JOINT_PROBABILITY_FLOOR = 0.15


# ---------------------------------------------------------------------------
# AgentNegotiator
# ---------------------------------------------------------------------------


class AgentNegotiator:
    """Resolves conflicts between the Photon, Recoverability, and Mismatch
    agents and decides whether reconstruction should proceed.

    The negotiator produces a :class:`NegotiationResult` containing:

    * A list of :class:`VetoReason` objects (empty if no conflicts).
    * A ``proceed`` flag (``True`` only if the veto list is empty).
    * A ``probability_of_success`` representing the joint probability from
      all three agent reports.

    Usage
    -----
    >>> negotiator = AgentNegotiator()
    >>> result = negotiator.negotiate(photon_report, recoverability_report, mismatch_report)
    >>> if not result.proceed:
    ...     for veto in result.vetoes:
    ...         print(f"VETO from {veto.source}: {veto.reason}")
    """

    def negotiate(
        self,
        photon: PhotonReport,
        recoverability: RecoverabilityReport,
        mismatch: MismatchReport,
    ) -> NegotiationResult:
        """Run the negotiation logic and return a decision.

        Parameters
        ----------
        photon : PhotonReport
            Output of the Photon Agent.
        recoverability : RecoverabilityReport
            Output of the Recoverability Agent.
        mismatch : MismatchReport
            Output of the Mismatch Agent.

        Returns
        -------
        NegotiationResult
            Vetoes (if any), proceed flag, and joint probability of success.
        """
        vetoes: List[VetoReason] = []

        # -- Conflict 1: Low photon + high compression ------------------------
        vetoes.extend(
            self._check_photon_compression_conflict(photon, recoverability)
        )

        # -- Conflict 2: Severe mismatch without correction -------------------
        vetoes.extend(
            self._check_mismatch_severity(mismatch)
        )

        # -- Conflict 3: All three marginal / insufficient --------------------
        vetoes.extend(
            self._check_all_marginal(photon, recoverability, mismatch)
        )

        # -- Joint probability of success -------------------------------------
        probability = self._compute_joint_probability(
            photon, recoverability, mismatch
        )

        # Additional veto if joint probability is below floor
        if probability < _JOINT_PROBABILITY_FLOOR and not any(
            v.source == "joint_probability" for v in vetoes
        ):
            vetoes.append(
                VetoReason(
                    source="joint_probability",
                    reason=(
                        f"Joint probability of success ({probability:.1%}) is below "
                        f"the minimum threshold ({_JOINT_PROBABILITY_FLOOR:.1%}).  "
                        f"Reconstruction is unlikely to produce a useful result."
                    ),
                    suggested_resolution=[
                        "Address the primary bottleneck identified by the Analysis Agent",
                        "Collect higher-quality measurements",
                        "Reduce the problem difficulty (fewer channels, lower resolution)",
                    ],
                )
            )

        proceed = len(vetoes) == 0

        logger.info(
            "Negotiation complete: proceed=%s, vetoes=%d, p_success=%.2f",
            proceed,
            len(vetoes),
            probability,
        )

        return NegotiationResult(
            vetoes=vetoes,
            proceed=proceed,
            probability_of_success=probability,
        )

    # ----- Conflict detectors ------------------------------------------------

    @staticmethod
    def _check_photon_compression_conflict(
        photon: PhotonReport,
        recoverability: RecoverabilityReport,
    ) -> List[VetoReason]:
        """Detect low-photon + high-compression conflict.

        When the photon budget is marginal or insufficient AND the
        recoverability verdict is also poor, compressed sensing recovery
        is unreliable: the measurement noise floor exceeds the signal
        structure the solver needs to exploit.

        Parameters
        ----------
        photon : PhotonReport
        recoverability : RecoverabilityReport

        Returns
        -------
        list[VetoReason]
            Zero or one veto reason.
        """
        photon_is_low = photon.quality_tier in _LOW_PHOTON_TIERS
        recovery_is_poor = recoverability.verdict in _POOR_RECOVERABILITY_VERDICTS

        if photon_is_low and recovery_is_poor:
            return [
                VetoReason(
                    source="photon_compression_conflict",
                    reason=(
                        f"Photon budget is '{photon.quality_tier}' "
                        f"(SNR={photon.snr_db:.1f} dB) while recoverability "
                        f"is '{recoverability.verdict}' "
                        f"(score={recoverability.recoverability_score:.2f}).  "
                        f"Compressed sensing recovery requires sufficient SNR "
                        f"to distinguish signal structure from noise."
                    ),
                    suggested_resolution=[
                        "Increase exposure time or source power to improve SNR",
                        "Reduce compression ratio by acquiring more measurements",
                        "Use a denoising pre-processing step before reconstruction",
                    ],
                )
            ]
        return []

    @staticmethod
    def _check_mismatch_severity(
        mismatch: MismatchReport,
    ) -> List[VetoReason]:
        """Detect severe forward-model mismatch without correction.

        A severity score above the threshold means the forward operator
        used for reconstruction differs substantially from the true
        physical operator.  Without applying operator correction, the
        solver will converge to a wrong solution.

        Parameters
        ----------
        mismatch : MismatchReport

        Returns
        -------
        list[VetoReason]
            Zero or one veto reason.
        """
        if mismatch.severity_score > _MISMATCH_SEVERITY_VETO_THRESHOLD:
            # Check if a correction method is available and non-trivial
            correction_available = (
                mismatch.correction_method
                and mismatch.correction_method.lower() not in ("none", "n/a", "")
            )
            if not correction_available:
                return [
                    VetoReason(
                        source="mismatch_severity",
                        reason=(
                            f"Forward-model mismatch severity is "
                            f"{mismatch.severity_score:.2f} (threshold: "
                            f"{_MISMATCH_SEVERITY_VETO_THRESHOLD:.2f}) and no "
                            f"correction method is available.  The solver will "
                            f"converge to an incorrect solution."
                        ),
                        suggested_resolution=[
                            "Apply operator correction (e.g. UPWMI) before reconstruction",
                            "Recalibrate the forward operator from measured data",
                            "Use a robust solver that tolerates operator uncertainty",
                        ],
                    )
                ]
        return []

    @staticmethod
    def _check_all_marginal(
        photon: PhotonReport,
        recoverability: RecoverabilityReport,
        mismatch: MismatchReport,
    ) -> List[VetoReason]:
        """Detect when all three subsystems are marginal or worse.

        Individually, each subsystem might be borderline acceptable.
        Together, the compounding uncertainties make the reconstruction
        unreliable.

        Parameters
        ----------
        photon : PhotonReport
        recoverability : RecoverabilityReport
        mismatch : MismatchReport

        Returns
        -------
        list[VetoReason]
            Zero or one veto reason.
        """
        photon_marginal = photon.quality_tier in _LOW_PHOTON_TIERS
        recovery_marginal = recoverability.verdict in _POOR_RECOVERABILITY_VERDICTS
        mismatch_marginal = mismatch.severity_score > 0.5

        if photon_marginal and recovery_marginal and mismatch_marginal:
            return [
                VetoReason(
                    source="all_subsystems_marginal",
                    reason=(
                        f"All three subsystems are marginal or worse: "
                        f"photon='{photon.quality_tier}', "
                        f"recoverability='{recoverability.verdict}', "
                        f"mismatch_severity={mismatch.severity_score:.2f}.  "
                        f"The compounding uncertainties make reconstruction "
                        f"unreliable."
                    ),
                    suggested_resolution=[
                        "Address at least one subsystem to bring it to 'acceptable' or better",
                        "Prioritise the subsystem with the highest bottleneck score",
                        "Consider a simpler imaging configuration with fewer unknowns",
                    ],
                )
            ]
        return []

    # ----- Probability computation -------------------------------------------

    @staticmethod
    def _compute_joint_probability(
        photon: PhotonReport,
        recoverability: RecoverabilityReport,
        mismatch: MismatchReport,
    ) -> float:
        """Compute joint probability of success from all three reports.

        Each subsystem contributes a survival factor:

        * **Photon**: maps quality tier to a probability.
        * **Recoverability**: uses the recoverability score directly.
        * **Mismatch**: ``1.0 - severity_score * 0.7`` (mismatch has an
          outsized impact because it biases the solution, not just adds noise).

        The joint probability is the product of the three factors, clipped
        to [0, 1].

        Parameters
        ----------
        photon : PhotonReport
        recoverability : RecoverabilityReport
        mismatch : MismatchReport

        Returns
        -------
        float
            Joint probability of success in [0, 1].
        """
        # Photon factor
        _tier_probabilities = {
            "excellent": 0.95,
            "acceptable": 0.80,
            "marginal": 0.50,
            "insufficient": 0.20,
        }
        p_photon = _tier_probabilities.get(photon.quality_tier, 0.50)

        # Recoverability factor: score is already in [0, 1]
        p_recoverability = recoverability.recoverability_score

        # Mismatch factor: severity has outsized impact
        p_mismatch = 1.0 - mismatch.severity_score * 0.7

        joint = p_photon * p_recoverability * p_mismatch
        return max(0.0, min(1.0, round(joint, 4)))
