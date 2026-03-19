"""pwm_core.agents.photon_agent

Deterministic Photon Agent for the PWM agent system.

Computes photon budgets, signal-to-noise ratios, noise regimes, and noise
models for any imaging modality in the registry.  All computation is fully
deterministic -- the optional LLM client is used only for generating
human-readable explanation narratives.

Physics models
--------------
The ``PhotonModelRegistry`` maps ``model_id`` strings to closed-form photon
budget computations:

- ``microscopy_fluorescence``  -- N = P * QE * (NA/n)^2 / (4*pi) * t / E_photon
- ``ct_xray``                  -- N = N_0 * exp(-mu*L) * eta_det
- ``mri_thermal``              -- SNR proxy ~ B0 * sqrt(V * N_avg / BW) * 1e6
- ``photoacoustic_optical``    -- N = fluence * mu_a * Grueneisen * eta
- ``oct_interferometric``      -- N = P_ref * P_sample / E_photon * eta
- ``generic_detector``         -- N = N_source * QE * t

Noise regime classification uses **variance dominance** (not fixed SNR
thresholds), ensuring physically meaningful categorisation across modalities.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .base import AgentContext, BaseAgent
from .contracts import NoiseRegime, PhotonReport

if TYPE_CHECKING:
    from .registry import RegistryBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_PLANCK_H: float = 6.62607015e-34   # J*s
_SPEED_C: float = 2.99792458e8      # m/s


def _photon_energy(wavelength_nm: float) -> float:
    """Return the energy of a single photon at *wavelength_nm* in joules.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometres.

    Returns
    -------
    float
        Energy per photon in joules.

    Raises
    ------
    ValueError
        If *wavelength_nm* is non-positive.
    """
    if wavelength_nm <= 0.0:
        raise ValueError(
            f"Wavelength must be positive, got {wavelength_nm} nm."
        )
    wavelength_m = wavelength_nm * 1e-9
    return _PLANCK_H * _SPEED_C / wavelength_m


# ═══════════════════════════════════════════════════════════════════════════════
# Individual photon-budget models
# ═══════════════════════════════════════════════════════════════════════════════


def _microscopy_fluorescence(params: Dict[str, Any]) -> float:
    """Fluorescence microscopy photon count at the detector.

    Formula
    -------
    N = power * QE * (NA / n_medium)^2 / (4 * pi) * exposure / E_photon

    Required keys in *params*: ``power_w``, ``qe``, ``na``, ``n_medium``,
    ``exposure_s``, ``wavelength_nm``.

    Parameters
    ----------
    params : dict
        Physical parameters for the fluorescence model.

    Returns
    -------
    float
        Estimated photon count at the detector.
    """
    power_w: float = float(params["power_w"])
    qe: float = float(params["qe"])
    na: float = float(params["na"])
    n_medium: float = float(params["n_medium"])
    exposure_s: float = float(params["exposure_s"])
    wavelength_nm: float = float(params["wavelength_nm"])

    e_photon = _photon_energy(wavelength_nm)
    collection_solid_angle = (na / n_medium) ** 2 / (4.0 * math.pi)

    n_photons = power_w * qe * collection_solid_angle * exposure_s / e_photon
    return max(n_photons, 0.0)


def _ct_xray(params: Dict[str, Any]) -> float:
    """X-ray CT photon count after Beer--Lambert attenuation.

    Formula
    -------
    N = tube_current_photons * exp(-mu * L) * eta_det

    Required keys: ``tube_current_photons``, ``mu``, ``L``, ``eta_det``.

    Parameters
    ----------
    params : dict
        Physical parameters for the CT X-ray model.

    Returns
    -------
    float
        Estimated photon count at the detector.
    """
    n0: float = float(params["tube_current_photons"])
    mu: float = float(params["mu"])
    length: float = float(params["L"])
    eta_det: float = float(params["eta_det"])

    n_photons = n0 * math.exp(-mu * length) * eta_det
    return max(n_photons, 0.0)


def _mri_thermal(params: Dict[str, Any]) -> float:
    """MRI thermal-noise SNR proxy expressed as photon-equivalent count.

    Formula
    -------
    SNR_proxy = B0 * sqrt(voxel_mm3 * n_averages / bandwidth_hz) * 1e6

    This is not a true photon count but a dimensionless proxy that allows
    the downstream SNR and noise-regime logic to operate uniformly.

    Required keys: ``B0``, ``voxel_mm3``, ``n_averages``, ``bandwidth_hz``.

    Parameters
    ----------
    params : dict
        Physical parameters for the MRI thermal model.

    Returns
    -------
    float
        Photon-equivalent SNR proxy.
    """
    b0: float = float(params["B0"])
    voxel_mm3: float = float(params["voxel_mm3"])
    n_avg: float = float(params["n_averages"])
    bw_hz: float = float(params["bandwidth_hz"])

    if bw_hz <= 0.0:
        raise ValueError(
            f"MRI bandwidth must be positive, got {bw_hz} Hz."
        )

    snr_proxy = b0 * math.sqrt(voxel_mm3 * n_avg / bw_hz) * 1e6
    return max(snr_proxy, 0.0)


def _photoacoustic_optical(params: Dict[str, Any]) -> float:
    """Photoacoustic imaging photon-equivalent count.

    Formula
    -------
    N = fluence * mu_a * grueneisen * eta_det * detector_area / E_photon

    Required keys: ``fluence_j_per_cm2``, ``mu_a_per_cm``,
    ``grueneisen``, ``eta_det``, ``detector_area_cm2``,
    ``wavelength_nm``.

    Parameters
    ----------
    params : dict
        Physical parameters for the photoacoustic model.

    Returns
    -------
    float
        Estimated photon-equivalent count.
    """
    fluence: float = float(params["fluence_j_per_cm2"])
    mu_a: float = float(params["mu_a_per_cm"])
    grueneisen: float = float(params["grueneisen"])
    eta_det: float = float(params["eta_det"])
    area: float = float(params["detector_area_cm2"])
    wavelength_nm: float = float(params["wavelength_nm"])

    e_photon = _photon_energy(wavelength_nm)
    # Energy deposited = fluence * mu_a * volume proxy (area * 1cm depth)
    # Pressure ~ Grueneisen * deposited energy density
    # Detected signal ~ pressure * eta * area
    n_photons = fluence * mu_a * grueneisen * eta_det * area / e_photon
    return max(n_photons, 0.0)


def _oct_interferometric(params: Dict[str, Any]) -> float:
    """OCT interferometric photon count.

    Formula
    -------
    N = sqrt(P_ref * P_sample) * eta * exposure / E_photon

    Required keys: ``power_ref_w``, ``power_sample_w``, ``eta_det``,
    ``exposure_s``, ``wavelength_nm``.

    Parameters
    ----------
    params : dict
        Physical parameters for the OCT model.

    Returns
    -------
    float
        Estimated photon count at the detector.
    """
    p_ref: float = float(params["power_ref_w"])
    p_sample: float = float(params["power_sample_w"])
    eta: float = float(params["eta_det"])
    exposure: float = float(params["exposure_s"])
    wavelength_nm: float = float(params["wavelength_nm"])

    e_photon = _photon_energy(wavelength_nm)
    # Interferometric signal scales with geometric mean of reference and
    # sample arm powers.
    n_photons = math.sqrt(p_ref * p_sample) * eta * exposure / e_photon
    return max(n_photons, 0.0)


def _generic_detector(params: Dict[str, Any]) -> float:
    """Generic detector photon count.

    Formula
    -------
    N = source_photons * QE * exposure_s

    Required keys: ``source_photons``, ``qe``, ``exposure_s``.

    Parameters
    ----------
    params : dict
        Physical parameters for the generic detector model.

    Returns
    -------
    float
        Estimated photon count at the detector.
    """
    source: float = float(params["source_photons"])
    qe: float = float(params["qe"])
    exposure: float = float(params["exposure_s"])

    n_photons = source * qe * exposure
    return max(n_photons, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Photon model registry
# ═══════════════════════════════════════════════════════════════════════════════


class PhotonModelRegistry:
    """Maps ``model_id`` strings to deterministic photon-budget functions.

    Each registered function has signature ``(params: dict) -> float`` and
    returns the estimated photon count (or photon-equivalent proxy) at the
    detector plane.

    Supported model IDs
    -------------------
    - ``microscopy_fluorescence``
    - ``ct_xray``
    - ``mri_thermal``
    - ``photoacoustic_optical``
    - ``oct_interferometric``
    - ``generic_detector``
    """

    def __init__(self) -> None:
        self._models: Dict[str, Callable[[Dict[str, Any]], float]] = {
            "microscopy_fluorescence": _microscopy_fluorescence,
            "ct_xray": _ct_xray,
            "mri_thermal": _mri_thermal,
            "photoacoustic_optical": _photoacoustic_optical,
            "oct_interferometric": _oct_interferometric,
            "generic_detector": _generic_detector,
        }

    @property
    def available_models(self) -> List[str]:
        """Return sorted list of registered model IDs."""
        return sorted(self._models.keys())

    def compute(self, model_id: str, params: Dict[str, Any]) -> float:
        """Compute photon count using the model identified by *model_id*.

        Parameters
        ----------
        model_id : str
            One of the supported model identifiers.
        params : dict
            Physical parameters required by the selected model.

        Returns
        -------
        float
            Estimated photon count (or photon-equivalent proxy) at the
            detector.

        Raises
        ------
        KeyError
            If *model_id* is not in the registry.
        KeyError, ValueError
            If required parameters are missing or invalid.
        """
        if model_id not in self._models:
            raise KeyError(
                f"Unknown photon model '{model_id}'. "
                f"Available: {self.available_models}"
            )
        func = self._models[model_id]
        try:
            return func(params)
        except KeyError as exc:
            raise KeyError(
                f"Missing parameter for model '{model_id}': {exc}"
            ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# Photon Agent
# ═══════════════════════════════════════════════════════════════════════════════


class PhotonAgent(BaseAgent):
    """Deterministic Photon Agent for photon budget and noise analysis.

    Walks the imaging system element chain, computes cumulative throughput,
    estimates the photon count at the detector, derives SNR, classifies the
    noise regime and noise model, and assigns a quality tier.

    All computation is deterministic.  The LLM client, if available, is used
    only to generate an optional explanation narrative appended to the report.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Optional LLM client for narrative generation.
    registry : RegistryBuilder, optional
        Source of truth for modality and photon-model metadata.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        registry: Optional["RegistryBuilder"] = None,
    ) -> None:
        super().__init__(llm_client=llm_client, registry=registry)
        self._photon_registry = PhotonModelRegistry()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, context: AgentContext) -> PhotonReport:
        """Execute the photon budget analysis pipeline.

        Steps
        -----
        1. Walk the element chain to compute cumulative throughput.
        2. Retrieve the photon model from the registry and compute raw
           detector photon count.
        3. Apply cumulative throughput to get effective photon count.
        4. Compute SNR in dB.
        5. Classify noise regime via variance dominance.
        6. Classify noise model.
        7. Assign quality tier.
        8. Optionally generate LLM explanation narrative.

        Parameters
        ----------
        context : AgentContext
            Shared pipeline state.  Must contain ``budget`` with at least
            ``model_id`` and ``params`` keys (or an ``imaging_system`` with
            element chain information).

        Returns
        -------
        PhotonReport
            Complete photon budget report.

        Raises
        ------
        ValueError
            If required budget information is missing or invalid.
        """
        budget = context.budget or {}

        # --- 1. Extract photon model parameters ---------------------------
        model_id, model_params = self._resolve_photon_model(context)

        # --- 2. Walk element chain for cumulative throughput ---------------
        throughput_chain, cumulative_throughput = self._compute_throughput(
            context
        )

        # --- 3. Compute raw photon count at detector ----------------------
        raw_photons = self._photon_registry.compute(model_id, model_params)
        n_photons = raw_photons * cumulative_throughput

        # Enforce non-negative (defensive)
        n_photons = max(n_photons, 0.0)

        logger.info(
            "Photon budget: model=%s, raw=%.2e, throughput=%.4f, "
            "effective=%.2e",
            model_id,
            raw_photons,
            cumulative_throughput,
            n_photons,
        )

        # --- 4. Extract noise parameters ----------------------------------
        read_noise = float(budget.get("read_noise", 5.0))
        dark_current = float(budget.get("dark_current", 0.1))
        exposure = float(
            model_params.get("exposure_s", budget.get("exposure_s", 1.0))
        )

        # --- 5. Compute noise variances -----------------------------------
        shot_var = n_photons                       # Poisson variance
        read_var = read_noise ** 2                  # Gaussian read noise
        dark_var = dark_current * exposure          # Dark current accumulation
        total_var = shot_var + read_var + dark_var

        # Noise standard deviations
        shot_sigma = math.sqrt(max(shot_var, 0.0))
        read_sigma = read_noise                     # Already a std dev
        total_sigma = math.sqrt(max(total_var, 0.0))

        # --- 6. Compute SNR (dB) ------------------------------------------
        if total_sigma > 0.0:
            snr_linear = n_photons / total_sigma
        else:
            # Perfect noiseless detector (theoretical edge case)
            snr_linear = float("inf") if n_photons > 0.0 else 0.0

        if snr_linear > 0.0 and not math.isinf(snr_linear):
            snr_db = 20.0 * math.log10(snr_linear)
        elif math.isinf(snr_linear):
            snr_db = 300.0  # Cap at a very high but finite value
        else:
            snr_db = 0.0

        # --- 7. Classify noise regime (variance dominance) ----------------
        noise_regime = self._classify_noise_regime(
            n_photons, shot_var, read_var, dark_var, total_var
        )

        # --- 8. Classify noise model (variance dominance) -----------------
        noise_model = self._classify_noise_model(
            shot_var, read_var, dark_var, total_var
        )

        # --- 9. Quality tier based on SNR in dB ---------------------------
        quality_tier = self._classify_quality_tier(snr_db)

        # --- 10. Feasibility check ----------------------------------------
        feasible = snr_db > 10.0

        # --- 11. Optional LLM explanation ---------------------------------
        explanation = self._generate_explanation(
            model_id=model_id,
            n_photons=n_photons,
            snr_db=snr_db,
            noise_regime=noise_regime,
            noise_model=noise_model,
            quality_tier=quality_tier,
            cumulative_throughput=cumulative_throughput,
        )

        # --- 12. Look up recommended photon levels from registry ----------
        recommended_levels = None
        noise_recipe = None
        if self.registry is not None:
            try:
                photon_meta = self._lookup_registry_photon_model(
                    context.modality_key
                )
                if photon_meta is not None:
                    raw_levels = photon_meta.get("photon_levels")
                    if raw_levels:
                        recommended_levels = {
                            k: dict(v) if isinstance(v, dict) else v.model_dump()
                            for k, v in raw_levels.items()
                        }
                    nm = photon_meta.get("noise_model")
                    if nm:
                        noise_recipe = {
                            "noise_model": nm,
                            "default_level": "standard",
                        }
            except Exception:
                logger.debug(
                    "Photon level lookup failed for '%s'; skipping.",
                    context.modality_key,
                    exc_info=True,
                )

        # --- 13. Build report ---------------------------------------------
        report = PhotonReport(
            n_photons_per_pixel=n_photons,
            snr_db=round(snr_db, 2),
            noise_regime=noise_regime,
            shot_noise_sigma=round(shot_sigma, 4),
            read_noise_sigma=round(read_sigma, 4),
            total_noise_sigma=round(total_sigma, 4),
            feasible=feasible,
            quality_tier=quality_tier,
            throughput_chain=throughput_chain,
            noise_model=noise_model,
            explanation=explanation,
            recommended_levels=recommended_levels,
            noise_recipe=noise_recipe,
        )

        logger.info(
            "PhotonAgent complete: SNR=%.1f dB, regime=%s, model=%s, "
            "tier=%s, feasible=%s",
            snr_db,
            noise_regime.value,
            noise_model,
            quality_tier,
            feasible,
        )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_photon_model(
        self, context: AgentContext
    ) -> tuple[str, Dict[str, Any]]:
        """Resolve the photon model ID and parameters from context.

        Resolution order:
        1. Explicit ``model_id`` and ``params`` in ``context.budget``.
        2. Look up from registry using ``context.modality_key``.
        3. Fall back to ``generic_detector`` with conservative defaults.

        Parameters
        ----------
        context : AgentContext
            Shared pipeline state.

        Returns
        -------
        tuple[str, dict]
            ``(model_id, params)`` ready for computation.
        """
        budget = context.budget or {}

        # Explicit specification in budget
        if "model_id" in budget and "params" in budget:
            return str(budget["model_id"]), dict(budget["params"])

        # Registry lookup via photon_db metadata
        if self.registry is not None:
            try:
                photon_meta = self._lookup_registry_photon_model(
                    context.modality_key
                )
                if photon_meta is not None:
                    return (
                        str(photon_meta["model_id"]),
                        dict(photon_meta["parameters"]),
                    )
            except Exception:
                logger.warning(
                    "Registry lookup for photon model '%s' failed; "
                    "using fallback.",
                    context.modality_key,
                    exc_info=True,
                )

        # Conservative fallback
        logger.warning(
            "No photon model found for modality '%s'; using "
            "generic_detector with conservative defaults.",
            context.modality_key,
        )
        return "generic_detector", {
            "source_photons": 1e6,
            "qe": 0.7,
            "exposure_s": 0.01,
        }

    def _lookup_registry_photon_model(
        self, modality_key: str
    ) -> Optional[Dict[str, Any]]:
        """Look up photon model metadata from the registry.

        Parameters
        ----------
        modality_key : str
            Registry key for the imaging modality.

        Returns
        -------
        dict or None
            Dictionary with ``model_id`` and ``parameters`` keys, or
            ``None`` if not found.
        """
        registry = self.registry
        if registry is None:
            return None

        # Try photon_db attribute (RegistryBuilder convention)
        if hasattr(registry, "photon_db"):
            photon_db = registry.photon_db
            if hasattr(photon_db, "modalities"):
                modalities = photon_db.modalities
                if modality_key in modalities:
                    entry = modalities[modality_key]
                    result = {
                        "model_id": entry.model_id,
                        "parameters": dict(entry.parameters),
                    }
                    if hasattr(entry, "noise_model") and entry.noise_model is not None:
                        result["noise_model"] = entry.noise_model
                    if hasattr(entry, "photon_levels") and entry.photon_levels is not None:
                        result["photon_levels"] = entry.photon_levels
                    return result

        # Try dict-style access
        if hasattr(registry, "get_photon_model"):
            result = registry.get_photon_model(modality_key)
            if result is not None:
                return result

        return None

    def _compute_throughput(
        self, context: AgentContext
    ) -> tuple[List[Dict[str, float]], float]:
        """Walk the element chain to compute per-element and cumulative throughput.

        Parameters
        ----------
        context : AgentContext
            Must contain ``imaging_system`` with an ``elements`` attribute,
            or ``budget`` with a ``throughput_chain`` list.

        Returns
        -------
        tuple[list[dict], float]
            ``(throughput_chain, cumulative_throughput)`` where
            ``throughput_chain`` is a list of ``{"element": name,
            "throughput": value}`` dicts and ``cumulative_throughput`` is
            the product of all element throughputs.
        """
        throughput_chain: List[Dict[str, float]] = []
        cumulative: float = 1.0

        # Try imaging system element chain first
        if context.imaging_system is not None:
            elements = getattr(context.imaging_system, "elements", None)
            if elements is not None:
                for elem in elements:
                    name = getattr(elem, "name", str(elem))
                    tp = float(getattr(elem, "throughput", 1.0))
                    tp = max(0.0, min(1.0, tp))  # Clamp to [0, 1]
                    cumulative *= tp
                    throughput_chain.append({name: tp})
                return throughput_chain, cumulative

        # Try budget-supplied throughput chain
        budget = context.budget or {}
        if "throughput_chain" in budget:
            for entry in budget["throughput_chain"]:
                if isinstance(entry, dict):
                    for name, tp_val in entry.items():
                        tp = float(tp_val)
                        tp = max(0.0, min(1.0, tp))
                        cumulative *= tp
                        throughput_chain.append({name: tp})
            return throughput_chain, cumulative

        # No element chain available -- assume unity throughput
        logger.debug(
            "No element chain found; assuming unity cumulative throughput."
        )
        throughput_chain.append({"system": 1.0})
        return throughput_chain, 1.0

    @staticmethod
    def _classify_noise_regime(
        n_photons: float,
        shot_var: float,
        read_var: float,
        dark_var: float,
        total_var: float,
    ) -> NoiseRegime:
        """Classify noise regime using variance dominance.

        Classification rules (evaluated in order):

        1. If ``shot_var / total > 0.9`` -> ``shot_limited``
        2. If ``read_var / total > 0.5`` -> ``read_limited``
        3. If ``n_photons < 100``        -> ``photon_starved``
        4. Otherwise                     -> ``shot_limited`` (default)

        Parameters
        ----------
        n_photons : float
            Effective photon count at the detector.
        shot_var, read_var, dark_var : float
            Individual variance components.
        total_var : float
            Sum of all variance components.

        Returns
        -------
        NoiseRegime
            Classified noise regime.
        """
        if total_var <= 0.0:
            # Noiseless (theoretical); classify by photon count
            if n_photons < 100.0:
                return NoiseRegime.photon_starved
            return NoiseRegime.shot_limited

        shot_frac = shot_var / total_var
        read_frac = read_var / total_var

        if shot_frac > 0.9:
            return NoiseRegime.shot_limited

        if read_frac > 0.5:
            return NoiseRegime.read_limited

        if n_photons < 100.0:
            return NoiseRegime.photon_starved

        # Mixed regime -- default to shot_limited
        return NoiseRegime.shot_limited

    @staticmethod
    def _classify_noise_model(
        shot_var: float,
        read_var: float,
        dark_var: float,
        total_var: float,
    ) -> str:
        """Classify the noise model using variance dominance.

        Classification rules:

        - If ``shot_var / total > 0.9`` -> ``"poisson"``
        - If ``read_var / total > 0.5`` -> ``"gaussian"``
        - Otherwise                     -> ``"mixed_poisson_gaussian"``

        Parameters
        ----------
        shot_var, read_var, dark_var : float
            Individual variance components.
        total_var : float
            Sum of all variance components.

        Returns
        -------
        str
            One of ``"poisson"``, ``"gaussian"``, or
            ``"mixed_poisson_gaussian"``.
        """
        if total_var <= 0.0:
            return "poisson"

        shot_frac = shot_var / total_var
        read_frac = read_var / total_var

        if shot_frac > 0.9:
            return "poisson"

        if read_frac > 0.5:
            return "gaussian"

        return "mixed_poisson_gaussian"

    @staticmethod
    def _classify_quality_tier(
        snr_db: float,
    ) -> str:
        """Assign a quality tier based on SNR in dB.

        Tiers:

        - ``"excellent"``    : SNR > 30 dB
        - ``"acceptable"``   : SNR > 20 dB
        - ``"marginal"``     : SNR > 10 dB
        - ``"insufficient"`` : SNR <= 10 dB

        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in decibels.

        Returns
        -------
        str
            Quality tier string.
        """
        if snr_db > 30.0:
            return "excellent"
        if snr_db > 20.0:
            return "acceptable"
        if snr_db > 10.0:
            return "marginal"
        return "insufficient"

    def _generate_explanation(
        self,
        *,
        model_id: str,
        n_photons: float,
        snr_db: float,
        noise_regime: NoiseRegime,
        noise_model: str,
        quality_tier: str,
        cumulative_throughput: float,
    ) -> str:
        """Generate a human-readable explanation of the photon budget.

        If an LLM client is available, it is used to produce a richer
        narrative.  Otherwise a deterministic summary is returned.

        Parameters
        ----------
        model_id : str
            Photon model used for computation.
        n_photons : float
            Effective photon count at the detector.
        snr_db : float
            Signal-to-noise ratio in dB.
        noise_regime : NoiseRegime
            Classified noise regime.
        noise_model : str
            Classified noise model.
        quality_tier : str
            Quality tier string.
        cumulative_throughput : float
            Product of all element throughputs.

        Returns
        -------
        str
            Human-readable explanation.
        """
        deterministic_summary = (
            f"Photon budget ({model_id}): {n_photons:.2e} photons/pixel, "
            f"SNR = {snr_db:.1f} dB ({quality_tier}). "
            f"Noise regime: {noise_regime.value}, model: {noise_model}. "
            f"Cumulative throughput: {cumulative_throughput:.4f}."
        )

        if self.llm is not None:
            try:
                prompt = (
                    "You are a computational imaging physicist. Summarise "
                    "the following photon budget analysis in 2-3 clear "
                    "sentences suitable for a lab notebook:\n\n"
                    f"- Photon model: {model_id}\n"
                    f"- Photons per pixel: {n_photons:.2e}\n"
                    f"- SNR: {snr_db:.1f} dB\n"
                    f"- Quality tier: {quality_tier}\n"
                    f"- Noise regime: {noise_regime.value}\n"
                    f"- Noise model: {noise_model}\n"
                    f"- Cumulative throughput: {cumulative_throughput:.4f}\n"
                )
                return self.llm.generate(prompt)
            except Exception:
                logger.warning(
                    "LLM explanation generation failed; using deterministic "
                    "summary.",
                    exc_info=True,
                )

        return deterministic_summary
