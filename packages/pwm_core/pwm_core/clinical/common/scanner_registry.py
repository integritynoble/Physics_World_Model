"""Scanner model registry for clinical QC.

v1 ships with manually curated entries for common CT models from published
literature and vendor specifications. Community contributions expand the
registry over time.

Usage
-----
>>> from pwm_core.clinical.common.scanner_registry import ScannerRegistry
>>> registry = ScannerRegistry()
>>> model = registry.get_model("Siemens", "SOMATOM Force")
>>> model.typical_noise_std
7.5
>>> registry.list_models()  # returns all built-in ScannerModelInfo objects
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScannerModelInfo(BaseModel):
    """Static specification for a scanner model.

    Attributes
    ----------
    manufacturer : str
        Scanner manufacturer (e.g., ``"Siemens"``, ``"GE"``).
    model : str
        Model name (e.g., ``"SOMATOM Force"``).
    modality : Literal["CT", "PET_CT", "SPECT_CT"]
        Imaging modality category.
    typical_noise_std : float | None
        Expected noise standard deviation in the uniform region of a
        phantom scan, in HU. ``None`` if not characterized.
    typical_uniformity : float | None
        Expected uniformity (center-to-edge HU difference) for a phantom
        scan. ``None`` if not characterized.
    tube_types : list[str]
        X-ray tube type(s) (e.g., ``["Straton"]``).
    detector_type : str | None
        Detector technology description.
    max_rotation_speed : float | None
        Fastest gantry rotation time in seconds.
    year_introduced : int | None
        Year the model was introduced to market.
    notes : str
        Free-text notes or caveats about this model.
    """

    manufacturer: str
    model: str
    modality: Literal["CT", "PET_CT", "SPECT_CT"]
    typical_noise_std: float | None = None
    typical_uniformity: float | None = None
    tube_types: list[str] = Field(default_factory=list)
    detector_type: str | None = None
    max_rotation_speed: float | None = None
    year_introduced: int | None = None
    notes: str = ""


class ScannerInstance(BaseModel):
    """A site-specific scanner installation.

    Extends :class:`ScannerModelInfo` with location and service information.

    Attributes
    ----------
    scanner_id : str
        Unique identifier for this specific scanner installation
        (e.g., ``"HOSP-CT-01"``).
    model_info : ScannerModelInfo
        Reference to the scanner model specification.
    site_name : str
        Name of the facility or hospital site.
    location : str
        Room or building location within the site.
    installation_date : str | None
        Date of initial installation (ISO 8601 format preferred).
    last_service_date : str | None
        Date of most recent service/PM visit.
    custom_params : dict
        Site-specific parameter overrides (e.g., locally measured noise
        baselines, custom thresholds).
    """

    scanner_id: str
    model_info: ScannerModelInfo
    site_name: str = ""
    location: str = ""
    installation_date: str | None = None
    last_service_date: str | None = None
    custom_params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in scanner model database (v1, manually curated)
# ---------------------------------------------------------------------------

def _build_builtin_models() -> list[ScannerModelInfo]:
    """Construct the v1 built-in scanner model list.

    Sources: published ACR phantom test results, vendor datasheets,
    AAPM TG-233 reference data, and peer-reviewed literature.

    Returns
    -------
    list[ScannerModelInfo]
        Manually curated scanner model entries.
    """
    return [
        ScannerModelInfo(
            manufacturer="Siemens",
            model="SOMATOM Force",
            modality="CT",
            typical_noise_std=7.5,
            typical_uniformity=2.0,
            tube_types=["Vectron"],
            detector_type="Stellar Infinity (dual-source)",
            max_rotation_speed=0.25,
            year_introduced=2014,
            notes=(
                "Dual-source CT. Two X-ray tubes and two detector arrays "
                "enable high temporal resolution and dual-energy imaging. "
                "Noise values are for standard body protocol at 120 kVp."
            ),
        ),
        ScannerModelInfo(
            manufacturer="Siemens",
            model="SOMATOM Definition AS+",
            modality="CT",
            typical_noise_std=8.0,
            typical_uniformity=None,
            tube_types=["Straton"],
            detector_type="Ultra Fast Ceramic (UFC)",
            max_rotation_speed=0.3,
            year_introduced=2010,
            notes=(
                "Single-source workhorse CT. Widely deployed in clinical "
                "practice. Noise reference is for standard body protocol "
                "at 120 kVp with B30f kernel."
            ),
        ),
        ScannerModelInfo(
            manufacturer="GE",
            model="Revolution CT",
            modality="CT",
            typical_noise_std=8.5,
            typical_uniformity=2.5,
            tube_types=["Performix HD"],
            detector_type="Gemstone Clarity (160 mm coverage)",
            max_rotation_speed=0.28,
            year_introduced=2014,
            notes=(
                "Wide-coverage (16 cm) CT with 256 rows. Supports cardiac "
                "imaging in a single rotation. Noise values at 120 kVp "
                "standard protocol."
            ),
        ),
        ScannerModelInfo(
            manufacturer="Philips",
            model="iCT 256",
            modality="CT",
            typical_noise_std=9.0,
            typical_uniformity=2.5,
            tube_types=["iMRC"],
            detector_type="NanoPanel Prism (128 rows, 256-slice)",
            max_rotation_speed=0.27,
            year_introduced=2012,
            notes=(
                "256-slice CT with 8 cm z-coverage. iDose4 iterative "
                "reconstruction available. Noise values for standard FBP "
                "at 120 kVp."
            ),
        ),
        ScannerModelInfo(
            manufacturer="Canon",
            model="Aquilion ONE",
            modality="CT",
            typical_noise_std=8.0,
            typical_uniformity=None,
            tube_types=["Megacool"],
            detector_type="PUREViSION (320 rows, 640-slice)",
            max_rotation_speed=0.275,
            year_introduced=2007,
            notes=(
                "320-detector-row volumetric CT (16 cm z-coverage). "
                "Enables whole-organ perfusion in a single rotation. "
                "GENESIS edition (2017) improved dose efficiency. "
                "Noise reference at 120 kVp standard body."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class ScannerRegistry:
    """In-memory registry of scanner models and site-specific instances.

    The registry is initialized with built-in scanner model data for common
    CT scanners. Site-specific instances can be registered at runtime.

    Examples
    --------
    >>> reg = ScannerRegistry()
    >>> force = reg.get_model("Siemens", "SOMATOM Force")
    >>> force.typical_noise_std
    7.5

    >>> from pwm_core.clinical.common.scanner_registry import ScannerInstance
    >>> inst = ScannerInstance(
    ...     scanner_id="MAIN-CT-01",
    ...     model_info=force,
    ...     site_name="General Hospital",
    ...     location="Building A, Room 102",
    ... )
    >>> reg.register_instance(inst)
    >>> reg.get_instance("MAIN-CT-01").site_name
    'General Hospital'
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in scanner models."""
        self._models: list[ScannerModelInfo] = _build_builtin_models()
        self._instances: dict[str, ScannerInstance] = {}

    # -- Model queries -------------------------------------------------------

    def get_model(
        self,
        manufacturer: str,
        model: str,
    ) -> ScannerModelInfo | None:
        """Look up a scanner model by manufacturer and model name.

        Parameters
        ----------
        manufacturer : str
            Manufacturer name. Matching is case-insensitive.
        model : str
            Model name. Matching is case-insensitive.

        Returns
        -------
        ScannerModelInfo | None
            The matching model info, or ``None`` if not found.
        """
        manufacturer_lower = manufacturer.lower()
        model_lower = model.lower()
        for m in self._models:
            if (
                m.manufacturer.lower() == manufacturer_lower
                and m.model.lower() == model_lower
            ):
                return m
        return None

    def list_models(self) -> list[ScannerModelInfo]:
        """Return all registered scanner models.

        Returns
        -------
        list[ScannerModelInfo]
            A copy of the internal model list.
        """
        return list(self._models)

    # -- Instance management -------------------------------------------------

    def register_instance(self, instance: ScannerInstance) -> None:
        """Register a site-specific scanner instance.

        Parameters
        ----------
        instance : ScannerInstance
            The scanner instance to register. If an instance with the same
            ``scanner_id`` already exists, it will be overwritten.
        """
        self._instances[instance.scanner_id] = instance

    def get_instance(self, scanner_id: str) -> ScannerInstance | None:
        """Retrieve a registered scanner instance by its unique ID.

        Parameters
        ----------
        scanner_id : str
            The unique identifier of the scanner instance.

        Returns
        -------
        ScannerInstance | None
            The matching instance, or ``None`` if not found.
        """
        return self._instances.get(scanner_id)

    def list_instances(self) -> list[ScannerInstance]:
        """Return all registered scanner instances.

        Returns
        -------
        list[ScannerInstance]
            All site-specific scanner instances currently registered.
        """
        return list(self._instances.values())

    def register_model(self, model_info: ScannerModelInfo) -> None:
        """Add a scanner model to the registry.

        Allows community contributions to extend the built-in database
        at runtime. If a model with the same manufacturer and model name
        already exists, it will be replaced.

        Parameters
        ----------
        model_info : ScannerModelInfo
            The scanner model specification to register.
        """
        # Check for existing entry and replace if found
        for i, existing in enumerate(self._models):
            if (
                existing.manufacturer.lower() == model_info.manufacturer.lower()
                and existing.model.lower() == model_info.model.lower()
            ):
                self._models[i] = model_info
                return
        self._models.append(model_info)
