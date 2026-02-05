"""
pwm_core.api.errors

Typed exceptions for API boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class PWMError(Exception):
    """Base PWM error."""


class SpecValidationError(PWMError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class OperatorLoadError(PWMError):
    pass


class MeasurementLoadError(PWMError):
    pass


class ReconstructionError(PWMError):
    pass


class CalibrationError(PWMError):
    pass


class ExportError(PWMError):
    pass
