"""pwm_core.core.runbundle.policy

Copy vs reference policy for big datasets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataPolicy:
    mode: str = "auto"  # auto|copy|reference
    copy_threshold_mb: int = 100


DEFAULT_DATA_POLICY = DataPolicy()
