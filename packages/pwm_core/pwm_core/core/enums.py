"""pwm_core.core.enums
======================

Execution mode enum for the v2 pipeline.

Modes
-----
simulate   Mode S — forward simulation only
invert     Mode I — reconstruction / inverse problem
calibrate  Mode C — belief-state calibration loop
"""

from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    """Pipeline execution mode for run_pipeline_v2."""

    simulate = "simulate"
    invert = "invert"
    calibrate = "calibrate"
