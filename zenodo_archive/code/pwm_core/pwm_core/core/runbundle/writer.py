"""pwm_core.core.runbundle.writer

Create RunBundle folder structure.

v3: Added ``agents/`` subdirectory for agent report snapshots.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def write_runbundle_skeleton(out_dir: str, spec_id: str) -> str:
    """Create the RunBundle directory structure.

    Layout::

        run_{spec_id}_{uuid}/
            artifacts/
                images/
            internal_state/
            agents/
            logs/

    Parameters
    ----------
    out_dir : str
        Parent directory for the RunBundle.
    spec_id : str
        Experiment spec ID.

    Returns
    -------
    str
        Absolute path to the created RunBundle directory.
    """
    rb_id = f"run_{spec_id}_{uuid.uuid4().hex[:8]}"
    rb_dir = os.path.join(out_dir, rb_id)
    Path(rb_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(rb_dir, "artifacts")).mkdir(exist_ok=True)
    Path(os.path.join(rb_dir, "artifacts", "images")).mkdir(exist_ok=True)
    Path(os.path.join(rb_dir, "artifacts", "trace")).mkdir(exist_ok=True)
    Path(os.path.join(rb_dir, "artifacts", "trace", "png")).mkdir(exist_ok=True)
    Path(os.path.join(rb_dir, "internal_state")).mkdir(exist_ok=True)
    Path(os.path.join(rb_dir, "agents")).mkdir(exist_ok=True)
    Path(os.path.join(rb_dir, "logs")).mkdir(exist_ok=True)
    return rb_dir
