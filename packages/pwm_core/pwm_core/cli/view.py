"""pwm_core.cli.view

Launch Streamlit viewer for a RunBundle.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def view(runbundle_dir: str):
    rb = Path(runbundle_dir)
    app = Path(__file__).resolve().parents[1] / "viewer" / "app.py"
    subprocess.check_call(["streamlit", "run", str(app), "--", str(rb)])
