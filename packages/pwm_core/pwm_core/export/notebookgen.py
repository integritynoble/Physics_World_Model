"""pwm_core.export.notebookgen

Generate a minimal Jupyter notebook that loads a RunBundle and renders:
- images
- metrics
- diagnosis
- suggested actions
"""

from __future__ import annotations

from typing import Any, Dict, List


def generate_notebook_json(manifest: Dict[str, Any]) -> Dict[str, Any]:
    run_id = manifest.get("run_id", "")
    cells: List[Dict[str, Any]] = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# PWM RunBundle Viewer\n\nRun: `{run_id}`\n"]
    })

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "root = Path('.').resolve()\n",
            "m = json.loads((root/'runbundle_manifest.json').read_text())\n",
            "m.keys()\n",
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb
