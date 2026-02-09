"""Generate _generated_literals.py from YAML registries.

Usage:
    python -m pwm_core.agents._generate_literals
"""
from __future__ import annotations

import pathlib
import yaml


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate(contrib_dir: pathlib.Path | None = None) -> str:
    if contrib_dir is None:
        contrib_dir = pathlib.Path(__file__).resolve().parents[2] / "contrib"

    # Modality keys
    mod_data = _load_yaml(contrib_dir / "modalities.yaml")
    mod_keys = sorted(mod_data["modalities"].keys())

    # Signal prior keys from compression_db
    comp_data = _load_yaml(contrib_dir / "compression_db.yaml")
    priors = set()
    for table in comp_data.get("calibration_tables", {}).values():
        if isinstance(table, dict):
            sp = table.get("signal_prior_class", "")
            if sp:
                priors.add(sp)
    prior_keys = sorted(priors)

    # Build source
    lines = [
        '"""Auto-generated Literal types from YAML registries.',
        "",
        "DO NOT EDIT MANUALLY. Regenerate with:",
        "    python -m pwm_core.agents._generate_literals",
        "",
        "These types enable static type checking (mypy/pyright) to catch invalid",
        "registry keys at type-check time, not just runtime.",
        '"""',
        "# fmt: off",
        "from typing import Literal",
        "",
        "ModalityKey = Literal[",
    ]
    for k in mod_keys:
        lines.append(f'    "{k}",')
    lines.append("]")
    lines.append("")
    lines.append("SignalPriorKey = Literal[")
    for k in prior_keys:
        lines.append(f'    "{k}",')
    lines.append("]")
    lines.append("")
    lines.append("# fmt: on")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out = pathlib.Path(__file__).resolve().parent / "_generated_literals.py"
    src = generate()
    out.write_text(src)
    print(f"Generated {out} ({len(src)} bytes)")


if __name__ == "__main__":
    main()
