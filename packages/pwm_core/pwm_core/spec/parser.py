"""pwm_core.spec.parser

spec.md <-> CanonicalCoreSpec round-trip parser/renderer.

Implements the strategy S5 contract:
  spec.md -> parse -> CoreSpec (typed) -> render -> spec.md'
  round-trip guarantee: spec.md == spec.md'

The spec.md format uses YAML frontmatter + markdown body:

---
omega:
  type: image_domain
  dims: [64, 64]
equations:
  forward_model: radon
  noise_model: poisson
boundary_conditions: {}
initial_conditions: {}
observables:
  detector: line_array
tolerance: 0.01
spec_version: "1.0.0"
domain_profile_ref: "imaging/v1"
---

# CT Reconstruction Problem

Human-readable description of the problem...
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def parse_spec_md(text: str) -> Dict[str, Any]:
    """Parse a spec.md string into a dict suitable for CanonicalCoreSpec(**d).

    Extracts YAML frontmatter between --- delimiters.
    The markdown body is stored under the '_description' key.
    """
    # Extract YAML frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n?(.*)', text, re.DOTALL)
    if not match:
        raise ValueError("spec.md must start with YAML frontmatter between --- delimiters")

    frontmatter_str = match.group(1)
    body = match.group(2).strip()

    if yaml is None:
        raise ImportError("PyYAML is required for spec.md parsing: pip install pyyaml")

    frontmatter = yaml.safe_load(frontmatter_str) or {}

    # Validate required fields
    required = {"omega", "equations", "observables", "tolerance"}
    missing = required - set(frontmatter.keys())
    if missing:
        raise ValueError(f"spec.md frontmatter missing required fields: {sorted(missing)}")

    # Store body as metadata
    if body:
        frontmatter.setdefault("_description", body)

    return frontmatter


def render_spec_md(spec_dict: Dict[str, Any], title: Optional[str] = None) -> str:
    """Render a CanonicalCoreSpec dict to spec.md format.

    Produces YAML frontmatter + optional markdown body.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for spec.md rendering: pip install pyyaml")

    # Separate description from spec fields
    d = dict(spec_dict)
    description = d.pop("_description", "")

    # Render YAML frontmatter
    frontmatter = yaml.dump(d, default_flow_style=False, sort_keys=False, allow_unicode=True)

    parts = ["---", frontmatter.rstrip(), "---", ""]

    if title:
        parts.append(f"# {title}")
        parts.append("")

    if description:
        parts.append(description)
        parts.append("")

    return "\n".join(parts)


def parse_spec_file(path: str) -> Dict[str, Any]:
    """Parse a spec.md file from disk."""
    return parse_spec_md(Path(path).read_text(encoding="utf-8"))


def render_spec_file(spec_dict: Dict[str, Any], path: str, title: Optional[str] = None) -> Path:
    """Render a spec dict to a spec.md file on disk."""
    p = Path(path)
    p.write_text(render_spec_md(spec_dict, title), encoding="utf-8")
    return p


def roundtrip_spec_md(text: str) -> str:
    """Parse and re-render a spec.md string. Should produce identical output."""
    d = parse_spec_md(text)
    return render_spec_md(d)
