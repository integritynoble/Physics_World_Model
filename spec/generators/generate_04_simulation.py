#!/usr/bin/env python3
"""Usage 4: Scientific Simulation — one spec per domain from universal_simulation.

Reads papers/universal_simulation/benchmark/*/spec.md and reformats minimally.

Output: spec/04_simulation/{domain}/spec.md
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import SPEC_DIR

PAPER_SIM = SPEC_DIR.parent / 'papers' / 'universal_simulation' / 'benchmark'
OUT_DIR = SPEC_DIR / '04_simulation'


def extract_run_code(text: str) -> str:
    """Extract Python code block if present."""
    m = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else ''


def extract_section(text: str, section_name: str, max_lines: int = 15) -> str:
    """Extract a ## Section block from spec.md."""
    pattern = rf'(?:^|\n)## {section_name}(.*?)(?=\n## |\Z)'
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not m:
        return ''
    content = m.group(1).strip()
    lines = content.splitlines()
    # Remove blank-only lines from start
    while lines and not lines[0].strip():
        lines.pop(0)
    return '\n'.join(lines[:max_lines])


def make_minimal_spec(domain: str, src_text: str) -> str:
    """Reformat the source spec.md into a minimal run-button spec."""
    lines = src_text.strip().splitlines()

    # Title (first # line)
    title = next((l.lstrip('# ').strip() for l in lines if l.startswith('# ')), domain)

    # Extract sections
    equations = extract_section(src_text, 'Equations', 12)
    params_raw = extract_section(src_text, 'Parameters', 12) or extract_section(src_text, 'parameter', 12)
    # Parse parameters: lines like "  key: value  # comment"
    param_lines = []
    for line in (params_raw or '').splitlines():
        m = re.match(r'\s+(\w+):\s*([^\s#]+)', line)
        if m:
            param_lines.append(f'| `{m.group(1)}` | `{m.group(2)}` |')
    params_table = ''
    if param_lines:
        params_table = '| Parameter | Value |\n|-----------|-------|\n' + '\n'.join(param_lines[:12])
    else:
        params_table = params_raw or '*See source spec.*'

    observables = extract_section(src_text, 'Observables', 6)
    tolerance = extract_section(src_text, 'Tolerance', 4)
    variations = extract_section(src_text, 'Task Instance', 5)

    # Run code
    code = extract_run_code(src_text)

    # Generate a minimal run button if no code in spec
    if not code:
        code = f"""\
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/{domain}/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/{domain}/public/')
# Run the simulation task according to the spec above
"""

    obs_tol = ''
    if observables:
        obs_tol += observables + '\n'
    if tolerance:
        obs_tol += '\n**Tolerance**: ' + ' '.join(tolerance.splitlines()[:2])

    spec = f"""\
# {title}

> Source: `papers/universal_simulation/benchmark/{domain}/spec.md`

## Equations

```
{equations or 'See source spec.'}
```

## Parameters

{params_table}

## Observables & Tolerance

{obs_tol or '*See source spec.*'}

## Variations

{variations or '*See source spec.*'}

## Run

```python
{code}
```

## Full Spec

`papers/universal_simulation/benchmark/{domain}/spec.md`
"""
    return spec


def run():
    if not PAPER_SIM.exists():
        print(f'WARNING: {PAPER_SIM} not found — skipping usage 4')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for domain_dir in sorted(PAPER_SIM.iterdir()):
        if not domain_dir.is_dir():
            continue
        spec_src = domain_dir / 'spec.md'
        if not spec_src.exists():
            continue

        src_text = spec_src.read_text(encoding='utf-8', errors='replace')
        domain = domain_dir.name
        out_domain = OUT_DIR / domain
        out_domain.mkdir(exist_ok=True)

        spec = make_minimal_spec(domain, src_text)
        (out_domain / 'spec.md').write_text(spec, encoding='utf-8')
        total += 1

    print(f'Usage 4: generated {total} simulation specs in {OUT_DIR}')


if __name__ == '__main__':
    run()
