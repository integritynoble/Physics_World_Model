#!/usr/bin/env python3
"""Usage 4: Scientific Simulation — one minimal spec per domain. ~15 lines."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import SPEC_DIR

PAPER_SIM = SPEC_DIR.parent / 'papers' / 'universal_simulation' / 'benchmark'
OUT_DIR   = SPEC_DIR / '04_simulation'


def first_equation(text: str) -> str:
    """Extract first meaningful equation line from source spec."""
    # Look for lines inside an 'equations:' block or after '## Equations'
    in_eq = False
    for line in text.splitlines():
        if re.search(r'##\s*Equation|^equations:', line, re.IGNORECASE):
            in_eq = True
            continue
        if in_eq:
            s = line.strip().lstrip('#').strip()
            if s and not s.startswith('##') and len(s) > 5:
                return s[:100]
    return ''


def run_code(text: str) -> str:
    """Extract Python code block if present."""
    m = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else ''


def make_spec(domain: str, src_text: str) -> str:
    lines = src_text.strip().splitlines()
    title = next((l.lstrip('# ').strip() for l in lines if l.startswith('# ')), domain)
    eq    = first_equation(src_text)
    code  = run_code(src_text)
    if not code:
        code = (
            f"import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')\n"
            f"from pathlib import Path\n"
            f"public_dir = Path('papers/universal_simulation/benchmark/{domain}/public/')\n"
            f"# implement simulation according to spec below"
        )
    eq_line = f'`{eq}`\n' if eq else ''
    return f"""\
# {title}

{eq_line}**Source**: `papers/universal_simulation/benchmark/{domain}/spec.md`

```python
{code}
```
"""


def run():
    if not PAPER_SIM.exists():
        print(f'WARNING: {PAPER_SIM} not found — skipping usage 4')
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for domain_dir in sorted(PAPER_SIM.iterdir()):
        if not domain_dir.is_dir(): continue
        spec_src = domain_dir / 'spec.md'
        if not spec_src.exists(): continue
        domain   = domain_dir.name
        out_dir  = OUT_DIR / domain
        out_dir.mkdir(exist_ok=True)
        src_text = spec_src.read_text(encoding='utf-8', errors='replace')
        (out_dir / 'spec.md').write_text(make_spec(domain, src_text), encoding='utf-8')
        total += 1
    print(f'Usage 4: generated {total} simulation specs in {OUT_DIR}')


if __name__ == '__main__':
    run()
