#!/usr/bin/env python3
"""Build algorithm_base/ — per-modality algorithm catalog with direct-runnable entries.

Structure:
  algorithm_base/
    __init__.py               # Package init + registry
    _registry.py              # Central registry: get_solver(), list_solvers(), etc.
    cassi/
      __init__.py             # from .solvers import *
      solvers.py              # All solver functions for cassi
      README.md               # Modality info + algorithm table
    cacti/
      __init__.py
      solvers.py
      README.md
    mri/
      ...
    (168 modality folders)

Usage:
    # Direct import by modality
    from algorithm_base.cassi import run_solver
    x_hat = run_solver("traditional_cpu", y, operator)

    # Via registry
    from algorithm_base import get_solver, list_solvers
    solver_fn = get_solver("cassi", "traditional_cpu")
    x_hat = solver_fn(y, operator, {})

    # List all
    for name, info in list_solvers("cassi"):
        print(name, info["gpu"])
"""
import json, os, sys, yaml, importlib
from pathlib import Path
import io, warnings
warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
CONFIG_DIR = ROOT / "benchmarks" / "configs"
ALGO_BASE = ROOT / "algorithm_base"
PER_ALGO = ROOT / "benchmark_results" / "per_algorithm_verification.json"
SOLVER_REG = ROOT / "benchmark_results" / "solver_registry.json"

# Ensure PWM5 imports
pwm5_pkg = str(ROOT / "packages" / "pwm_core")
sys.path = [p for p in sys.path if 'pwm_core' not in p or 'PWM5' in p]
sys.path.insert(0, pwm5_pkg)
for k in list(sys.modules):
    if 'pwm_core' in k:
        del sys.modules[k]

def check_importable(module, function):
    try:
        mod = importlib.import_module(module)
        getattr(mod, function)
        return True
    except:
        return False

def main():
    print("="*70)
    print("BUILD ALGORITHM BASE")
    print("="*70)

    # Load configs
    with open(PER_ALGO, "r", encoding="utf-8") as f:
        pa = json.load(f)

    # Create base dir
    ALGO_BASE.mkdir(exist_ok=True)

    # Load all YAML configs
    configs = {}
    for fn in sorted(os.listdir(str(CONFIG_DIR))):
        if not fn.endswith(".yaml") or fn == "_template.yaml":
            continue
        with open(CONFIG_DIR / fn, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mid = cfg.get("modality_id", fn.replace(".yaml", ""))
        configs[mid] = cfg

    total_mods = 0
    total_solvers = 0

    # Build per-modality folders
    for mid in sorted(configs.keys()):
        cfg = configs[mid]
        display = cfg.get("display_name", mid)
        category = cfg.get("category", "Other")
        solvers = cfg.get("solvers", {})
        if not solvers or not isinstance(solvers, dict):
            continue

        mod_dir = ALGO_BASE / mid
        mod_dir.mkdir(exist_ok=True)
        total_mods += 1

        # Collect solver info
        solver_entries = []
        for sk, spec in solvers.items():
            if not isinstance(spec, dict):
                continue
            module = spec.get("module", "")
            function = spec.get("function", "")
            gpu = spec.get("gpu", False)
            name = spec.get("name", sk)
            reference = spec.get("reference", "")
            params = spec.get("params", {})
            importable = check_importable(module, function)

            solver_entries.append({
                "key": sk,
                "name": name,
                "module": module,
                "function": function,
                "gpu": gpu,
                "reference": reference,
                "params": params,
                "importable": importable,
            })
            total_solvers += 1

        # Write solvers.py
        write_solvers_py(mod_dir / "solvers.py", mid, display, solver_entries)

        # Write __init__.py
        write_init_py(mod_dir / "__init__.py", mid, display, solver_entries)

        # Write README.md
        algos = pa.get("modalities", {}).get(mid, [])
        write_readme(mod_dir / "README.md", mid, display, category, solver_entries, algos)

    # Write top-level __init__.py
    write_top_init(ALGO_BASE / "__init__.py", sorted(configs.keys()))

    # Write _registry.py
    write_registry(ALGO_BASE / "_registry.py", sorted(configs.keys()))

    # Write top-level README.md
    write_top_readme(ALGO_BASE / "README.md", configs, pa)

    print(f"\nBuilt algorithm_base/")
    print(f"  {total_mods} modality folders")
    print(f"  {total_solvers} solver entries")
    print(f"  Path: {ALGO_BASE}")


def write_solvers_py(path, mid, display, entries):
    """Write the per-modality solvers.py with all solver functions."""
    lines = [
        f'"""Solvers for {display} ({mid}).',
        f'',
        f'Each function wraps a solver from pwm_core.recon.*.',
        f'All follow the standard interface: fn(y, operator, cfg) -> x_hat',
        f'"""',
        f'',
        f'from __future__ import annotations',
        f'import importlib',
        f'import numpy as np',
        f'from typing import Any, Dict, Optional',
        f'',
        f'MODALITY_ID = "{mid}"',
        f'DISPLAY_NAME = "{display}"',
        f'',
        f'',
        f'# Solver registry for {mid}',
        f'SOLVERS = {{',
    ]
    for e in entries:
        gpu_str = "True" if e["gpu"] else "False"
        lines.append(f'    "{e["key"]}": {{')
        lines.append(f'        "name": "{e["name"]}",')
        lines.append(f'        "module": "{e["module"]}",')
        lines.append(f'        "function": "{e["function"]}",')
        lines.append(f'        "gpu": {gpu_str},')
        lines.append(f'        "reference": "{e["reference"]}",')
        lines.append(f'    }},')
    lines.append(f'}}')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def _load_fn(solver_key: str):')
    lines.append(f'    """Dynamically load solver function."""')
    lines.append(f'    spec = SOLVERS[solver_key]')
    lines.append(f'    mod = importlib.import_module(spec["module"])')
    lines.append(f'    return getattr(mod, spec["function"])')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,')
    lines.append(f'               cfg: Optional[Dict] = None) -> np.ndarray:')
    lines.append(f'    """Run a solver by key.')
    lines.append(f'')
    lines.append(f'    Args:')
    lines.append(f'        solver_key: One of {[e["key"] for e in entries]}')
    lines.append(f'        y: Measurement data (float32)')
    lines.append(f'        operator: Forward operator')
    lines.append(f'        cfg: Hyperparameters (optional)')
    lines.append(f'')
    lines.append(f'    Returns:')
    lines.append(f'        x_hat: Reconstructed signal')
    lines.append(f'    """')
    lines.append(f'    if solver_key not in SOLVERS:')
    lines.append(f'        raise ValueError(f"Unknown solver {{solver_key}}. Available: {{list(SOLVERS.keys())}}")')
    lines.append(f'    fn = _load_fn(solver_key)')
    lines.append(f'    result = fn(y.astype(np.float32), operator, cfg or {{}})')
    lines.append(f'    if isinstance(result, tuple):')
    lines.append(f'        return np.asarray(result[0], dtype=np.float32)')
    lines.append(f'    return np.asarray(result, dtype=np.float32)')
    lines.append(f'')
    lines.append(f'')
    lines.append(f'def list_solvers():')
    lines.append(f'    """List all available solvers for {mid}."""')
    lines.append(f'    return [(k, v) for k, v in SOLVERS.items()]')
    lines.append(f'')

    # Add per-solver convenience functions
    for e in entries:
        safe_key = e["key"].replace("-", "_")
        lines.append(f'')
        lines.append(f'def run_{safe_key}(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:')
        lines.append(f'    """{e["name"]}. {"GPU required." if e["gpu"] else "CPU only."}')
        if e["reference"]:
            lines.append(f'    Reference: {e["reference"]}')
        lines.append(f'    """')
        lines.append(f'    return run_solver("{e["key"]}", y, operator, cfg)')

    lines.append(f'')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_init_py(path, mid, display, entries):
    """Write per-modality __init__.py."""
    lines = [
        f'"""{display} ({mid}) — algorithm solvers."""',
        f'from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME',
    ]
    for e in entries:
        safe_key = e["key"].replace("-", "_")
        lines.append(f'from .solvers import run_{safe_key}')
    lines.append(f'')
    lines.append(f'__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME"]')
    lines.append(f'')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_readme(path, mid, display, category, entries, algos):
    """Write per-modality README.md."""
    lines = [
        f'# {display} (`{mid}`)',
        f'',
        f'Category: {category}',
        f'',
        f'## Solvers',
        f'',
        f'| Key | Name | Module | GPU | Reference |',
        f'|-----|------|--------|-----|-----------|',
    ]
    for e in entries:
        gpu = "Yes" if e["gpu"] else "No"
        lines.append(f'| `{e["key"]}` | {e["name"]} | `{e["module"]}.{e["function"]}` | {gpu} | {e["reference"]} |')

    lines.append(f'')
    lines.append(f'## Usage')
    lines.append(f'')
    lines.append(f'```python')
    lines.append(f'# Import and run')
    lines.append(f'from algorithm_base.{mid} import run_solver')
    lines.append(f'x_hat = run_solver("{entries[0]["key"]}", y, operator)')
    lines.append(f'')
    lines.append(f'# Or use specific function')
    safe0 = entries[0]["key"].replace("-", "_")
    lines.append(f'from algorithm_base.{mid} import run_{safe0}')
    lines.append(f'x_hat = run_{safe0}(y, operator)')
    lines.append(f'```')
    lines.append(f'')

    # Algorithm table
    if algos:
        lines.append(f'## Algorithm Leaderboard')
        lines.append(f'')
        lines.append(f'| Algorithm | Year | Ref PSNR | PWM PSNR | Status |')
        lines.append(f'|-----------|------|----------|----------|--------|')
        for a in algos:
            name = a.get("algorithm") or a.get("name", "?")
            year = a.get("year", "")
            rp = f'{a["ref_psnr"]:.1f}' if a.get("ref_psnr") else "—"
            pp = f'{a["pwm_psnr"]:.1f}' if a.get("pwm_psnr") else "—"
            status = a.get("status", "?")
            lines.append(f'| {name} | {year} | {rp} | {pp} | {status} |')
    lines.append(f'')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_top_init(path, modality_ids):
    """Write top-level algorithm_base/__init__.py."""
    lines = [
        '"""PWM Algorithm Base — 168 modalities, 541 solvers.',
        '',
        'Usage:',
        '    # Import by modality',
        '    from algorithm_base.cassi import run_solver',
        '    x_hat = run_solver("traditional_cpu", y, operator)',
        '',
        '    # Via registry',
        '    from algorithm_base import get_solver, list_solvers, list_modalities',
        '    solver_fn = get_solver("cassi", "traditional_cpu")',
        '    x_hat = solver_fn(y, operator, {})',
        '"""',
        'from ._registry import get_solver, list_solvers, list_modalities, run_solver',
        '',
        f'MODALITIES = {modality_ids}',
        '',
        '__all__ = ["get_solver", "list_solvers", "list_modalities", "run_solver", "MODALITIES"]',
        '',
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_registry(path, modality_ids):
    """Write algorithm_base/_registry.py — central dispatch."""
    lines = [
        '"""Central algorithm registry for PWM Algorithm Base.',
        '',
        'Provides unified access to all 168 modalities and their solvers.',
        '"""',
        'from __future__ import annotations',
        '',
        'import importlib',
        'import numpy as np',
        'from typing import Any, Callable, Dict, List, Optional, Tuple',
        '',
        '',
        'def list_modalities() -> List[str]:',
        '    """List all available modality IDs."""',
        '    from pathlib import Path',
        '    base = Path(__file__).parent',
        '    return sorted([',
        '        d.name for d in base.iterdir()',
        '        if d.is_dir() and not d.name.startswith("_") and (d / "solvers.py").exists()',
        '    ])',
        '',
        '',
        'def list_solvers(modality_id: str) -> List[Tuple[str, Dict[str, Any]]]:',
        '    """List all solvers for a modality.',
        '',
        '    Returns:',
        '        List of (solver_key, solver_info) tuples.',
        '    """',
        '    mod = importlib.import_module(f"algorithm_base.{modality_id}.solvers")',
        '    return mod.list_solvers()',
        '',
        '',
        'def get_solver(modality_id: str, solver_key: str) -> Callable:',
        '    """Get a solver function by modality and key.',
        '',
        '    Returns:',
        '        Callable with signature fn(y, operator, cfg) -> x_hat',
        '    """',
        '    mod = importlib.import_module(f"algorithm_base.{modality_id}.solvers")',
        '    return mod._load_fn(solver_key)',
        '',
        '',
        'def run_solver(',
        '    modality_id: str,',
        '    solver_key: str,',
        '    y: np.ndarray,',
        '    operator: Any = None,',
        '    cfg: Optional[Dict] = None,',
        ') -> np.ndarray:',
        '    """Run a solver directly.',
        '',
        '    Args:',
        '        modality_id: e.g. "cassi", "mri", "ct"',
        '        solver_key: e.g. "traditional_cpu", "best_quality"',
        '        y: Measurement data',
        '        operator: Forward operator',
        '        cfg: Hyperparameters',
        '',
        '    Returns:',
        '        x_hat: Reconstructed signal',
        '',
        '    Example:',
        '        >>> from algorithm_base import run_solver',
        '        >>> x_hat = run_solver("cassi", "traditional_cpu", y, op)',
        '    """',
        '    mod = importlib.import_module(f"algorithm_base.{modality_id}.solvers")',
        '    return mod.run_solver(solver_key, y, operator, cfg)',
        '',
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_top_readme(path, configs, pa):
    """Write algorithm_base/README.md."""
    lines = [
        '# PWM Algorithm Base',
        '',
        '168 modalities, 541 solvers — organized for server, paper, and CLI usage.',
        '',
        '## Quick Start',
        '',
        '```python',
        '# 1. Import by modality name',
        'from algorithm_base.cassi import run_solver',
        'x_hat = run_solver("traditional_cpu", y, operator)',
        '',
        '# 2. Use specific solver function',
        'from algorithm_base.cacti import run_traditional_cpu',
        'x_hat = run_traditional_cpu(y, operator)',
        '',
        '# 3. Via central registry',
        'from algorithm_base import get_solver, list_solvers, list_modalities',
        '',
        '# List all 168 modalities',
        'for mod in list_modalities():',
        '    print(mod)',
        '',
        '# List solvers for a modality',
        'for key, info in list_solvers("mri"):',
        '    print(f"{key}: {info[\'name\']} (GPU={info[\'gpu\']})")',
        '',
        '# Get and run a solver',
        'solver_fn = get_solver("ct", "traditional_cpu")',
        'x_hat = solver_fn(sinogram, radon_op, {"iterations": 50})',
        '',
        '# Or run directly',
        'from algorithm_base import run_solver',
        'x_hat = run_solver("mri", "best_quality", kspace, mri_op)',
        '```',
        '',
        '## Three Use Cases',
        '',
        '### 1. Main Server (no GPU)',
        '```python',
        'from algorithm_base import run_solver, list_solvers',
        '',
        '# CPU solvers run directly',
        'x_hat = run_solver("cassi", "traditional_cpu", y, op)',
        '',
        '# For GPU solvers, use SolverDispatcher with Modal',
        'from pwm_core.recon.solver_dispatch import SolverDispatcher',
        'dispatcher = SolverDispatcher(use_modal=True)',
        'x_hat, meta = dispatcher.run("cacti", "hisvit13", y, op)',
        '```',
        '',
        '### 2. Paper Benchmarks',
        '```python',
        'from algorithm_base import list_modalities, list_solvers, run_solver',
        '',
        'for modality in list_modalities():',
        '    for key, info in list_solvers(modality):',
        '        x_hat = run_solver(modality, key, y_dict[modality], op_dict[modality])',
        '        # compute PSNR, SSIM for paper table',
        '```',
        '',
        '### 3. PWM CLI',
        '```bash',
        'pwm evaluate --method traditional_cpu --modality cassi --track correct',
        '```',
        '',
        '## Adding New Algorithms',
        '',
        '1. Create solver function in `packages/pwm_core/pwm_core/recon/my_solver.py`:',
        '```python',
        'def run_my_solver(y, operator, cfg):',
        '    # your reconstruction code',
        '    return x_hat',
        '```',
        '',
        '2. Register in modality YAML (`benchmarks/configs/{modality}.yaml`):',
        '```yaml',
        'solvers:',
        '  my_solver:',
        '    name: "My Solver"',
        '    module: "pwm_core.recon.my_solver"',
        '    function: "run_my_solver"',
        '    params:',
        '      iterations: 100',
        '    gpu: false',
        '    reference: "Author et al., 2026"',
        '```',
        '',
        '3. Rebuild algorithm base:',
        '```bash',
        'python scripts/build_algorithm_base.py',
        '```',
        '',
        '## Directory Structure',
        '',
        '```',
        'algorithm_base/',
        '  __init__.py          # from ._registry import get_solver, ...',
        '  _registry.py         # Central dispatch: get_solver(), list_solvers(), ...',
        '  README.md            # This file',
        '  cassi/',
        '    __init__.py        # from .solvers import run_solver, run_traditional_cpu, ...',
        '    solvers.py         # SOLVERS dict + run_solver() + per-key functions',
        '    README.md          # Modality info + algorithm leaderboard',
        '  cacti/',
        '    ...',
        '  mri/',
        '    ...',
        '  (168 modality folders)',
        '```',
        '',
    ]

    # Modality index
    lines.append('## Modality Index')
    lines.append('')
    lines.append('| Modality | Display Name | Category | Solvers |')
    lines.append('|----------|-------------|----------|---------|')
    for mid in sorted(configs.keys()):
        cfg = configs[mid]
        display = cfg.get("display_name", mid)
        cat = cfg.get("category", "Other")
        n = len(cfg.get("solvers", {})) if isinstance(cfg.get("solvers"), dict) else 0
        lines.append(f'| [`{mid}`](./{mid}/) | {display} | {cat} | {n} |')

    lines.append('')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
