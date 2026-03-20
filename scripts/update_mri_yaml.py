#!/usr/bin/env python3
"""Update mri.yaml with all 33 solvers from SOLVERS dict."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yaml_path = os.path.join(ROOT, "benchmarks", "configs", "mri.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Fix category_module
data["category_module"] = "medical_mri_kspace"

# Import SOLVERS registry
from algorithm_base.mri.solvers import SOLVERS

# Build solver entries for YAML
solvers_yaml = {}
for key, spec in SOLVERS.items():
    entry = {
        "name": spec["name"],
        "module": spec["module"],
        "function": spec["function"],
        "params": spec.get("cfg_override", {}),
        "gpu": spec.get("gpu", False),
        "reference": spec.get("reference", ""),
    }
    solvers_yaml[key] = entry

data["solvers"] = solvers_yaml

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"Updated {yaml_path} with {len(solvers_yaml)} solvers")
