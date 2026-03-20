"""Update __init__.py files for all 8 modalities after solver expansion."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODALITIES = {
    "spc": ("Single-Pixel Camera (SPC)", "spc"),
    "lensless": ("Lensless (Diffuser Camera) Imaging", "lensless"),
    "holography": ("Digital Holographic Microscopy", "holography"),
    "ptychography": ("Ptychographic Imaging", "ptychography"),
    "cbct": ("Cone-Beam Computed Tomography (CBCT)", "cbct"),
    "ultrasound": ("Ultrasound B-mode Imaging", "ultrasound"),
    "cryo_em": ("Cryo-EM Single Particle Analysis", "cryo_em"),
    "widefield": ("Widefield Fluorescence Microscopy", "widefield"),
}

TEMPLATE = '''"""{display_name} ({mod_id}) — algorithm solvers."""
from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME
from .solvers import run_traditional_cpu
from .solvers import run_best_quality
from .solvers import run_famous_dl
from .solvers import run_small_gpu

__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME"]
'''

for mod_id, (display_name, _) in MODALITIES.items():
    init_path = os.path.join(ROOT, "algorithm_base", mod_id, "__init__.py")
    content = TEMPLATE.format(display_name=display_name, mod_id=mod_id)
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Updated: algorithm_base/{mod_id}/__init__.py")

print("\nDone. All 8 __init__.py files updated.")
