#!/usr/bin/env python3
"""Replace non-ASCII characters in generate_dataset.py files for Windows cp1252 compatibility."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "benchmark"

REPLACEMENTS = {
    '\u2500': '-',   # box-drawing horizontal
    '\u2502': '|',   # box-drawing vertical
    '\u251c': '|',   # box-drawing tee right
    '\u2514': '`',   # box-drawing corner
    '\u2510': '+',   # box-drawing corner
    '\u2518': '+',   # box-drawing corner
    '\u250c': '+',   # box-drawing corner
    '\u2524': '|',   # box-drawing tee left
    '\u252c': '+',   # box-drawing tee down
    '\u2534': '+',   # box-drawing tee up
    '\u253c': '+',   # box-drawing cross
    '\u2192': '->',  # right arrow
    '\u2190': '<-',  # left arrow
    '\u2014': '-',   # em dash
    '\u2013': '-',   # en dash
    '\u2018': "'",   # left single quote
    '\u2019': "'",   # right single quote
    '\u201c': '"',   # left double quote
    '\u201d': '"',   # right double quote
    '\u2026': '...', # ellipsis
    '\u00b2': '^2',  # superscript 2
    '\u00b3': '^3',  # superscript 3
    '\u00d7': 'x',   # multiplication sign
    '\u2248': '~=',  # approximately equal
    '\u2264': '<=',  # less than or equal
    '\u2265': '>=',  # greater than or equal
    '\u0394': 'Delta',  # Greek Delta
    '\u03b1': 'alpha',  # Greek alpha
    '\u03b2': 'beta',   # Greek beta
    '\u03b3': 'gamma',  # Greek gamma
    '\u03b4': 'delta',  # Greek delta
    '\u03b8': 'theta',  # Greek theta
    '\u03bb': 'lambda', # Greek lambda
    '\u03bc': 'mu',     # Greek mu
    '\u03c0': 'pi',     # Greek pi
    '\u03c3': 'sigma',  # Greek sigma
    '\u03c6': 'phi',    # Greek phi
    '\u03c9': 'omega',  # Greek omega
    '\u2713': 'OK',     # check mark
    '\u2717': 'x',      # cross mark
    '\u2022': '*',      # bullet
    '\u00b1': '+/-',    # plus-minus
}

MODALITIES = [
    "diffusion_mri", "endoscopy", "mammography", "pet",
    "two_photon", "ultrasound",
]

for mod in MODALITIES:
    fpath = ROOT / mod / "generate_dataset.py"
    if not fpath.exists():
        continue

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)

    # Check for remaining non-ASCII
    remaining = []
    for i, ch in enumerate(content):
        if ord(ch) > 127:
            remaining.append(f"  U+{ord(ch):04X} at pos {i}")

    if content != original:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed: {mod}")
    else:
        print(f"No changes: {mod}")

    if remaining:
        print(f"  WARNING: {len(remaining)} remaining non-ASCII chars:")
        for r in remaining[:5]:
            print(r)

print("\nDone!")
