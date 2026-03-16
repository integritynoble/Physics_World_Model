#!/usr/bin/env python3
"""Fix algorithm names in per_algorithm_verification.json by extracting from original git version."""
import json, re, subprocess, sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
PER_ALGO = ROOT / "benchmark_results" / "per_algorithm_verification.json"

# Get original algorithm_state.md from git
result = subprocess.run(
    ['git', 'show', '4fd50ed8:datasets/benchmark/algorithm_state.md'],
    capture_output=True, text=True, encoding='utf-8', errors='replace',
    cwd=str(ROOT)
)
original_md = result.stdout

# Parse algorithm names from original markdown
# Format: | Rank | Algorithm | Year | Reference | Ref PSNR | ...
# Group by modality (preceded by ### N. Display Name (`modality_id`))
mod_algos = {}
current_mod = None

for line in original_md.split('\n'):
    # Match modality header
    m = re.match(r'###\s+\d+\.\s+.*\(`(\w+)`\)', line)
    if m:
        current_mod = m.group(1)
        mod_algos[current_mod] = []
        continue

    # Match algorithm row
    if current_mod and line.startswith('|') and not line.startswith('|--') and not line.startswith('| Rank'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 4:
            rank = parts[1]
            algo_name = parts[2]
            year = parts[3]
            ref = parts[4] if len(parts) > 4 else ''
            try:
                int(rank)  # Verify it's a data row
                mod_algos[current_mod].append({
                    'algorithm': algo_name,
                    'year': year,
                    'reference': ref,
                })
            except ValueError:
                pass

# Now update per_algorithm_verification.json
with open(PER_ALGO, "r", encoding="utf-8") as f:
    pa = json.load(f)

fixed = 0
for mod_id, algos in pa['modalities'].items():
    orig = mod_algos.get(mod_id, [])
    for i, a in enumerate(algos):
        if a.get('algorithm', '?') == '?' and i < len(orig):
            a['algorithm'] = orig[i]['algorithm']
            if not a.get('year') or a['year'] == '':
                a['year'] = orig[i].get('year', '')
            if not a.get('reference') or a['reference'] == '':
                a['reference'] = orig[i].get('reference', '')
            fixed += 1

with open(PER_ALGO, "w", encoding="utf-8") as f:
    json.dump(pa, f, indent=2, ensure_ascii=False)

print(f"Fixed {fixed} algorithm names")

# Verify
q_count = sum(1 for algos in pa['modalities'].values() for a in algos if a.get('algorithm','?') == '?')
print(f"Remaining '?' algorithms: {q_count}")
