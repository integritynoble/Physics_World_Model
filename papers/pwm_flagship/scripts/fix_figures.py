import re

content = open('main.tex', encoding='utf-8').read()

def extract_figure(content, filename):
    """Extract a figure block containing the given filename."""
    pattern = r'\n\\begin\{figure\}\[!ht\]\n\\centering\n\\includegraphics\[width=\\textwidth\]\{' + re.escape(filename) + r'\}.*?\\end\{figure\}'
    m = re.search(pattern, content, re.DOTALL)
    if not m:
        raise ValueError(f"Figure block not found for {filename}")
    return m.group(0), m.start(), m.end()

# ── FIG 2 ──────────────────────────────────────────────────────────────────
fig2_block, f2s, f2e = extract_figure(content, 'figures/fig2_operatorgraph.pdf')
content = content[:f2s] + content[f2e:]  # remove from old position

# Insert after proof sketch that references fig:operatorgraph
anchor2 = r'alongside the four-tier fidelity ladder are shown in \cref{fig:operatorgraph}. \hfill$\square$'
idx = content.index(anchor2)
insert_at = idx + len(anchor2)
content = content[:insert_at] + '\n' + fig2_block + content[insert_at:]
print("Fig 2 moved OK")

# ── FIG 3 ──────────────────────────────────────────────────────────────────
fig3_block, f3s, f3e = extract_figure(content, 'figures/fig3_triad.pdf')
content = content[:f3s] + content[f3e:]  # remove from old position

# Insert after Theorem 2 proof, before "Lifecycle prediction" paragraph
anchor3 = r'\noindent\textit{Proof.} See Supplementary Note~1, Proposition~2. \hfill$\square$'
idx = content.index(anchor3)
insert_at = idx + len(anchor3)
content = content[:insert_at] + '\n' + fig3_block + content[insert_at:]
print("Fig 3 moved OK")

# ── FIG 5 ──────────────────────────────────────────────────────────────────
fig5_block, f5s, f5e = extract_figure(content, 'figures/fig5_hardware.pdf')
content = content[:f5s] + content[f5e:]  # remove from old position

# Insert BEFORE "Simulation-to-hardware gap" paragraph
anchor5 = '\n\\paragraph{Simulation-to-hardware gap.}'
idx = content.index(anchor5)
content = content[:idx] + '\n' + fig5_block + content[idx:]
print("Fig 5 moved OK")

open('main.tex', 'w', encoding='utf-8').write(content)
print("Done — main.tex updated.")
