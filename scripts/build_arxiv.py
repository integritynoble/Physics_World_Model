#!/usr/bin/env python3
"""Build arXiv submission folder: combine paper.tex + supplementary.tex into one file,
copy figures, and create a zip ready for upload."""

import os
import sys
import re
import shutil
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "papers" / "universal_simulation"
ARXIV_DIR = PAPER_DIR / "arxiv"

def read_file(path):
    return path.read_text(encoding='utf-8')

def main():
    print("=" * 60)
    print("  Building arXiv submission package")
    print("=" * 60)

    # Clean and create arxiv directory
    if ARXIV_DIR.exists():
        shutil.rmtree(ARXIV_DIR)
    ARXIV_DIR.mkdir()
    (ARXIV_DIR / "figures").mkdir()

    # Read both files
    main_tex = read_file(PAPER_DIR / "paper.tex")
    supp_tex = read_file(PAPER_DIR / "supplementary.tex")

    # ================================================================
    # 1. Extract supplementary body (between \begin{document} and \end{document})
    # ================================================================
    supp_body_match = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        supp_tex, re.DOTALL
    )
    if not supp_body_match:
        print("ERROR: Could not extract supplementary body")
        sys.exit(1)
    supp_body = supp_body_match.group(1)

    # Remove the title block from supplementary (we'll add our own transition)
    # Remove everything up to first \section
    supp_body = re.sub(
        r'^\s*\\begin\{center\}.*?\\end\{center\}\s*\\vspace\{.*?\}\s*\\tableofcontents\s*\\newpage',
        '',
        supp_body,
        flags=re.DOTALL
    )

    # ================================================================
    # 2. Modify main paper preamble: add missing packages and commands
    # ================================================================
    # Add longtable package (used in supplementary)
    main_tex = main_tex.replace(
        r'\usepackage{multirow}',
        r'\usepackage{multirow}' + '\n' + r'\usepackage{longtable}'
    )

    # Add \Z command from supplementary
    main_tex = main_tex.replace(
        r'\newcommand{\Spec}{\mathcal{S}}',
        r'\newcommand{\Z}{\mathbb{Z}}' + '\n' + r'\newcommand{\Spec}{\mathcal{S}}'
    )

    # Add theorem and lemma environments (supplementary uses them)
    main_tex = main_tex.replace(
        r'\newtheorem{remark}[definition]{Remark}',
        r'\newtheorem{remark}[definition]{Remark}' + '\n'
        + r'\newtheorem{theorem}[definition]{Theorem}' + '\n'
        + r'\newtheorem{lemma}[definition]{Lemma}'
    )

    # ================================================================
    # 3. Remove \end{document} from main and append supplementary
    # ================================================================
    main_tex = main_tex.rstrip()
    if main_tex.endswith(r'\end{document}'):
        main_tex = main_tex[:-len(r'\end{document}')]

    # Build the supplementary transition
    supp_transition = r"""

% =====================================================================
% SUPPLEMENTARY INFORMATION
% =====================================================================
\clearpage

% --- Reset section numbering for Supplementary ---
\setcounter{section}{0}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{S\arabic{section}.\arabic{subsection}}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{table}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{figure}{0}
\renewcommand{\theequation}{S\arabic{equation}}
\setcounter{equation}{0}
\renewcommand{\theproposition}{S\arabic{proposition}}

% --- Supplementary header ---
\fancyhead[L]{\small Supplementary Information}
\fancyhead[R]{\small Yang (2026)}
\renewcommand{\headrulewidth}{0.4pt}

\begin{center}
{\Large\bfseries Supplementary Information}\\[12pt]
{\large A Judge Agent Closes the Reliability Gap in AI-Generated Scientific Simulation}\\[8pt]
{\normalsize Chengshuai Yang}\\[4pt]
{\small NextGen PlatformAI C Corp, USA}
\end{center}

\vspace{12pt}
\renewcommand{\contentsname}{Supplementary Contents}
\tableofcontents
\newpage
"""

    # Combine
    combined = main_tex + supp_transition + supp_body + '\n\n\\end{document}\n'

    # ================================================================
    # 4. Fix cross-reference issues in supplementary body
    # ================================================================
    # The supplementary uses \ref{prop:bound} which is defined in itself
    # This should work fine since it's now in the same document

    # ================================================================
    # 5. Write combined file
    # ================================================================
    combined_path = ARXIV_DIR / "main.tex"
    combined_path.write_text(combined, encoding='utf-8')
    print(f"  Combined tex: {combined_path}")

    # ================================================================
    # 6. Copy figures
    # ================================================================
    fig_src = PAPER_DIR / "figures"
    fig_dst = ARXIV_DIR / "figures"
    fig_count = 0
    for f in fig_src.iterdir():
        if f.suffix.lower() in ('.png', '.pdf', '.jpg', '.jpeg', '.eps'):
            shutil.copy2(f, fig_dst / f.name)
            fig_count += 1
            print(f"  Copied: figures/{f.name}")

    # ================================================================
    # 7. Create zip
    # ================================================================
    zip_path = PAPER_DIR / "arxiv_submission"
    shutil.make_archive(str(zip_path), 'zip', ARXIV_DIR)
    print(f"\n  ZIP: {zip_path}.zip")

    # Summary
    n_files = sum(1 for _ in ARXIV_DIR.rglob('*') if _.is_file())
    total_size = sum(f.stat().st_size for f in ARXIV_DIR.rglob('*') if f.is_file())
    zip_size = (Path(str(zip_path) + '.zip')).stat().st_size

    print(f"\n{'=' * 60}")
    print(f"  arXiv package ready!")
    print(f"  Folder: {ARXIV_DIR}")
    print(f"  Files: {n_files} ({fig_count} figures + main.tex)")
    print(f"  Folder size: {total_size / 1024:.1f} KB")
    print(f"  ZIP size: {zip_size / 1024:.1f} KB")
    print(f"  Upload: {zip_path}.zip")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
