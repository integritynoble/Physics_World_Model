#!/usr/bin/env python3
"""Build arXiv submission for PWM Flagship paper.

Combines main.tex (with preamble.tex, methods.tex inlined) + supplementary.tex
into a single main.tex, copies figures, and creates a zip.

Adds Prof. David Brady as co-author.
"""

import os
import sys
import re
import shutil
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "papers" / "pwm_flagship"
ARXIV_DIR = PAPER_DIR / "arxiv"


def read_file(path):
    return path.read_text(encoding='utf-8')


def main():
    print("=" * 60)
    print("  Building PWM Flagship arXiv submission package")
    print("=" * 60)

    # Clean and create arxiv directory
    if ARXIV_DIR.exists():
        shutil.rmtree(ARXIV_DIR)
    ARXIV_DIR.mkdir()
    (ARXIV_DIR / "figures").mkdir()

    # Read all source files
    preamble = read_file(PAPER_DIR / "preamble.tex")
    main_tex = read_file(PAPER_DIR / "main.tex")
    methods_tex = read_file(PAPER_DIR / "methods.tex")
    supp_tex = read_file(PAPER_DIR / "supplementary.tex")
    main_bbl = read_file(PAPER_DIR / "main.bbl")
    supp_bbl = read_file(PAPER_DIR / "supplementary.bbl")

    # ================================================================
    # 1. Inline \input{preamble} and \input{methods} into main.tex
    # ================================================================
    main_tex = main_tex.replace(r'\input{preamble}', preamble)
    main_tex = main_tex.replace(r'\input{methods}', methods_tex)

    # ================================================================
    # 2. Add packages from supplementary that main doesn't have
    # ================================================================
    # supplementary uses longtable and array — add them
    if r'\usepackage{longtable' not in main_tex:
        main_tex = main_tex.replace(
            r'\usepackage{multirow}',
            r'\usepackage{multirow}' + '\n' + r'\usepackage{longtable}'
        )

    # Add proposition theorem style from supplementary
    if r'\newtheorem{proposition}' not in main_tex:
        main_tex = main_tex.replace(
            r'\newtheorem{definition}{Definition}',
            r'\newtheorem{definition}{Definition}' + '\n'
            + r'\newtheorem{proposition}[theorem]{Proposition}'
        )

    # ================================================================
    # 3. Add Prof. David Brady as co-author
    # ================================================================
    main_tex = main_tex.replace(
        r"""\author{
Chengshuai Yang$^{1,*}$
\and
Xin Yuan$^{2}$
}""",
        r"""\author{
Chengshuai Yang$^{1,*}$
\and
David J. Brady$^{2}$
\and
Xin Yuan$^{3}$
}"""
    )

    main_tex = main_tex.replace(
        r"""$^1$NextGen PlatformAI C Corp, USA.\quad
$^2$School of Engineering, Westlake University, Hangzhou, China.\\[2pt]""",
        r"""$^1$NextGen PlatformAI C Corp, USA.\quad
$^2$School of Electrical and Computer Engineering, University of Arizona, Tucson, AZ, USA.\quad
$^3$School of Engineering, Westlake University, Hangzhou, China.\\[2pt]"""
    )

    # Update supplementary author line too
    supp_tex = supp_tex.replace(
        r'\author{Chengshuai Yang and Xin Yuan}',
        r'\author{Chengshuai Yang, David J. Brady, and Xin Yuan}'
    )

    # ================================================================
    # 4. Replace \bibliography{pwm_flagship} with inlined .bbl content
    # ================================================================
    main_tex = main_tex.replace(
        r"""\bibliographystyle{unsrtnat}
\bibliography{pwm_flagship}""",
        main_bbl
    )

    # ================================================================
    # 5. Extract supplementary body
    # ================================================================
    supp_body_match = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        supp_tex, re.DOTALL
    )
    if not supp_body_match:
        print("ERROR: Could not extract supplementary body")
        sys.exit(1)
    supp_body = supp_body_match.group(1)

    # Remove \maketitle from supplementary body (we add our own header)
    supp_body = supp_body.replace(r'\maketitle', '')

    # Remove the counter resets from supp body (we add them ourselves)
    supp_body = re.sub(
        r'\\renewcommand\{\\thetable\}\{S\\arabic\{table\}\}\s*'
        r'\\renewcommand\{\\thefigure\}\{S\\arabic\{figure\}\}\s*'
        r'\\renewcommand\{\\theequation\}\{S\\arabic\{equation\}\}\s*'
        r'\\setcounter\{table\}\{0\}\s*'
        r'\\setcounter\{figure\}\{0\}\s*'
        r'\\setcounter\{equation\}\{0\}',
        '',
        supp_body
    )

    # Remove \tableofcontents and \newpage at start
    supp_body = supp_body.replace(r'\tableofcontents', '')

    # Remove the supplementary's own \bibliography
    supp_body = re.sub(
        r'\\bibliographystyle\{unsrtnat\}\s*\\bibliography\{pwm_flagship\}',
        '',
        supp_body
    )

    # ================================================================
    # 6. Remove \end{document} from main and append supplementary
    # ================================================================
    main_tex = main_tex.rstrip()
    if main_tex.endswith(r'\end{document}'):
        main_tex = main_tex[:-len(r'\end{document}')]

    supp_transition = r"""

% =====================================================================
% SUPPLEMENTARY INFORMATION
% =====================================================================
\clearpage

% --- Reset section/figure/table/equation numbering for Supplementary ---
\setcounter{section}{0}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{S\arabic{section}.\arabic{subsection}}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{table}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{figure}{0}
\renewcommand{\theequation}{S\arabic{equation}}
\setcounter{equation}{0}

\begin{center}
{\Large\bfseries Supplementary Information}\\[12pt]
{\large Eleven Primitives and Three Gates:\\
The Universal Structure of Computational Imaging}\\[8pt]
{\normalsize Chengshuai Yang, David J. Brady, and Xin Yuan}
\end{center}

\vspace{12pt}
\renewcommand{\contentsname}{Supplementary Contents}
\tableofcontents
\newpage
"""

    # Combine
    combined = main_tex + supp_transition + supp_body + '\n\n\\end{document}\n'

    # ================================================================
    # 7. Write combined file
    # ================================================================
    combined_path = ARXIV_DIR / "main.tex"
    combined_path.write_text(combined, encoding='utf-8')
    print(f"  Combined tex: {combined_path}")

    # ================================================================
    # 8. Copy figures
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
    # 9. Create zip
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
