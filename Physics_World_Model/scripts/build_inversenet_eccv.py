#!/usr/bin/env python3
"""Build ECCV resubmission for InverseNet paper.

Splits the 18-page paper into:
  - main.tex (<=14 pages content, excluding references)
  - supplementary.tex (moved content + existing supplementary)

Changes from original:
  1. Move Fig.4 (cassi_real_comparison) to supplementary
  2. Move Fig.5 (cacti_real_comparison) to supplementary
  3. Remove Table 7 (spc_scenario_iv) from main (already in supp)
  4. Condense real hardware + Scenario IV text
  => Saves ~1.5 pages of content
"""

import sys
import re
import shutil
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "papers" / "inversenet"
OUT_DIR = PAPER_DIR / "eccv_resubmission"
SUPP_SRC = PAPER_DIR / "inversenet_supplementary.tex"

def read(p):
    return p.read_text(encoding='utf-8')

def write(p, t):
    p.write_text(t, encoding='utf-8')

def main():
    print("=" * 60)
    print("  Building InverseNet ECCV resubmission")
    print("=" * 60)

    # Clean output
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    (OUT_DIR / "figures").mkdir()

    main_tex = read(PAPER_DIR / "inversenet_paper.tex")
    supp_tex = read(SUPP_SRC)

    # ================================================================
    # 1. Remove Fig.4 (cassi_real_comparison) from main
    #    Move to supplementary with reference in main
    # ================================================================
    # The figure block spans from \begin{figure} with cassi_real_comparison
    # to the \label{fig:cassi_real} \end{figure}
    cassi_fig_block = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/cassi_real_comparison.pdf}
\caption{CASSI real data (Scene~1, band~14): calibrated (top) vs.\ mismatched (bottom) reconstructions for GAP-TV and PnP-HSICNN. The visual differences are subtle, consistent with the modest residual ratios in \cref{tab:cassi_real}---spatial mask shift alone is not the dominant degradation source for CASSI.}
\label{fig:cassi_real}
\end{figure}"""

    main_tex = main_tex.replace(cassi_fig_block,
        "% Visual comparison moved to supplementary (Fig.~S1).\n")

    # ================================================================
    # 2. Remove Fig.5 (cacti_real_comparison) from main
    # ================================================================
    cacti_fig_block = r"""\begin{figure}[t]
\centering
\includegraphics[width=0.9\textwidth]{figures/cacti_real_comparison.pdf}
\caption{CACTI real data: calibrated vs.\ mismatched reconstruction for \textit{duomino} and \textit{hand} scenes. Mismatched masks produce temporal ghosting artifacts, mirroring simulation findings.}
\label{fig:cacti_real}
\end{figure}"""

    main_tex = main_tex.replace(cacti_fig_block,
        "% Visual comparison moved to supplementary (Fig.~S2).\n")

    # ================================================================
    # 3. Remove Table 7 (spc_scenario_iv) from main
    #    Already exists in supplementary Table S4
    # ================================================================
    spc_table_block = r"""\begin{table}[t]
\centering
\caption{SPC blind calibration via TV minimisation (\cref{eq:scenario_iv_tv}). II: mismatched, no calibration. IV: estimated $\hat{\alpha}$ via TV grid search. III: oracle (true $\alpha$). Recovery: (IV$-$II)\,/\,(III$-$II). True $\alpha{=}0.0015$; estimated $\hat{\alpha}{=}0.00125$.}
\label{tab:spc_scenario_iv}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{II} & \textbf{IV} & \textbf{III} & \textbf{Recovery} \\
\midrule
FISTA-TV   & 19.78 & 26.54 & 27.60 & 86\% \\
PnP-DRUNet & 18.34 & 25.39 & 26.01 & 92\% \\
\bottomrule
\end{tabular}
\end{table}"""

    main_tex = main_tex.replace(spc_table_block, "")

    # ================================================================
    # 4. Condense the real hardware text
    # ================================================================

    # Condense CASSI real data paragraph
    old_cassi_real = r"""\paragraph{CASSI real data.}
We use the TSA real dataset~\cite{choi2017kaist}: 5 real scenes captured on a DD-CASSI prototype with a $660 \times 660$ coded aperture mask and 28 spectral bands (450--650\,nm).
Measurements are $660 \times 714$ detector images.
Since \textbf{no ground truth} exists for real CASSI captures (the true spectral cube cannot be independently measured), we use the normalised measurement residual $r = \|\mathbf{y} - \mathbf{\Phi}_{\text{cal}}\hat{\mathbf{x}}\|^2 / \|\mathbf{y}\|^2$, always evaluated with the \emph{calibrated} mask regardless of which mask was used for reconstruction (see supplementary for why this ``cross-residual'' is essential).
We evaluate two conditions: \emph{calibrated} (hardware mask as-is) and \emph{mismatched} (mask shifted by $dx{=}0.5$, $dy{=}0.3$\,pixels).
\Cref{tab:cassi_real} shows the results."""

    new_cassi_real = r"""\paragraph{CASSI real data.}
We use the TSA real dataset~\cite{choi2017kaist}: 5 scenes from a DD-CASSI prototype ($660 \times 660 \times 28$).
Since no ground truth exists, we use the normalised measurement residual $r = \|\mathbf{y} - \mathbf{\Phi}_{\text{cal}}\hat{\mathbf{x}}\|^2 / \|\mathbf{y}\|^2$, evaluated with the calibrated mask (see supplementary for details).
\Cref{tab:cassi_real} compares calibrated vs.\ mismatched ($dx{=}0.5$, $dy{=}0.3$\,px) conditions."""

    main_tex = main_tex.replace(old_cassi_real, new_cassi_real)

    # Condense the CASSI real interpretation paragraph
    old_cassi_interp = r"""Both mask-aware iterative methods (GAP-TV and PnP-HSICNN) show residual increases under mismatch, because they explicitly fit the forward model during reconstruction: when reconstructed with a wrong mask, the result is inconsistent with the true measurement.
GAP-TV shows a modest $1.8\times$ increase, while PnP-HSICNN shows only $1.1\times$---the deep denoiser regularises the reconstruction enough to partially mask the forward-model inconsistency.
This contrasts sharply with CACTI (\cref{sec:cacti_real_data} below), where residuals increase $9.4$--$11.0\times$ under the same spatial shift.

By contrast, our \emph{simulation} experiments (\cref{tab:cassi}) show 3--14\,dB PSNR degradation because they include dispersion perturbation ($a_1$, $\alpha$)---which the real-data experiment does not.
The modest $1.8\times$ residual increase from spatial shift alone (vs.\ $10\times$ for CACTI) validates that \textbf{dispersion mismatch, not spatial shift, is the dominant CASSI degradation source} (\cref{fig:cassi_real})."""

    new_cassi_interp = r"""GAP-TV shows a modest $1.8\times$ residual increase vs.\ PnP-HSICNN's $1.1\times$---the denoiser partially masks forward-model inconsistency.
This contrasts with CACTI's $10.4\times$ (\cref{sec:cacti_real_data}), confirming that \textbf{dispersion mismatch, not spatial shift, is the dominant CASSI degradation source}.
Visual comparisons are in the supplementary material."""

    main_tex = main_tex.replace(old_cassi_interp, new_cassi_interp)

    # Condense CACTI real data paragraph
    old_cacti_real = r"""\paragraph{CACTI real data.}
\label{sec:cacti_real_data}
We evaluate 4 real CACTI scenes from the EfficientSCI dataset (cr$=$10), using the same measurement residual metric (no ground truth).
\Cref{tab:cacti_real} shows the results.
Under mask mismatch, GAP-TV residuals increase $9.4$--$11.0\times$, confirming mismatch severely degrades real data fidelity.
PnP-FFDNet shows increases of $1.3$--$2.8\times$ (mean $2.0\times$), higher than GAP-TV's baseline but moderated by FFDNet's learned denoiser.
Mismatched reconstructions exhibit temporal ghosting (\cref{fig:cacti_real})."""

    new_cacti_real = r"""\paragraph{CACTI real data.}
\label{sec:cacti_real_data}
We evaluate 4 real CACTI scenes (EfficientSCI dataset, cr$=$10) using the measurement residual (\cref{tab:cacti_real}).
GAP-TV residuals increase $9.4$--$11.0\times$ under mismatch, confirming severe real-data degradation; PnP-FFDNet shows $2.0\times$ (moderated by FFDNet's denoiser).
Visual comparisons showing temporal ghosting are in the supplementary."""

    main_tex = main_tex.replace(old_cacti_real, new_cacti_real)

    # Condense Scenario IV SPC text
    old_spc_text = r"""\Cref{tab:spc_scenario_iv} shows the SPC calibration results using TV minimisation (\cref{eq:scenario_iv_tv}).
Grid search over $\alpha \in [0, 0.005]$ with 41 points ($33 \times 33$ blocks, ISTA-Net's learned $\mathbf{\Phi}$) estimates $\hat{\alpha} = 0.00125$ vs.\ true $\alpha = 0.0015$ for both methods, recovering 86--92\% of the oracle bound.
The TV surface exhibits a clear bowl-shaped minimum near the true $\alpha$, confirming that reconstruction sparsity is a viable calibration objective for radiometric mismatch.
PnP-DRUNet achieves higher recovery (92\%) than FISTA-TV (86\%) because the learned denoiser prior produces sharper reconstructions whose TV is more sensitive to residual gain artifacts.
This contrasts with the measurement residual, which is flat across all $\alpha$ values for SPC (the underdetermined system can always self-consistently fit any gain-corrected measurements)."""

    new_spc_text = r"""For SPC, TV minimisation (\cref{eq:scenario_iv_tv}) with grid search over $\alpha \in [0, 0.005]$ estimates $\hat{\alpha} = 0.00125$ vs.\ true $0.0015$, recovering 86--92\% of the oracle bound (see supplementary for per-method results)."""

    main_tex = main_tex.replace(old_spc_text, new_spc_text)

    # Condense Scenario IV criterion comparison
    old_criterion = r"""\paragraph{Criterion comparison.}
The geometric/radiometric distinction reveals that blind calibration requires \emph{matching the objective to the mismatch type}: measurement residual for geometric mismatch (where operator errors create large data infidelities), and reconstruction sparsity for radiometric mismatch (where the operator structure is preserved but measurement values are scaled).
Across all three modalities, Scenario~IV recovers 85--100\% of the oracle bound, demonstrating that blind calibration is practical for all mismatch types studied."""

    new_criterion = r"""\paragraph{Summary.}
Across all modalities, Scenario~IV recovers 85--100\% of the oracle bound---measurement residual for geometric mismatch, reconstruction sparsity for radiometric."""

    main_tex = main_tex.replace(old_criterion, new_criterion)

    # ================================================================
    # 5. Add moved figures to supplementary
    # ================================================================
    # Insert before the first \section in supplementary body
    moved_figs = r"""
% ============================================================================
% MOVED FROM MAIN PAPER
% ============================================================================
\section{Real Hardware Visual Comparisons}
\label{sec:real_visual}

The following figures were moved from the main paper for space.

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/cassi_real_comparison.pdf}
\caption{CASSI real data (Scene~1, band~14): calibrated (top) vs.\ mismatched (bottom) reconstructions for GAP-TV and PnP-HSICNN. The visual differences are subtle, consistent with the modest residual ratios in Table~5 of the main paper---spatial mask shift alone is not the dominant degradation source for CASSI.}
\label{fig:cassi_real_supp}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{figures/cacti_real_comparison.pdf}
\caption{CACTI real data: calibrated vs.\ mismatched reconstruction for \textit{duomino} and \textit{hand} scenes. Mismatched masks produce temporal ghosting artifacts, mirroring simulation findings.}
\label{fig:cacti_real_supp}
\end{figure}

"""
    # Insert after \maketitle
    supp_tex = supp_tex.replace(
        r"""\maketitle

% ============================================================================
% A. PER-SCENE RESULTS""",
        r"""\maketitle

""" + moved_figs + r"""% ============================================================================
% A. PER-SCENE RESULTS"""
    )

    # ================================================================
    # 6. Write files
    # ================================================================
    write(OUT_DIR / "main.tex", main_tex)
    write(OUT_DIR / "supplementary.tex", supp_tex)
    print(f"  Written: main.tex")
    print(f"  Written: supplementary.tex")

    # ================================================================
    # 7. Copy figures
    # ================================================================
    fig_src = PAPER_DIR / "figures"
    fig_dst = OUT_DIR / "figures"
    fig_count = 0
    for f in fig_src.iterdir():
        if f.is_file() and f.suffix.lower() in ('.png', '.pdf', '.jpg', '.jpeg', '.eps'):
            shutil.copy2(f, fig_dst / f.name)
            fig_count += 1
    print(f"  Copied {fig_count} figures")

    # Copy llncs.cls and splncs04.bst
    for name in ['llncs.cls', 'splncs04.bst']:
        src = PAPER_DIR / name
        if src.exists():
            shutil.copy2(src, OUT_DIR / name)
            print(f"  Copied: {name}")

    # ================================================================
    # 8. Compile PDFs
    # ================================================================
    import subprocess
    import os

    os.chdir(str(OUT_DIR))

    for doc in ['main', 'supplementary']:
        print(f"\n  Compiling {doc}.tex...")
        for pass_num in range(3):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', f'{doc}.tex'],
                capture_output=True, text=True, timeout=120
            )

        pdf_path = OUT_DIR / f"{doc}.pdf"
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            # Count pages
            try:
                import fitz
                doc_obj = fitz.open(str(pdf_path))
                pages = len(doc_obj)
                doc_obj.close()
                print(f"  {doc}.pdf: {pages} pages, {size_kb:.0f} KB")
            except:
                print(f"  {doc}.pdf: {size_kb:.0f} KB")
        else:
            print(f"  ERROR: {doc}.pdf not created!")
            # Show last few lines of log
            log_path = OUT_DIR / f"{doc}.log"
            if log_path.exists():
                lines = log_path.read_text(encoding='utf-8', errors='replace').split('\n')
                for line in lines[-20:]:
                    if 'Error' in line or 'error' in line:
                        print(f"    {line}")

    print(f"\n{'=' * 60}")
    print(f"  ECCV resubmission ready!")
    print(f"  Folder: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
