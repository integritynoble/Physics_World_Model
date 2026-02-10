"""InverseNet: A Benchmark Dataset for Inverse-Problem-Aware Calibration.

This package generates paired (measurement, ground-truth, mismatch-label)
datasets for three compressive-imaging modalities (SPC, CACTI, CASSI) and
runs four benchmark tasks:

    T1 -- Operator parameter estimation  (theta-error RMSE)
    T2 -- Mismatch identification        (accuracy, F1)
    T3 -- Calibration                    (theta-error reduction, CI coverage)
    T4 -- Reconstruction under mismatch  (PSNR, SSIM, SAM)

Modules
-------
gen_spc         Generate SPC dataset sweep
gen_cacti       Generate CACTI dataset sweep
gen_cassi       Generate CASSI dataset sweep
mismatch_sweep  Parameterised mismatch injection helpers
run_baselines   Run all four InverseNet tasks with baseline methods
leaderboard     Score and rank submissions per task
package         Package dataset for HuggingFace / Zenodo
manifest_schema Pydantic schema for manifest.jsonl records
"""

__version__ = "0.1.0"
