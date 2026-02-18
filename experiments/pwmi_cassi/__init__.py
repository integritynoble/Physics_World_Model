"""PWMI-CASSI experiments (Paper 3).

Evaluate UPWMI calibration on CASSI mismatch families with bootstrap
uncertainty quantification and capture-advisor integration.

Modules
-------
run_families     Run calibration across all mismatch families
cal_budget       Sweep calibration budget (number of captures)
comparisons      Run 5 baselines on identical CASSI data
stats            Statistical analysis (paired t-tests, effect sizes, CI coverage)
"""

__version__ = "0.1.0"
