# Modify Plan: CT (X-ray Computed Tomography)

**Created:** 2026-03-03
**Status:** Done (items 1-3 fixed; item 4 deferred — category-level reference)

## Changes Required

### 1. Fix FBP Citation (ERROR)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
**Line ~27:** Change FBP source from `"Jin et al., IEEE TIP 2017"` to `"Kak & Slaney, IEEE Press 1988"`
**Line ~669:** Same fix in CATEGORY_REAL_SCORES

### 2. Fix DuDoTrans Citation Venue (ERROR)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
**Line ~33:** Change `"Wang et al., IEEE TMI 2022"` to `"Wang et al., MLMIR 2022"`
**Line ~675:** Same fix in CATEGORY_REAL_SCORES

### 3. Complete PnP-ADMM Citation (WARNING)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
**Line ~29:** Change `"Venkatakrishnan et al., 2013"` to `"Venkatakrishnan et al., IEEE GlobalSIP 2013"`
**Line ~671:** Same fix in CATEGORY_REAL_SCORES

### 4. Fix Dataset Size Description (WARNING)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
**Line ~548:** Change CT description from "512x512" to "362x362" to match actual LoDoPaB-CT data

## Implementation

All changes are in `_algorithm_catalog.py` — citation string updates only. No algorithm logic changes.
