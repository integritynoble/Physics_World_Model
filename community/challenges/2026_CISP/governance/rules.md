# CISP 2026 Rules

## 1. Eligibility

- Open to all individuals and teams worldwide
- No limit on team size
- A person may be on at most 2 teams
- PWM core developers may participate but are scored separately ("host track")

## 2. Registration

- Teams must register between 2026-07-01 and 2026-08-01
- Registration requires: team name, team members, institutional affiliation, tracks entered
- Late registration accepted until 2026-08-15 (sealed data release) but no sandbox data provided

## 3. Submission Rules

### 3.1 Format

- Submissions are standard PWM RunBundles (v0.3.0)
- One submission per team per track (resubmission allowed until deadline)
- RunBundle integrity: SHA-256 hashes verified by stewards

### 3.2 Solver Constraints

- Same solver configuration across all scenes within a track
- For Track 4 (Cross-Modal): same solver configuration across ALL modalities
- Solver must pass isolation checks (`pwm contrib check`)
- No access to ground truth during reconstruction

### 3.3 Compute Budget

- Declared GPU-hours must be honest
- Budget ratio > 2x declared = Disqualification
- Budget ratio < 0.5x declared = Compute dishonesty penalty (-10%)
- Compute budget is per-modality, not shared across modalities

### 3.4 Code Availability

- **Anonymous tier**: RunBundle only (no code required)
- **Identified tier**: RunBundle + contributor profile
- **Reproducible tier**: RunBundle + full source code (required for prizes)

## 4. Anti-Gaming Rules

### 4.1 Scoring Integrity

- S_rank = 0.3 * S_retrospective + 0.7 * S_prospective
- Retrospective: scored on training/sandbox data (teams see these results)
- Prospective: scored on sealed test data (teams do NOT see until after deadline)
- The 70/30 split prevents overfitting to known scenarios

### 4.2 Prohibited Actions

| Action | Consequence |
|--------|-------------|
| Accessing sealed ground truth | Immediate DQ + 2-year ban |
| Submitting results from another team | Immediate DQ + permanent ban |
| Fabricating compute budget declarations | DQ + public disclosure |
| Reverse-engineering sealed simulator parameters | DQ (if detected via statistical tests) |
| Using test data for training/tuning | DQ |
| Multiple accounts / Sybil submissions | DQ for all associated accounts |

### 4.3 Detection

- Statistical tests for memorization (sealed data includes traps)
- Compute profiling vs declared budgets
- Source code analysis (for reproducible tier)
- Cross-submission similarity checks

## 5. Scoring

### 5.1 Primary Metric

All tracks use **rho** (recovery ratio) as the primary metric:

```
rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)
```

### 5.2 Track-Specific Scoring

- **Track 1 (Correct)**: Mean rho across CASSI scenes
- **Track 2 (Temporal)**: Mean rho + temporal consistency bonus
- **Track 3 (Medical)**: Mean rho + clinical ROI SSIM
- **Track 4 (Cross-Modal)**: Harmonic mean of per-modality rho

### 5.3 Safety Brakes

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Recovery ratio | rho < 0.30 | Blocked (not ranked) |
| Budget exceeded | > 2x declared | Disqualification |
| Uncertainty miscalibration | Coverage deviates > 15% | Flagged |
| Re-projection error | > 3x median | Quarantined |

### 5.4 Ranking

- Within each track: sorted by final_score (after anti-Goodhart)
- Ties broken by: (1) oracle_gap, (2) RoIC, (3) submission time
- Composite ranking: weighted sum across entered tracks

## 6. Appeals

### 6.1 Grounds for Appeal

- Scoring error (computational or procedural)
- Incorrect disqualification
- Steward conflict of interest
- Data integrity issue

### 6.2 Process

1. Appeal filed within 7 days of results announcement
2. Filed via GitHub Issue with label `cisp-appeal`
3. Steward board reviews within 14 days
4. Decision by majority vote (2/3 stewards)
5. Decision is final (no further appeal)

### 6.3 What Cannot Be Appealed

- Subjective judgment on scientific merit
- Anti-Goodhart penalty calculations (these are mechanical)
- Safety brake enforcement
- Late submission rejection

## 7. Prizes

| Place | Prize |
|-------|-------|
| 1st per track | Co-authorship on CISP proceedings + invited talk at workshop |
| 2nd-3rd per track | Named in CISP proceedings |
| Best calibrator | Special recognition for calibration innovation |
| Best new modality | Recognition for novel modality contribution |

- Only reproducible-tier submissions are eligible for 1st place
- Anonymous and identified submissions are eligible for 2nd-3rd
- Prize decisions are made by the steward board

## 8. Timeline

| Phase | Dates | What Happens |
|-------|-------|-------------|
| Registration | 2026-07-01 to 2026-08-01 | Teams register, receive sandbox data |
| Sealed Data Release | 2026-08-15 | Sealed-simulator test data released |
| Submission Window | 2026-08-15 to 2026-10-15 | Teams submit RunBundles |
| Evaluation | 2026-10-15 to 2026-11-01 | Stewards verify, score, rank |
| Results Announcement | 2026-11-15 | Public leaderboard + proceedings |
| Appeals Window | 2026-11-15 to 2026-11-22 | 7-day appeal period |
| Workshop | 2026-12-01 (TBD) | Invited talks from top teams |

## 9. Data Rights

- Sealed-simulator data is synthetic (CC-BY-4.0)
- Submitted RunBundles become public after results announcement (CC-BY-4.0)
- Solver source code (reproducible tier) retains contributor's license
- See `docs/IP_POLICY.md` for full intellectual property policy
