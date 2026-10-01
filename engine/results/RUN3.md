# Knowledge engine, run 3 (1 Oct 2026, integrity-wsl, 696 s, scripted: no language model)

Corpus ldct-v2 (24 TCIA patients) after **qualification** (low-dose noise >= 1.5x full-dose noise): 12 chest patients
qualify (ratio 2.15-3.57); all 12 abdomen patients are excluded with their measured ratio (1.09-1.35), see
run3_qualification.json. 324 problems; patient-level split: development C016 C021 C030 C027 C004 C002, held out C052
C067 C081 C012 C077 C050 (162 problems). Tuned on two development patients, validated on a third (validator-ldct-1).
All 9 submissions accepted.

| Arm | Library | Held-out pass rate | Paired vs common mode |
|---|---|---|---|
| Common mode (adopted defaults) | none | 128/162 (79.0%) | |
| PWM mode, L0 | principle, twin, benchmark | 128/162 (79.0%) | identical |
| PWM mode, L3 | 3 accepted (150 HU) | 139/162 (85.8%) | 11 gains, 0 losses, p = 0.001 |
| PWM mode, L6 | 6 (+100 HU) | 153/162 (94.4%) | 25 gains, 0 losses, p = 6e-8 |
| PWM mode, L9 | 9 (+200 HU) | 156/162 (96.3%) | 28 gains, 0 losses, p = 7.5e-9 |

Common mode's failures: lesion contrast retained < 0.5 (33) and CNR < 3 (2), concentrated on radius-3 lesions; the
library's small-lesion settings, tuned on other patients, fix them. Across runs 1-3 the gain appears only once
matching solutions exist, grows with the library, and never costs a problem.

Limits unchanged: one field, one method family (a solution is a validated setting); scripted, not an agent; chest only
until noisier abdomen data (projections) is available.
