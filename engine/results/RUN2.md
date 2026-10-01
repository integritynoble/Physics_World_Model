# Knowledge engine, run 2 (1 Oct 2026, integrity-wsl, 2,492 s, scripted: no language model)

Corpus ldct-v2: 24 real TCIA patients (12 chest, 12 abdomen), 3 middle slices each; 648 problems. Patient-level split:
12 development (8 chest, 4 abdomen), 12 held out (4 chest, 8 abdomen); 324 held-out problems. Tuned per anatomy on two
development patients, checked by validator-ldct-1 on a third (chest: C052, C021 -> C081; abdomen: L033, L072 -> L049).

**The validator rejected every abdomen submission and 6 of 9 chest ones;** the library ended with 6 chest solutions.

| Held out | Common mode (adopted defaults) | PWM mode, library L6 | Paired |
|---|---|---|---|
| Chest, 4 new patients (108) | 87 (80.6%) | 101 (93.5%) | 14 gains, 0 losses, exact p = 0.0001 |
| Abdomen, 8 patients (216) | 0 | 17 | benchmark invalid there (below) |

**Why abdomen fails for everyone:** in this collection the abdomen "low-dose" reconstructions are nearly as clean as
the full-dose ones (noise ~31-34 HU against ~26-29 HU; PSNR of the untouched slice 42.5-43.5 dB; correlation 0.9995;
pair RMSE ~11-16 HU, against 110-190 HU on chest). Every smoothing moves the image away from the reference, so "PSNR
above the untouched slice" cannot be met: common mode 0/216, every failure on that criterion. That is a benchmark
qualification failure, not a solver or engine failure, and the validator refused all of it (no false acceptance). The
17 abdomen passes in PWM mode used chest-tuned settings (very gentle smoothing); not counted while the benchmark is
invalid there.

**Reading:** the engine's effect replicates on new patients (run 1: 85% -> 98% on 2 chest patients; run 2: 81% -> 94%
on 4 others, no losses). The next fix is qualification before optimization: a problem family must exclude pairs whose
low-dose noise is not clearly above full-dose noise (the paper's own thesis, E1).
