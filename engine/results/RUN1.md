# Knowledge engine, run 1 (1 Oct 2026, integrity-wsl, 290 s, scripted: no language model)

Real paired low-dose CT (TCIA LDCT-and-Projection-data, 4 patients x 3 slices). Problems: patient x slice x lesion
contrast {100, 150, 200 HU} x radius {3, 5, 7 px}. Patient-level split: development C016, C067; held out C052, C111
(54 held-out problems). Acceptance by validator-ldct-1 (independent implementation; agrees with the runtime's judge).

| Arm | Library | Held-out pass rate |
|---|---|---|
| Common mode (adopted defaults) | none | 46/54 (85.2%) |
| PWM mode, L0 | principle, twin, benchmark | 46/54 (85.2%) |
| PWM mode, L3 | 3 accepted solutions (150 HU) | 49/54 (90.7%) |
| PWM mode, L5 | 5 (+100 HU; 1 rejected by the validator) | 52/54 (96.3%) |
| PWM mode, L8 | 8 (+200 HU) | 53/54 (98.1%) |

Paired against common mode: PWM mode lost no problem common mode solved; gains 3, 6, 7 (exact McNemar p = 0.25,
0.031, 0.016). Both held-out patients improved (C052 24 -> 26/27, C111 22 -> 27/27). Every common-mode failure is a
radius-3 lesion: the defaults smooth too widely; the library's small-lesion settings, tuned on other patients,
transfer. Budget per held-out problem: one solver run and one validation in both arms; the one-time development search
was 675 runs.

Limits: two held-out patients; a "solution" here is a validated setting of one method (transfer by known conditions),
not a new method; scripted, not an agent. Next: more patients from TCIA, then the real agent loop (OpenCode + model)
with pwm_search / pwm_get / pwm_submit, then growth driven by workers Abr guides.
