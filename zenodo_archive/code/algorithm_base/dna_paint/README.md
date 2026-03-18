# DNA-PAINT Super-Resolution (`dna_paint`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `dna_paint_dl` | DECODE-PAINT | `pwm_core.recon.smlm_solvers.decode_smlm_recon` | Yes | Speiser, A. et al. (2021) DL for dense SMLM, Nature Methods 18:1090 |

## Usage

```python
# Import and run
from algorithm_base.dna_paint import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.dna_paint import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy (PWM) | — | 30.9 | 35.6 | done |
| CARE (PWM) | — | 30.9 | 35.6 | done |
| DECODE-PAINT (PWM) | — | 30.9 | 35.6 | done |
| precomputed_baseline (test) | — | 30.9 | 35.6 | done |
| DeepSTORM | 2018 | 22.0 | 35.6 | done |
| PICASSO | 2020 | 20.0 | 35.6 | done |
