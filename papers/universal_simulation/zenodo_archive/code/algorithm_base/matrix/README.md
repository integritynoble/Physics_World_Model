# Generic Matrix Sensing (`matrix`)

Category: Compressive Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L1 | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | FISTA-L1 (high quality) | `pwm_core.recon.classical.run_fista_l2` | No | Beck & Teboulle 2009 |
| `famous_dl` | LISTA | `pwm_core.recon.lista.run_lista` | No | Gregor & LeCun, ICML 2010 |
| `small_gpu` | LISTA | `pwm_core.recon.lista.run_lista` | No |  |

## Usage

```python
# Import and run
from algorithm_base.matrix import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.matrix import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| LISTA | 2010 | 28.5 | 19.5 | partial |
| FISTA | 2009 | 27.0 | 19.5 | partial |
| OMP | 1993 | 24.0 | 19.5 | partial |
| FISTA-L1 (high quality) (PWM) | — | 22.1 | 19.5 | done |
| precomputed_baseline (test) | — | 22.1 | 19.5 | done |
