# Coherent Diffractive Imaging / Phase Retrieval

**Input**: diffraction intensities (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/public/`

## Algorithms (16 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | HIO |  | CPU |
| `best_quality` | RAAR [proxy] |  | CPU |
| `famous_dl` | prDeep [proxy] |  | CPU |
| `small_gpu` | prDeep [proxy] |  | CPU |
| `wiener` | Wiener Deconvolution |  | CPU |
| `landweber` | Landweber Iteration |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `tv_admm` | TV-ADMM |  | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `dl_phasenet` | PhaseNet |  | GPU |
| `dl_prdeep` | prDeep |  | GPU |
| `dl_transformer` | Phase-Transformer |  | GPU |
| `dl_diffusion` | Phase-Diffusion |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.phase_retrieval.solvers import run_solver, list_solvers
list_solvers()                    # 16 algorithms
y = ...                           # diffraction intensities (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Papers

- `papers/universal_simulation/benchmark/03_quantum_chemistry/`
