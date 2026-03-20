# Digital Holographic Microscopy

**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

## Algorithms (17 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Angular Spectrum Method |  | CPU |
| `fresnel` | Fresnel Propagation |  | CPU |
| `gerchberg_saxton` | Gerchberg-Saxton |  | CPU |
| `hio` | Hybrid Input-Output (HIO) |  | CPU |
| `error_reduction` | Error Reduction |  | CPU |
| `raar` | RAAR |  | CPU |
| `tv_phase` | TV-Phase Retrieval |  | CPU |
| `tikhonov` | Tikhonov Regularisation |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) |  | CPU |
| `best_quality` | PhaseNet |  | GPU |
| `famous_dl` | prDeep |  | GPU |
| `deep_dih` | DeepDIH |  | GPU |
| `holonet` | HoloNet |  | GPU |
| `small_gpu` | PhaseGAN |  | GPU |
| `holo_diffusion` | HoloDiffusion |  | GPU |
| `neural_holo` | NeuralHolo |  | GPU |
| `holo_mamba` | HoloMamba |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.holography.solvers import run_solver, list_solvers
list_solvers()                    # 17 algorithms
y = ...                           # hologram (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```
