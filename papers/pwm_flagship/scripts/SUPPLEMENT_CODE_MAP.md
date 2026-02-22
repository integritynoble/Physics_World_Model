# Supplementary-to-Code Cross-Reference Map

Maps each supplementary table/figure to the script or data source that generates it.

| Supplement Section | Table/Figure | Generation Script | Data Source |
|---|---|---|---|
| Table S1 | 16 correction configurations | `run_real_data_4scenario.py` | KAIST, CACTI benchmark, Set11 |
| Table S2 | CASSI per-scene | `run_real_data_4scenario.py` | KAIST 10 scenes |
| Table S3 | 26-modality registry | Static YAML | `packages/pwm_core/contrib/graph_templates.yaml` |
| Table S4 | YAML registry summary | Static | `packages/pwm_core/contrib/` |
| Table S10 | SSIM comparison | `run_real_data_4scenario.py` | Same as Table S1 |
| Table S11 | CASSI SAM | `run_real_data_4scenario.py` | KAIST |
| Tables S12-S13 | Gate 1+2 validation | `run_real_data_4scenario.py --gate12` | Synthetic sweeps |
| Note 7 | Clinical CT QA | `run_ct_4scenario.py --clinical` | ACR phantom simulation |
| Note 11 | MRI realistic | `run_mri_realistic.py` | fastMRI / M4Raw |
| Note 14 | ESPIRiT comparison | `run_espirit_comparison.py` | M4Raw |
| Note 15 | Extended hardware | `run_real_data_4scenario.py --real-data` | TSA, EfficientSCI, FIPS, Zenodo |
| Note 16 | Per-scene + bootstrap CIs | `run_real_data_4scenario.py` | All modality datasets |

## Environment

- Python >= 3.9
- PyTorch >= 1.12 with CUDA >= 11.3
- Install: `pip install -e packages/pwm_core`
- GPU: NVIDIA GPU with >= 8 GB VRAM
- Approximate total runtime: ~4 hours (single GPU)

## Reproducibility

- All random seeds are recorded in RunBundle manifests
- SHA-256 hashes of all inputs/outputs are in RunBundle
- Platform info (Python version, GPU, CUDA) recorded per experiment
