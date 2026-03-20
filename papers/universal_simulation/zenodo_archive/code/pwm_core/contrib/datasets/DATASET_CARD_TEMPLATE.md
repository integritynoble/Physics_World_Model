# Dataset Card: [DATASET_NAME]

## Overview

| Field | Value |
|-------|-------|
| **Name** | [dataset_name] |
| **Version** | [v1.0] |
| **Modality** | [cassi / cacti / spc / widefield / ct / mri / ...] |
| **License** | [MIT / CC-BY-4.0 / ...] |
| **Total samples** | [N] |
| **Splits** | train: [N], val: [N], test: [N] |

## Description

[Brief description of the dataset, its purpose, and key characteristics.]

## Provenance

| Field | Value |
|-------|-------|
| **Source** | [Original data source or generation method] |
| **Generator script** | [Path to generation script, if synthetic] |
| **Generation date** | [YYYY-MM-DD] |
| **Random seed** | [seed value for reproducibility] |
| **Git hash** | [commit hash at generation time] |

## Data Format

- **File format**: [.npy / .npz / .pt]
- **Shape**: [H x W x C] or [T x H x W]
- **Dtype**: [float32 / float64]
- **Value range**: [0, 1] or [min, max]

## Integrity

| File | SHA256 |
|------|--------|
| manifest.json | [hash] |
| [sample_file] | [hash] |

## Usage Example

```python
import numpy as np
data = np.load("path/to/sample.npy")
```

## Citation

```bibtex
@misc{dataset_name,
  title={[Title]},
  author={[Author]},
  year={[Year]},
}
```
