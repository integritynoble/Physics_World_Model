# PWM Federated Dataset Registry

The federated dataset registry is a versioned, append-only catalog of public imaging datasets. PWM is the catalog — not the host. All data stays at the original source.

## Query the registry

```bash
# List all datasets
python tools/dataset_federation/fetch.py list

# Filter by modality
python tools/dataset_federation/fetch.py list --modality ct

# Show details for a dataset
python tools/dataset_federation/fetch.py info lodopab_ct

# Emit a PWM DatasetCard JSON
python tools/dataset_federation/fetch.py emit-card lodopab_ct
```

## Add a dataset

Add an entry to `registry.yaml` following the existing format. Submit a pull request. The registry is append-only within v1 — existing entries are not modified.

## Covered datasets

| Dataset | Modality | Samples | License |
|---------|----------|---------|---------|
| AAPM Low-Dose CT Grand Challenge | CT | 2,378 | Research only |
| fastMRI Knee | MRI | 1,594 | CC-BY-NC-4.0 |
| fastMRI Brain | MRI | 6,970 | CC-BY-NC-4.0 |
| BioImage Archive (Fluorescence) | Fluorescence microscopy | 800 | CC-BY-4.0 |
| LoDoPaB-CT | CT | 42,895 | CC-BY-4.0 |
| M4Raw MRI | MRI | 7,020 | CC-BY-4.0 |
| KAIST TSA (SD-CASSI) | CASSI | 30 | Research only |
