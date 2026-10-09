# LIBQ-xrf_tomo-calib observation report

**Outcome: no publicly accessible, modality-matched source-level XFCT calibration measurements were identified in the reviewed sources. No instrument twin was fit; the search observation is recorded and work stopped at the task's no-data condition.**

Task: PWM-22 backlog, priority P3, calibration-twin, scope A1; modality `xrf_tomo`, query `identify`, judge task IPWM-24. Branch `libq/LIBQ-xrf_tomo-calib` starts from `origin/ipwm/IPWM-24b` at `78da0f00116291b5433aaed411966064e968aee8`. No earlier `origin/libq/LIBQ-xrf_tomo-calib` branch existed in the fetched bundle.

## Search and disposition

| Source | Public contents and licence | Disposition |
|---|---|---|
| [XFCT low-dose phantom deposit, Zenodo 11243130](https://zenodo.org/records/11243130) | CC BY 4.0; PNG image outputs for a phantom at 25%, 50%, 75%, and 100% dose | XFCT data, but no raw per-position spectra/sinograms, calibration scans, or geometry metadata needed to fit an instrument forward model |
| [Hyperspectral chemical phantom CT, Zenodo 6787489](https://zenodo.org/records/6787489) | CC BY 4.0; multichannel sinograms for chemical phantoms plus scan geometry | Not modality-matched: energy-resolved attenuation X-ray CT, not X-ray fluorescence tomography |
| [2025 benchtop XFCT study](https://www.nature.com/articles/s41598-025-03900-0) | Study reports experimental XFCT phantom measurements and a calibration curve; its data availability statement says supporting data are available from the corresponding author upon reasonable request | Not a publicly accessible dataset, so not eligible for this task |

The retrieved Zenodo metadata JSON files and their SHA-256 values are retained under [`evidence`](evidence/). The 6787489 sinograms are multi-gigabyte files; they were not downloaded because the record identifies them as the wrong imaging modality. No paid spend or fitting was performed.

## Observation records

| Record | Outcome | Key metrics |
|---|---|---|
| XRF-TOMO-CAL-001 | Search completed; no qualifying public calibration measurements found; stopped before fitting | Calibrated forward model: none; fit error: none; judge verdict: not run |

Full environment, search method, source licences, metadata hashes, commands, and judge disposition are in [`records.jsonl`](records.jsonl), one JSON object per line.

## What failed and why

- The public XFCT phantom deposit contains only image outputs, not acquisition-level spectra or geometry needed to fit a measured instrument twin.
- A public chemical-phantom sinogram dataset includes geometry, but it is hyperspectral attenuation CT rather than fluorescence tomography.
- The 2025 XFCT measurement source is available only on request, not as public or licensed downloadable calibration data.
- The IPWM-24 `xrf_tomo/calibrate` judge record on the base branch is `NOT POSSIBLE` / `NOT BUILT`: it has no source-backed modality-specific dataset or calibration criterion. The D44 tool test therefore was not run; no judge score or fit performance is claimed.
- The second-machine sample check remains pending because no fit or portable calibration sample was produced.

## D44 moat

**D44 moat type (1), measurements / calibrations:** no qualifying public X-ray fluorescence tomography calibration measurements were found among the reviewed sources. No calibrated forward model or fit error is claimed. This records a data-access and modality-match observation, not a calibration result.

**D44 tool test:** not run. There was no qualifying source-level calibration dataset to exercise, and the bundled IPWM-24 calibration judge is `NOT POSSIBLE` / `NOT BUILT` for `xrf_tomo`.
