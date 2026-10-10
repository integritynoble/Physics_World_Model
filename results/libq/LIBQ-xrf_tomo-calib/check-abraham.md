# Second machine check — LIBQ-xrf_tomo-calib

**Machine:** Linux x86_64, Python 3.12.3 (`~/pwm-venv`); check date 2026-10-10.  
**Source:** `origin/libq/LIBQ-xrf_tomo-calib`, checked out as `libq-check/LIBQ-xrf_tomo-calib` at `cfb56170ee089d9c3f525ca3e5e3a42c1efe52c6`.  
**Result:** the sole checked record was replayed. It remains an observation stopped before fitting; there are no numeric calibration/fit values or judge score to reproduce. Records were not modified.

## Per-record comparison

| Record | Recorded result and metrics | Reproduced result | Comparison |
|---|---|---|---|
| `XRF-TOMO-CAL-001` | Search observation; no qualifying public source-level XFCT calibration data identified. `forward_model=null`, `fit_error=null`, `judge_verdict=not_run`, `public_qualifying_calibration_data_found=false`; `reproduced=null`. | No fit or judge run. Zenodo 11243130 lists CC BY 4.0 and PNG image outputs; Zenodo 6787489 lists CC BY 4.0 and multichannel phantom sinograms/geometry for attenuation CT. No qualifying public XFCT calibration measurements were established. | Outcome and null metrics match. No numeric results were expected or produced. |

## Commands and replay

The record lists these original commands:

```text
git fetch origin library
git switch -c libq/LIBQ-xrf_tomo-calib origin/ipwm/IPWM-24b
curl -L https://zenodo.org/api/records/6787489
curl -L https://zenodo.org/api/records/11243130
/Users/integrityyang/pwm-venv/bin/python --version and package version inspection
```

The requested check branch was based on the provided source branch at `cfb56170ee08…` rather than switching over it. Both API requests were replayed with their recorded `curl -L` URLs (responses saved under `/tmp` for comparison). The Python interpreter and dependency versions were inspected with `~/pwm-venv/bin/python`; observed versions: Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, Matplotlib 3.11.2, MCP 2.2.0. The provided core archive was extracted to `/tmp/pwm-core-3899e328`; `PWM_CORE_SRC` was set to its `Physics_World_Model/packages/pwm_core` directory. Commands ran with `nice -n 19` and BLAS/OpenMP thread variables set to 1.

No fitting command exists in the record: its action says no fitting code was run, and its D44 judge test was not run because there was no qualifying dataset and the judge was `NOT POSSIBLE` / `NOT BUILT`.

## Provenance and licence check

- **Zenodo 6787489:** recorded API metadata SHA-256 `69270e43a824c735fa38290d206250b146d9bf5c63c3943c60dddef5215ede77`; replayed API JSON has the same hash. The metadata reports `cc-by-4.0`. Its file listing includes multi-gigabyte chemical-phantom sinograms and scan-parameter files; these are hyperspectral attenuation CT data, not XRF tomography. The modality mismatch is consistent with the record.
- **Zenodo 11243130:** archived evidence SHA-256 matches the hash in `records.jsonl` (`b3796d01b3266f3d981f938b5d0a368448d6f2532cbb162a2fb622a4fb05cff4`) and reports `cc-by-4.0`. The replayed API response also reports `cc-by-4.0` and the image-output file listing, but its current raw JSON SHA-256 differs: `a770d82ca0a4e175b61939c5b82a16d7cc8219faeaef10553972986f90061bbe`.
- The cited 2025 benchtop XFCT study is described in the record as making supporting data available by request. No public underlying measurement dataset or separate licence was found in the retained record evidence; it is not treated as a usable licensed dataset.

The two Zenodo deposits’ licences and modality/file descriptions check out against the retained evidence and live metadata. Provenance for the cited paper’s underlying measurements remains request-only, as recorded.

## Mismatch

The only observed mismatch is metadata-byte drift for Zenodo 11243130. Reproduction command:

```sh
nice -n 19 curl -L https://zenodo.org/api/records/11243130 -o /tmp/libq-xrf11243130-replay.json
```

Recorded/archived SHA-256: `b3796d01b3266f3d981f938b5d0a368448d6f2532cbb162a2fb622a4fb05cff4`; replayed response SHA-256: `a770d82ca0a4e175b61939c5b82a16d7cc8219faeaef10553972986f90061bbe`. The live response still reports CC BY 4.0 and PNG image outputs, so the metadata change does not alter the observation. The archived evidence and checked `records.jsonl` were left unchanged.
