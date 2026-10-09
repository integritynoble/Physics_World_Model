# IPWM-24 diffraction judge status

**Status:** task inventory recorded for all 9 modalities and 4 functions. Three SIMULATE judges are implemented, 3 DESIGN slots are marked not built, and 30 slots are recorded as not possible in this attempt. This is not a complete set of judges and should not be used to unblock the PWM-mode library. The referenced `tasks/IPWM-24/{TASK,goal,status,log}.md` files and the IPWM-10 example files were absent from this checkout. Domain-expert review is queued for every judge and has not been done.

| Modality | DESIGN | SIMULATE | RECONSTRUCT | CALIBRATE |
|---|---|---|---|---|
| atom_probe | Not built | Judge v1.0.0; 12 held out; trivial 0%, best fixed 33% | Not possible | Not possible |
| cathodoluminescence | Not built | Judge v1.0.0; 3 held out; trivial 0%, best fixed 0%; below 12-unit target | Not possible | Not possible |
| maldi_msi | Not possible | Not possible | Not possible | Not possible |
| neutron_diffraction | Not possible | Not possible | Not possible | Not possible |
| saxs | Not built | Judge v1.0.0; 12 held out; trivial 0%, best fixed 100% | Not possible | Not possible |
| waxs | Not possible | Not possible: 2 measured frames only; larger candidate source unavailable (HTTP 401) | Not possible | Not possible |
| xray_crystallography | Not possible | Not possible | Not possible | Not possible |
| xrf_tomo | Not possible | Not possible | Not possible | Not possible |
| xfel_sfx | Not possible | Not possible | Not possible | Not possible |

“Not built” denotes a task with no judge/result implementation yet. “Not possible” is explained in that task’s `.json` and `.version.json`; no performance rates are claimed. For the three implemented judges, best-trivial rates are zero. The public PWM-core baseline is zero for APT and SAXS; it is unscored for CL. The fixed comparator is selected on development units. SIMULATE criteria and key manifests are recorded in the version files. These judges are preliminary: atom-probe uses synthetic isotope spectra, CL has only 3 held-out spectra, and SAXS compares radial detector curves with detector radius rather than absolute q because of source/header distance disagreement.

## Public data and provenance

- Atom probe SIMULATE: NIST Atomic Weights and Isotopic Compositions, public domain; retained isotope table SHA-256 and source URL are in [`data_provenance.json`](IPWM-24/data_provenance.json). The evaluation units are synthetic spectra generated from the NIST isotope lines.
- Cathodoluminescence SIMULATE: Zenodo dataset [10.5281/zenodo.6803239](https://doi.org/10.5281/zenodo.6803239), CC BY 4.0. Archive and selected-spectrum hashes are in [`data_provenance.json`](IPWM-24/data_provenance.json).
- SAXS SIMULATE and WAXS limitation record: Leeds dataset [10.5518/1170](https://doi.org/10.5518/1170), CC BY 4.0. Retained-frame hashes are in [`data_provenance.json`](IPWM-24/data_provenance.json). The raw public samples were retained only long enough to hash and process them.

Each implemented SIMULATE criterion is versioned at 1.0.0. Judge functions live in `pwm_mode/imaging/diffraction/judges.py` and do not import solver code. Traps cover unchanged input, zero/constant output, blur, public-copy submissions, and malformed shape. Hidden keys and the salt are stored outside the repository under `/tmp/ipwm24-keys`; version records contain key-manifest hashes and commitments, not the keys. The other task records explicitly leave criteria, hidden keys, rates, and unit counts unset.

## Modality-by-modality commits

The partial starting work was preserved. Commits for this attempt:

- `3a7eae2c` atom_probe
- `3e4f13bf` cathodoluminescence
- `dc6a6cd9` maldi_msi
- `dcac87db` neutron_diffraction
- `4858819d` saxs
- `019febde` waxs
- `181652d7` xray_crystallography
- `95ae72f8` xrf_tomo
- `0a1668cc` xfel_sfx
- This summary is written last; no push was performed.
