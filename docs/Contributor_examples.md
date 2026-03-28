# PWM Contributor Roles — Step-by-Step Examples

> Two worked examples per role with exact data to copy-paste.
> Every example is a **UI workflow** on **https://pwm.platformai.org**.
> Sign in first, then follow each example step by step.

---

## Role 1: Modality Maintainer

### Example A: Scaffold a missing CT paper

**Account:** `platformaigpt@gmail.com`

1. Click your profile name (top-right) → click **Review & Approve Claims**
2. Click **+ Scaffold new ClaimCard**
3. Fill in exactly:

| Field | Value to enter |
|-------|---------------|
| arXiv ID | `2305.18727` |
| Paper Title | `DiffusionCT: Latent Diffusion Model for CT Image Standardization` |
| Authors | `Md Selim, Jie Zhang, Michael A. Brooks` |
| Modality | `ct` |
| Method | `DiffusionCT` |
| Claimed PSNR | `38.2` |
| Claimed SSIM | `0.965` |
| Claim Type | `Paper Claim` |

4. Click **Scaffold ClaimCard**
5. The claim appears below with a yellow **Pending** badge and a blue **paper claim** badge
6. To verify: click **Benchmarks** → click **CT** → after the claim is approved, it appears on the leaderboard with a gray **Draft** badge

### Example B: Report a suspicious leaderboard entry

**Account:** `platformaigpt@gmail.com`

1. In the top nav bar, click **Benchmarks** → click **CT**
2. Look at the leaderboard — notice an algorithm showing 45 dB PSNR (too high for this modality)
3. Scroll below the leaderboard → click **Report an issue with this leaderboard**
4. Fill in:

| Field | Value to enter |
|-------|---------------|
| What's wrong? | `PSNR/SSIM looks incorrect or impossible` |
| Details | `FBP variant showing 45 dB PSNR on sparse-view CT — this exceeds theoretical limits for filtered backprojection. Likely a data entry error or wrong dataset.` |

5. Click **Submit Report**
6. The report creates a claim in the review queue — admin sees it at profile → **Review & Approve Claims**

---

## Role 2: Claim Curator

### Example A: Approve a CASSI paper claim

**Account:** `platformaigpt@gmail.com`

1. Click your profile name → click **Review & Approve Claims**
2. Click **+ Scaffold new ClaimCard**
3. Fill in:

| Field | Value to enter |
|-------|---------------|
| arXiv ID | `2204.07908` |
| Paper Title | `MST: Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction` |
| Authors | `Yuanhao Cai, Jing Lin, Xiaowan Hu, Haoqian Wang, Xin Yuan` |
| Modality | `cassi` |
| Method | `MST` |
| Claimed PSNR | `35.18` |
| Claimed SSIM | `0.948` |
| Claim Type | `Paper Claim` |

4. Click **Scaffold ClaimCard** → claim appears with **Pending** badge
5. Click the **View paper** link to verify the paper exists on arXiv
6. Click **Approve** → the claim moves to **Approved** status and **Draft** trust tier
7. To verify: click **Benchmarks** → click **CASSI** → MST appears on the leaderboard with a **Draft** badge

### Example B: Reject a low-quality claim

**Account:** `platformaigpt@gmail.com`

1. Click your profile name → click **Review & Approve Claims**
2. Click **+ Scaffold new ClaimCard**
3. Fill in:

| Field | Value to enter |
|-------|---------------|
| Paper Title | `My Amazing Algorithm That Gets 99 dB PSNR` |
| Modality | `ct` |
| Method | `MagicNet` |
| Claimed PSNR | `99.0` |
| Claimed SSIM | `0.999` |
| Claim Type | `Paper Claim` |

4. Click **Scaffold ClaimCard** → claim appears with **Pending** badge
5. Notice: no arXiv ID, no authors, and 99 dB PSNR is physically impossible
6. Click **Reject** → enter reason: `No paper source provided. 99 dB PSNR is physically impossible for CT reconstruction. Rejected.`
7. The claim moves to **Rejected** status with a red badge

---

## Role 3: Benchmark Reviewer

### Example A: Reproduce a CASSI result using SpecLab

**Account:** `platformaigpt@gmail.com`

1. In the top nav bar, click **Explore** → click **Reproduction Queue**
2. You see approved claims awaiting reproduction — find a CASSI claim
3. Note the claimed PSNR and method name
4. Open a new tab → go to **https://pwm.platformai.org/speclab**
5. In the SpecLab chat, type: `Run MST-L on CASSI public dataset`
6. SpecLab runs the reconstruction and shows the result with PSNR/SSIM metrics
7. Compare the SpecLab result against the claimed PSNR
8. Go back to **Explore** → **Reproduction Queue** → find the claim
9. If the result matches within tolerance: click **Mark Reproduced** → claim upgrades to **Reproduced** trust tier (blue badge)
10. If it does NOT match: click your profile name → **Review & Approve Claims** → click **+ Scaffold new ClaimCard** → fill in:

| Field | Value to enter |
|-------|---------------|
| Paper Title | `[REPRODUCE] MST CASSI claim PSNR mismatch — claimed 35.18 dB, measured 28.5 dB` |
| Modality | `cassi` |
| Claim Type | `Reproduction Issue` |

11. Click **Scaffold ClaimCard** → the reproduction issue is now in the review queue

### Example B: Reproduce a CT result using SpecLab

**Account:** `platformaigpt@gmail.com`

1. Go to **https://pwm.platformai.org/speclab**
2. In the chat, type: `Run FBP on CT public dataset`
3. SpecLab shows the FBP reconstruction with PSNR and SSIM
4. To test a different algorithm, type: `Now run TV-ADMM on CT`
5. Compare TV-ADMM result against the CT leaderboard values
6. Click **Benchmarks** → click **CT** → compare your SpecLab PSNR with the leaderboard entry
7. If the leaderboard value matches: go to **Explore** → **Reproduction Queue** → click **Mark Reproduced** on the matching claim
8. If the leaderboard value is suspiciously different: use **Report an issue with this leaderboard** (below the CT leaderboard) to flag it

---

## Role 4: Dataset Steward

### Example A: Register a new CT dataset

**Account:** `platformaigpt@gmail.com`

1. In the top nav bar, click **Explore** → click **Datasets**
2. You see the dataset browser with existing datasets listed
3. Click **+ Register New Dataset**
4. Fill in:

| Field | Value to enter |
|-------|---------------|
| Modality | `ct` |
| Dataset Name | `CT Synthetic Public v1` |
| Description | `10 synthetic CT phantom images, 64x64, Radon forward model with Poisson noise` |
| Number of Samples | `10` |
| Format | `HDF5` |
| License | `CC-BY-4.0` |
| Source URL | `https://github.com/integritynoble/Physics_World_Model` |

5. Click **Register** → the dataset appears in the list with a green confirmation
6. To verify: scroll the dataset list — you see "CT Synthetic Public v1" with modality tag "ct"

### Example B: Contribute benchmark data via upload

**Account:** `platformaigpt@gmail.com`

1. Click **Benchmarks** → click **MRI** → scroll to the bottom → click **Contribute**
2. You see the Contribute page with three tier options: Public, Dev, Hidden
3. Click **Submit Public Dataset**
4. A form appears — fill in:

| Field | Value to enter |
|-------|---------------|
| Method Name | `Real MRI Brain Slices` |
| Description | `5 brain slices from a 3T scanner, 128x128, k-space + ground truth` |
| Paper URL | *(leave empty)* |
| Code URL | *(leave empty)* |
| Upload File | *(select an .h5 or .npy file from your computer)* |

5. Click **Submit** → you see a success message with a submission ID
6. The contribution is now in the review queue — admin approves it before adding to the benchmark

---

## Role 5: Method Integrator

### Example A: Test an algorithm in SpecLab

**Account:** `platformaigpt@gmail.com`

1. Go to **https://pwm.platformai.org/speclab**
2. In the chat, type: `What algorithms are available for CT?`
3. SpecLab lists the available algorithms (FBP, TV-ADMM, SART, PnP-ADMM, etc.)
4. Type: `Run TV-ADMM on CT public dataset, sample 0`
5. SpecLab runs the reconstruction and shows:
   - Reconstructed image
   - Ground truth comparison
   - PSNR and SSIM metrics
6. Type: `Now run FBP on the same data`
7. Compare the two results side by side
8. To verify on the leaderboard: click **Benchmarks** → click **CT** → find TV-ADMM and FBP in the table

### Example B: Upload a solver to a competition

**Account:** `platformaigpt@gmail.com`

1. Click **Benchmarks** → click **CT** → scroll down → click **Compete**
2. You see the competition page with three steps: Public, Dev, Hidden
3. Under **Step 1 — Test on Public Dataset**, click **Download Public Dataset** to get the test data
4. Scroll to **Quick Solver Upload**
5. Click **Choose File** → select your solver Python file (e.g. `my_solver.py`)
6. Click **Upload Solver** → you see "Solver uploaded successfully!"
7. To report your score: scroll back up to Step 1 → click **Report My Score**
8. Fill in:

| Field | Value to enter |
|-------|---------------|
| Method Name | `My TV Solver` |
| Description | `Total-variation regularized FBP with 100 iterations` |
| PSNR (dB) | `32.5` |
| SSIM | `0.89` |

9. Click **Submit** → your score appears in the review queue

---

## Role 6: Judge-Rule Author

### Example A: Propose a new gate via the web form

**Account:** `platformaigpt@gmail.com`

1. In the top nav bar, click **Explore** → click **Gate Dashboard**
2. Read the existing gates organized in four layers:
   - **Operational (R1-R4):** Spec completeness, Reproducibility, Metric integrity, Budget compliance
   - **Scientific (S1-S4):** Finite specifiability, Stability, Approximability, Certifiability
   - **Imaging Triad (G1-G3):** Recoverability, Carrier budget, Operator mismatch
   - **CT QC:** Drift detection, Artifact detection, Threshold breach
3. Scroll down to **Propose a New Gate (RFC)** form
4. Fill in:

| Field | Value to enter |
|-------|---------------|
| Gate Name | `CT Ring Artifact Detection` |
| Description | `Detect ring artifacts in CT reconstructions by analyzing radial frequency components. Pass: <2 HU amplitude. Warn: 2-5 HU. Fail: >=5 HU.` |
| Modality | `ct` |
| Rationale | `Ring artifacts are a common CT reconstruction failure mode caused by detector element miscalibration. Current gates do not check for this.` |

5. Click **Submit RFC Proposal**
6. The proposal appears in the **Submitted Proposals** section below with a "Pending" badge
7. The core team reviews and votes on the RFC before it becomes an active gate

### Example B: Review existing gates and their status

**Account:** `platformaigpt@gmail.com`

1. Click **Explore** → click **Gate Dashboard**
2. You see four gate layers with pass/fail indicators:
   - **R1 — Spec completeness**: checks that all required fields are present in the manifest
   - **R2 — Reproducibility**: verifies the result can be reproduced with the same seed
   - **R3 — Metric integrity**: validates that artifact hashes match (detects tampering)
   - **R4 — Budget compliance**: ensures runtime stays within allowed limits
3. Each gate shows its status and a brief description
4. Scroll to the **Scientific gates (S1-S4)** section to see the theoretical foundations:
   - **S1 — Finite specifiability**: the imaging system can be fully described
   - **S2 — Stability**: small perturbations don't cause catastrophic failure
   - **S3 — Approximability**: the inverse problem has a computable solution
   - **S4 — Certifiability**: error bounds can be verified post-hoc
5. Scroll to **CT QC** to see domain-specific gates for quality control

---

## Role 7: Red-Team Contributor

### Example A: Report a vulnerability finding

**Account:** `platformaigpt@gmail.com`

1. In the top nav bar, click **Explore** → click **Red Team**
2. Read the **Bounty Board** — it lists open challenges like:
   - "Bypass R3 (artifact integrity) without detection"
   - "Submit a result that passes all gates but has wrong PSNR"
   - "Find a modality where S2 (stability) fails"
3. After testing (locally or in SpecLab), report your finding:
4. Click your profile name → click **Review & Approve Claims**
5. Click **+ Scaffold new ClaimCard**
6. Fill in:

| Field | Value to enter |
|-------|---------------|
| Paper Title | `[RED-TEAM] R3 correctly catches SHA-256 artifact corruption` |
| Authors | *(your name)* |
| Modality | `ct` |
| Method | `artifact_corruption_test` |
| Claim Type | `Red-Team Report` |

7. Click **Scaffold ClaimCard**
8. The finding appears with a **Pending** badge and a **red team** type badge
9. Admin reviews the finding and may award a bounty

### Example B: Test algorithm robustness via SpecLab

**Account:** `platformaigpt@gmail.com`

1. Go to **https://pwm.platformai.org/speclab**
2. In the chat, type: `Run FBP on CT with only 10 projection angles`
3. SpecLab simulates sparse-view CT and shows the reconstruction — notice the severe streaking artifacts
4. Type: `Now run TV-ADMM on the same sparse-view data`
5. Compare: TV-ADMM should handle sparse views much better than FBP
6. Type: `What happens if I add 50% Gaussian noise to the CT measurement?`
7. SpecLab simulates noisy CT — observe how different algorithms degrade
8. If you find an algorithm that claims high PSNR but fails on noisy/sparse data, report it:
9. Click your profile name → **Review & Approve Claims** → **+ Scaffold new ClaimCard**
10. Fill in:

| Field | Value to enter |
|-------|---------------|
| Paper Title | `[RED-TEAM] FBP claims 30 dB on standard CT but drops to 12 dB with 10 views` |
| Modality | `ct` |
| Claim Type | `Red-Team Report` |

11. Click **Scaffold ClaimCard**

---

## Role 8: Instrument Contributor

### Example A: Register a CT scanner

**Account:** `platformaigpt@gmail.com`

1. In the top nav bar, click **Explore** → click **Instruments**
2. You see the Instrument Registry (may be empty initially)
3. Click **+ Register New Instrument**
4. Fill in:

| Field | Value to enter |
|-------|---------------|
| Manufacturer | `Siemens Healthineers` |
| Model | `SOMATOM Force` |
| Modality | `ct` |
| Serial Number | `SN-2024-001` |
| Installation Date | `2024-06-15` |
| Calibration State | `Calibrated` |

5. Click **Register**
6. The scanner appears in the instrument list with a green **calibrated** indicator
7. Click on the instrument name to see its detail page with full InstrumentCard information

### Example B: Register an MRI scanner

**Account:** `platformaigpt@gmail.com`

1. Click **Explore** → click **Instruments**
2. Click **+ Register New Instrument**
3. Fill in:

| Field | Value to enter |
|-------|---------------|
| Manufacturer | `GE Healthcare` |
| Model | `SIGNA Premier` |
| Modality | `mri` |
| Serial Number | `GE-MRI-2025-001` |
| Installation Date | `2025-01-10` |
| Calibration State | `Calibrated` |

4. Click **Register** → the MRI scanner appears in the instrument list
5. You now see both scanners listed:
   - **Siemens SOMATOM Force** (CT) — green calibrated badge
   - **GE SIGNA Premier** (MRI) — green calibrated badge
6. Click on **GE SIGNA Premier** to view the InstrumentCard detail page

---

## Quick Reference: Example Data

| Role | Example A | Example B |
|------|-----------|-----------|
| **Modality Maintainer** | Scaffold: arXiv `2305.18727`, DiffusionCT, CT, 38.2 dB | Report: "45 dB FBP is impossible" on CT leaderboard |
| **Claim Curator** | Approve: arXiv `2204.07908`, MST, CASSI, 35.18 dB | Reject: "99 dB MagicNet, no paper" |
| **Benchmark Reviewer** | SpecLab: `Run MST-L on CASSI` → Mark Reproduced | SpecLab: `Run FBP on CT` → compare with leaderboard |
| **Dataset Steward** | Register: "CT Synthetic Public v1", 10 samples, HDF5 | Contribute: MRI brain slices via `/benchmark/mri/contribute` |
| **Method Integrator** | SpecLab: compare TV-ADMM vs FBP on CT | Upload solver at `/benchmark/ct/compete` → report score |
| **Judge-Rule Author** | Gate Dashboard: propose "CT Ring Artifact Detection" RFC | Gate Dashboard: review R1-R4, S1-S4, G1-G3 gates |
| **Red-Team** | Scaffold: `[RED-TEAM] R3 catches artifact corruption` | SpecLab: test FBP robustness with sparse views / noise |
| **Instrument** | Register: Siemens SOMATOM Force CT scanner | Register: GE SIGNA Premier MRI scanner |
