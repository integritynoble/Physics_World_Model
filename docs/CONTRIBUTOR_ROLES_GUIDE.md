# PWM Contributor Economy Guide

> A Dyson Swarm works when every collector has identity and reward.

This guide explains how the PWM contributor economy works — roles, badges,
trust-tier promotion, and how to participate using the web platform at
**https://pwm.platformai.org**.

---

## Overview

PWM defines 8 contributor roles. Each role earns credit through specific
actions, and badges are awarded automatically when milestones are reached.

| Role | What you do | Platform access needed |
|------|------------|----------------------|
| **Claim Curator** | Review scaffolded ClaimCards, approve or reject | Reviewer or Admin |
| **Benchmark Reviewer** | Independently reproduce results | Reviewer or Admin |
| **Modality Maintainer** | Curate a modality's profile and leaderboard | Any user (assigned by admin) |
| **Dataset Steward** | Ingest and maintain datasets | Any user |
| **Method Integrator** | Adapt algorithms into PWM solver plugins | Any user |
| **Judge-Rule Author** | Propose and maintain domain Judge gates | Any user (RFC required) |
| **Red-Team Contributor** | Probe the harness for failure modes | Any user |
| **Instrument Contributor** | Provide InstrumentCards and calibration data | Any user |

---

## Getting Started

### 1. Sign up and log in

Go to **https://pwm.platformai.org/login** and sign in (or create an account).

### 2. Get your roles assigned

An admin assigns contributor roles via the **Users & Roles** page:

- Admin goes to **https://pwm.platformai.org/admin/users**
- Clicks **Edit** on your account
- Checks the contributor roles you should have
- Sets which modalities you maintain (if applicable)
- Clicks **Save**

### 3. View your profile

Your public contributor profile is at:
**https://pwm.platformai.org/contributors/YOUR_USERNAME**

It shows your roles, badges, stats, maintained modalities, and activity history.

---

## Trust-Tier Promotion (how results get certified)

Every benchmark result goes through a trust ladder:

```
Draft → Author-confirmed → Reproduced → Certified
```

| Step | Who does it | What happens |
|------|------------|-------------|
| **1. Scaffold** | Claim Curator or auto (arXiv scanner) | A ClaimCard is created in the review queue |
| **2. Approve** | Claim Curator (reviewer/admin) | Claim appears on leaderboard at Draft tier |
| **3. Reproduce** | Benchmark Reviewer | Independent reproduction confirms the result |
| **4. Certify** | Judge (automated) + Reviewer signoff | All R1-R4 gates pass; Certificate issued |

---

## Role 1: Claim Curator

**What you do**: Review ClaimCards in the queue and decide which ones go to the leaderboard.

### Via Web UI

1. Go to **https://pwm.platformai.org/claims**
2. Click **"+ Scaffold new ClaimCard"** to add a paper manually:
   - Enter arXiv ID, title, authors, modality, method, claimed PSNR/SSIM
   - Click **"Scaffold ClaimCard"**
3. The claim appears in the queue with status **Pending**
4. Click **"Approve"** to promote to leaderboard (Draft tier)
5. Click **"Reject"** to remove with a reason

### Auto-scaffolding from arXiv (server-side)

The arXiv scaffolder can automatically fetch recent imaging papers and create
Draft ClaimCards. This runs on the server:

```bash
cd /home/spiritai/pwm/Physics_World_Model
python3 -c "
import sys; sys.path.insert(0, 'pwm_product/platform')
from pwm_platform.services.arxiv_scaffolder import ArxivScaffolder
scaffolder = ArxivScaffolder('/tmp/pwm_claim_queue')
papers = scaffolder.fetch_recent_papers(max_results=20)
claims = scaffolder.scaffold_batch(papers)
print(f'Fetched {len(papers)} papers, scaffolded {len(claims)} claims')
"
```

Scaffolded claims appear at `/claims` for human review.

---

## Role 2: Benchmark Reviewer

**What you do**: Independently reproduce results to verify claims.

### Step 1: Find an approved claim

Go to **https://pwm.platformai.org/claims** and filter by **Approved**.

### Step 2: Reproduce the result

Run the claimed algorithm through the PWM harness:

```bash
pwm evaluate --modality cassi --solver traditional_cpu --track correct \
    --scenes 5 --seed 42 --emit-certificate
```

### Step 3: Compare results

```bash
pwm view run_cassi_traditional_cpu_*/
```

If measured PSNR/SSIM matches the claim within tolerance, the result is confirmed.

### Step 4: Mark as reproduced

On `/claims`, click **"Mark Reproduced"** on the approved claim.
The trust tier upgrades to **Reproduced**.

---

## Role 3: Modality Maintainer

**What you do**: Own a modality's DomainProfile, curate its leaderboard.

### View your modality

Go to: **https://pwm.platformai.org/modalities/ct** (or any modality you maintain)

You see:
- Spec DAG visualization (forward model pipeline)
- Algorithm leaderboard with trust-tier badges
- Diagnostic bars (imaging: G1/G2/G3 Triad)
- Your name in the **Modality Maintainers** section

### Review the leaderboard

Go to: **https://pwm.platformai.org/benchmark/ct**

As maintainer, verify that algorithm names, PSNR/SSIM values, and trust-tier
badges are correct.

---

## Role 4: Dataset Steward

**What you do**: Ingest, validate, and maintain benchmark datasets.

### Ingest a dataset (CLI)

```bash
# Create test data
mkdir -p /tmp/test_ct_data
python3 -c "
import numpy as np
for i in range(5):
    np.save(f'/tmp/test_ct_data/slice_{i:03d}.npy', np.random.randn(64,64).astype('float32'))
"

# Ingest into PWM format — produces a DatasetCard
pwm ingest /tmp/test_ct_data --modality ct --out /tmp/ingested_ct
```

### Verify the DatasetCard

```bash
cat /tmp/ingested_ct/dataset_card.json | python3 -m json.tool
```

---

## Role 5: Method Integrator

**What you do**: Adapt published algorithms into PWM-compatible solver plugins.

### Scaffold, implement, and test

```bash
# 1. Scaffold a new solver
pwm scaffold solver my_solver

# 2. Edit contrib/solvers/my_solver/solver.py with your algorithm

# 3. Test through the harness
pwm evaluate --modality ct --solver my_solver --scenes 3 --emit-certificate

# 4. Install as a plugin
pwm install ./contrib/solvers/my_solver --type solver --tier local

# 5. View results
pwm view run_ct_my_solver_*/
```

---

## Role 6: Judge-Rule Author

**What you do**: Propose and maintain domain-specific Judge gates.

PWM has two gate families:

| Family | Purpose | When checked |
|--------|---------|-------------|
| **R1-R4** (operational) | Spec completeness, reproducibility, metric integrity, budget | Every run |
| **S1-S4** (scientific) | Finite specifiability, stability, approximability, certifiability | Design/audit time |

To propose a new gate, create an RFC:

```bash
cp docs/governance/RFC_TEMPLATE.md docs/governance/rfcs/RFC-0001-my-new-gate.md
```

The RFC should specify: gate name, what it checks, pass/fail criteria, which
domain profile it belongs to, and whether it sets `safety-brake`.

---

## Role 7: Red-Team Contributor

**What you do**: Probe the harness for failure modes and earn credit.

### Test: Artifact corruption detection

```bash
python3 -c "
import json, tempfile, numpy as np, hashlib
from pathlib import Path
from pwm_core.targeting.runbundle_emitter import issue_certificate

d = Path(tempfile.mkdtemp()) / 'run_attack'
d.mkdir(parents=True); (d / 'artifacts').mkdir()
art = d / 'artifacts' / 'x.npy'
np.save(str(art), np.zeros((32,32)))
h = hashlib.sha256(art.read_bytes()).hexdigest()
manifest = {'version':'0.3.0','spec_id':'test','timestamp':'2026-01-01',
    'provenance':{'modality':'ct','solver':'test','git_hash':'abc','seeds':[42],'python_version':'3.12'},
    'metrics':{'psnr_db':99.9,'ssim':0.999,'runtime_s':0.1},
    'artifacts':{'x':'artifacts/x.npy'},'hashes':{'x':f'sha256:{h}'}}
(d / 'runbundle_manifest.json').write_text(json.dumps(manifest))
np.save(str(art), np.ones((32,32)) * 999)  # corrupt after hashing
cert = json.loads(issue_certificate(d).read_text())
print(f'R3 (integrity): {cert[\"gate_verdicts\"][\"r3\"][\"verdict\"]}')
print('PASS' if cert['trust_tier'] == 'rejected' else 'FAIL: gate missed corruption!')
"
```

If you find a way to bypass a gate, report it to earn red-team credit.

---

## Role 8: Instrument Contributor

**What you do**: Provide InstrumentCards and calibration data for physical devices.

### Run the CT QC workflow demo

```bash
python3 scripts/demo_ct_qc_workflow.py
```

This produces an InstrumentCard, commissioning baseline, drift detection
report, and a Certificate for the QC run — all at `/tmp/pwm_ct_qc_demo/`.

---

## Admin: User & Role Management

Only users with **admin** platform role can manage other accounts.

### Manage users

Go to: **https://pwm.platformai.org/admin/users**

You can:
- **Edit** any user — change username, email, platform role (user/reviewer/admin),
  contributor roles, and maintained modalities
- **Reset password** — set a new password for any user
- **Deactivate/Activate** — disable or re-enable login
- **Delete** — permanently remove an account

### Access levels

| Platform role | What they can see |
|--------------|------------------|
| **user** | Pricing, Subscription, Logout |
| **reviewer** | + Claims Review, Review, Submissions |
| **admin** | + Users & Roles management |

---

## Badges

Badges are awarded automatically when milestones are reached.

| Badge | Icon | Milestone |
|-------|------|-----------|
| First Certified | 🏆 | Get 1 result to Certified tier |
| Reproducer | 🔄 | Reproduce 1 result independently |
| Reproducer x10 | 🔄🔄 | Reproduce 10 results |
| Claim Curator | 📋 | Review 5 claims |
| Senior Curator | 📋⭐ | Review 25 claims |
| Modality Maintainer | 🔧 | Maintain at least 1 modality |

View your badges on your contributor profile page.

---

## Activation Trigger

The contributor economy activates when the platform reaches:

- **10+ RunBundles** at Draft tier or above
- From **3+ distinct contributors**

Check the current status:
**https://pwm.platformai.org/trust/contributor-economy-status**

---

## Quick Reference

| Action | Where |
|--------|-------|
| Sign in | `https://pwm.platformai.org/login` |
| Claim review queue | `https://pwm.platformai.org/claims` |
| Benchmark leaderboard | `https://pwm.platformai.org/benchmark/cassi` |
| Modality detail | `https://pwm.platformai.org/modalities/ct` |
| Admin: Users & Roles | `https://pwm.platformai.org/admin/users` |
| Your profile | `https://pwm.platformai.org/contributors/YOUR_USERNAME` |
| Run evaluation | `pwm evaluate --modality ct --emit-certificate` |
| View RunBundle | `pwm view run_ct_*/` |
| Reproduce a run | `pwm reproduce run_ct_*/` |
| System health check | `pwm doctor` |
| Install plugin | `pwm install ./solver_dir --type solver` |
| Ingest dataset | `pwm ingest /data/dir --modality ct` |
| Generate synthetic data | `pwm synthesize --modality ct --n 10` |
