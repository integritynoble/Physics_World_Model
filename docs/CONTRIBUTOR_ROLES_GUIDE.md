# PWM Contributor Economy — Role Testing Guide

This guide walks you through testing every contributor role defined in the
Dyson Swarm strategy. All actions use the live platform at
`https://pwm.platformai.org` and the `pwm` CLI.

**Your account**: `integrityyang@gmail.com` (role: **admin** — can perform all actions)

---

## Prerequisites

```bash
# 1. Make sure you're logged in on the platform
#    Go to https://pwm.platformai.org/login
#    Sign in with integrityyang@gmail.com

# 2. Make sure CLI is installed locally
cd /home/spiritai/pwm/Physics_World_Model
pip install -e packages/pwm_core/
```

---

## Role 1: Claim Curator

**Responsibility**: Reviews auto-scaffolded ClaimCards, promotes or flags them through trust tiers.

### Step 1: View the claim review queue

Go to: **https://pwm.platformai.org/claims**

You should see the review queue with any existing scaffolded claims.

### Step 2: Scaffold a new ClaimCard manually

On the `/claims` page, click **"+ Scaffold new ClaimCard"** and fill in:

| Field | Example value |
|-------|--------------|
| arXiv ID | `2603.12345` |
| Paper title | `MST-L: Mask-guided Spectral-wise Transformer for CASSI` |
| Authors | `Yuanhao Cai, Jing Lin` |
| Modality | `cassi` |
| Method | `MST-L` |
| Claimed PSNR | `35.4` |
| Claimed SSIM | `0.94` |

Click **"Scaffold ClaimCard"**. The claim appears in the queue with status **Pending**.

### Step 3: Scaffold claims automatically from arXiv

```bash
# Run the arXiv scaffolder to fetch recent imaging papers
python3 -c "
import sys; sys.path.insert(0, 'pwm_product/platform')
from pwm_platform.services.arxiv_scaffolder import ArxivScaffolder

scaffolder = ArxivScaffolder('/tmp/pwm_claim_queue')
papers = scaffolder.fetch_recent_papers(max_results=20)
claims = scaffolder.scaffold_batch(papers)
print(f'Fetched {len(papers)} papers, scaffolded {len(claims)} claims')
for c in claims[:5]:
    print(f'  [{c.relevance_score:.2f}] {c.title[:70]}')
"
```

Refresh `/claims` — the auto-scaffolded claims appear in the queue.

### Step 4: Approve a claim

On the `/claims` page, find a pending claim and click **"Approve"**.
The claim moves to status **Approved** and trust tier **Draft** — it now
appears on the leaderboard.

### Step 5: Reject a claim

Find another pending claim and click **"Reject"**.
Enter a reason (e.g., "Not relevant to computational imaging").
The claim moves to status **Rejected**.

---

## Role 2: Benchmark Reviewer

**Responsibility**: Independent reproduction of results; required for Reproduced tier.

### Step 1: Find an approved claim

On `/claims`, filter by **Approved** status.

### Step 2: Reproduce the result

Run the algorithm through the PWM harness to verify the claimed result:

```bash
# Example: reproduce a CASSI MST-L claim
pwm evaluate \
    --modality cassi \
    --solver traditional_cpu \
    --track correct \
    --scenes 5 \
    --seed 42 \
    --emit-certificate

# Check the certificate
cat run_cassi_traditional_cpu_*/certificate.json | python3 -m json.tool
```

Compare your measured PSNR/SSIM with the claim's values. If they match
within tolerance, the result is independently reproduced.

### Step 3: Mark as reproduced

On `/claims`, click **"Mark Reproduced"** on the approved claim.
The trust tier upgrades from **Draft** to **Reproduced**.

### Step 4: Demote if reproduction fails

If reproduction fails (PSNR differs by more than 3 dB):

```bash
# Via API
curl -X POST https://pwm.platformai.org/trust/demote/SUBMISSION_ID \
    -H "Content-Type: application/json" \
    -d '{"reason": "reproduction_failure", "reviewer_id": "integrityyang"}'
```

---

## Role 3: Modality Maintainer

**Responsibility**: Owns a modality's DomainProfile, curates MethodCards and DatasetCards.

### Step 1: View your modality

Go to: **https://pwm.platformai.org/modalities/ct** (or any modality)

You see:
- Spec DAG visualization (inline SVG)
- Algorithm leaderboard with trust-tier badges
- Triad diagnostic bars (G1/G2/G3)
- Related modalities

### Step 2: Check the DomainProfile

```bash
# View the registered profile for CT
python3 -c "
from pwm_core.spec.profile_registry import get_profile
p = get_profile('ct')
print(f'Modality: {p.display_name}')
print(f'Primitive chain: {p.primitive_chain}')
print(f'Gates: {[g.gate_id for g in p.domain_gates]}')
print(f'Noise model: {p.noise_model}')
print(f'Maturity: {p.extra.get(\"maturity\", \"unknown\")}')
"
```

### Step 3: Review a benchmark submission

Go to the variant benchmark page:
**https://pwm.platformai.org/benchmark/ct**

Check the leaderboard. As a maintainer, you verify that:
- Algorithm names are correct
- PSNR/SSIM values are plausible for this modality
- Trust-tier badges match the evidence

### Step 4: Curate MethodCards

```bash
# View existing method cards
python3 -c "
from pwm_core.cards.method_card import MethodCard
# Create a new MethodCard for a solver you maintain
card = MethodCard(
    method_id='gap_tv_ct',
    method_name='GAP-TV',
    version='1.0.0',
    modality='ct',
    solver_type='traditional_cpu',
    description='Total-variation regularized reconstruction via ADMM',
    code_uri='https://github.com/integritynoble/Physics_World_Model',
    compute_budget_s=60,
)
print(f'MethodCard: {card.method_name} for {card.modality}')
"
```

---

## Role 4: Dataset Steward

**Responsibility**: Maintains dataset quality, versioning, and access.

### Step 1: Ingest a new dataset

```bash
# Create some test data
mkdir -p /tmp/test_ct_data
python3 -c "
import numpy as np
for i in range(5):
    np.save(f'/tmp/test_ct_data/slice_{i:03d}.npy', np.random.randn(64,64).astype('float32'))
print('Created 5 test slices')
"

# Ingest into PWM format
pwm ingest /tmp/test_ct_data --modality ct --out /tmp/ingested_ct
```

### Step 2: Verify the DatasetCard

```bash
cat /tmp/ingested_ct/dataset_card.json | python3 -m json.tool
```

The DatasetCard should contain: modality, sample count, shape, dtype, license, provenance.

### Step 3: Upload to GCS (if maintaining a public dataset)

```bash
# Upload to the benchmark datasets bucket
gsutil -m cp -r /tmp/ingested_ct gs://pwm-benchmark-datasets/datasets/Benchmark/ct_test/
```

### Step 4: Search the federated registry

```bash
python3 -c "
from tools.dataset_federation.fetch import search_datasets
results = search_datasets(modality='ct')
for d in results:
    print(f'  {d[\"name\"]}: {d[\"source\"]} ({d.get(\"samples\", \"?\")} samples)')
"
```

---

## Role 5: Method Integrator

**Responsibility**: Adapts published algorithms into PWM-compatible solver plugins.

### Step 1: Scaffold a new solver plugin

```bash
pwm scaffold solver my_new_solver
# Creates: contrib/solvers/my_new_solver/
```

### Step 2: Implement the solver

Edit the scaffolded file to implement your algorithm:

```python
# contrib/solvers/my_new_solver/solver.py
import numpy as np

def solve(y, H_matrix, **kwargs):
    """Your reconstruction algorithm."""
    # y = measurements, H_matrix = forward model
    x_hat = np.linalg.lstsq(H_matrix, y.ravel(), rcond=None)[0]
    return x_hat.reshape(kwargs.get('x_shape', (64, 64)))
```

### Step 3: Test your solver through the harness

```bash
pwm evaluate \
    --modality ct \
    --solver my_new_solver \
    --track correct \
    --scenes 3 \
    --emit-certificate
```

### Step 4: Install as a plugin

```bash
pwm install ./contrib/solvers/my_new_solver --type solver --tier local
pwm plugins  # verify it appears
```

### Step 5: Benchmark and compare

```bash
# View the RunBundle
pwm view run_ct_my_new_solver_*/

# Compare against the golden reference
cat run_ct_my_new_solver_*/certificate.json | python3 -c "
import sys, json
cert = json.load(sys.stdin)
print(f'Trust tier: {cert[\"trust_tier\"]}')
print(f'Gates: {list(cert[\"gate_verdicts\"].keys())}')
for g, v in cert['gate_verdicts'].items():
    print(f'  {g}: {v[\"verdict\"]}')
"
```

---

## Role 6: Judge-Rule Author

**Responsibility**: Proposes and maintains domain Judge gates.

### Step 1: View existing gates

```bash
python3 -c "
from pwm_core.targeting.gates import run_r1_r4
from pwm_core.targeting.scientific_validity import verify_all_s_conditions
print('Operational gates (R1-R4):')
print('  R1: spec completeness')
print('  R2: reproducibility')
print('  R3: metric integrity')
print('  R4: budget compliance')
print()
print('Scientific conditions (S1-S4):')
print('  S1: finite specifiability')
print('  S2: Hadamard stability')
print('  S3: approximability')
print('  S4: certifiability')
"
```

### Step 2: View domain-specific gates

```bash
# Imaging gates
python3 -c "
from pwm_core.spec.profile_registry import get_profile
p = get_profile('ct')
for g in p.domain_gates:
    print(f'  {g.gate_id}: {g.description}')
"

# CT QC gates
python3 -c "
from pwm_core.core.runbundle.triad_report import CTQCDiagnosticReport
r = CTQCDiagnosticReport(run_id='test', spec_id='test', noise_std_hu=5.0, cnr=12.0)
print(f'CT QC gates: drift={r.drift_detected}, concern={r.dominant_concern}')
print(f'Complete: {r.is_complete}')
"
```

### Step 3: Propose a new gate (via RFC)

To propose a new Judge gate, create an RFC:

```bash
cp docs/governance/RFC_TEMPLATE.md docs/governance/rfcs/RFC-0001-new-ct-artifact-gate.md
# Edit the RFC with your proposed gate specification
```

The RFC should specify:
- Gate name and ID
- What it checks
- Pass/fail/warn criteria
- Which domain profile it belongs to
- Whether it's a hard gate (sets safety-brake) or diagnostic-only

---

## Role 7: Red-Team Contributor

**Responsibility**: Probes the harness for failure modes, earns credit.

### Step 1: Try to break the gates

```bash
# Create a bundle with a corrupted artifact
python3 -c "
import json, tempfile, numpy as np, hashlib
from pathlib import Path
from pwm_core.targeting.runbundle_emitter import issue_certificate

# Create a valid-looking bundle
d = Path(tempfile.mkdtemp()) / 'run_attack_test'
d.mkdir(parents=True)
(d / 'artifacts').mkdir()

# Write artifact with one hash, then corrupt it
art = d / 'artifacts' / 'x.npy'
np.save(str(art), np.zeros((32,32)))
good_hash = hashlib.sha256(art.read_bytes()).hexdigest()

manifest = {
    'version': '0.3.0', 'spec_id': 'attack_test', 'timestamp': '2026-01-01',
    'provenance': {'modality': 'ct', 'solver': 'test', 'git_hash': 'abc', 'seeds': [42], 'python_version': '3.12'},
    'metrics': {'psnr_db': 99.9, 'ssim': 0.999, 'runtime_s': 0.1},
    'artifacts': {'x': 'artifacts/x.npy'},
    'hashes': {'x': f'sha256:{good_hash}'},
}
(d / 'runbundle_manifest.json').write_text(json.dumps(manifest))

# Now corrupt the artifact AFTER the hash was computed
np.save(str(art), np.ones((32,32)) * 999)

# Try to get a certificate
cert_path = issue_certificate(d)
cert = json.loads(cert_path.read_text())
print(f'Trust tier: {cert[\"trust_tier\"]}')
print(f'R3 (integrity): {cert[\"gate_verdicts\"][\"r3\"][\"verdict\"]}')
print(f'Risk flags: {cert[\"risk_flags\"]}')
print()
if cert['trust_tier'] == 'rejected':
    print('PASS: Gate correctly caught the corruption')
else:
    print('FAIL: Gate missed the corruption — report this!')
"
```

### Step 2: Try adversarial metric inflation

```bash
# Try to submit inflated metrics that don't match artifacts
python3 -c "
import json, tempfile, numpy as np, hashlib
from pathlib import Path
from pwm_core.targeting.runbundle_emitter import issue_certificate

d = Path(tempfile.mkdtemp()) / 'run_inflate_test'
d.mkdir(parents=True)
(d / 'artifacts').mkdir()

# Real reconstruction (low quality)
art = d / 'artifacts' / 'x.npy'
np.save(str(art), np.random.randn(32,32))
h = hashlib.sha256(art.read_bytes()).hexdigest()

# But claim amazing metrics
manifest = {
    'version': '0.3.0', 'spec_id': 'inflate_test', 'timestamp': '2026-01-01',
    'provenance': {'modality': 'ct', 'solver': 'test', 'git_hash': 'abc', 'seeds': [42], 'python_version': '3.12', 'declared_budget_s': 600},
    'metrics': {'psnr_db': 50.0, 'ssim': 0.999, 'runtime_s': 1.0},
    'artifacts': {'x': 'artifacts/x.npy'},
    'hashes': {'x': f'sha256:{h}'},
}
(d / 'runbundle_manifest.json').write_text(json.dumps(manifest))

cert_path = issue_certificate(d)
cert = json.loads(cert_path.read_text())
print(f'Trust tier: {cert[\"trust_tier\"]}')
print(f'All gates: { {k: v[\"verdict\"] for k, v in cert[\"gate_verdicts\"].items()} }')
print()
print('Note: R3 passes because hashes match (artifact is authentic)')
print('The inflated metrics are caught at REPRODUCTION stage,')
print('when a reviewer re-runs and gets different PSNR.')
"
```

---

## Role 8: Instrument Contributor

**Responsibility**: Provides InstrumentCards and calibration data.

### Step 1: Create an InstrumentCard

```bash
python3 -c "
import json
from pwm_core.cards.instrument_card import InstrumentCard

card = InstrumentCard(
    instrument_id='siemens_somatom_001',
    manufacturer='Siemens Healthineers',
    model='SOMATOM Force',
    modality='ct',
    calibration_state='calibrated',
    operating_parameters={
        'kVp': [80, 100, 120, 140],
        'mA_range': [50, 800],
        'rotation_time_s': 0.5,
        'detector_rows': 192,
    },
)
print(json.dumps(card.to_dict(), indent=2, default=str))
"
```

### Step 2: Run a CT QC workflow

```bash
python3 scripts/demo_ct_qc_workflow.py
```

This produces:
- `instrument_card.json` — the scanner's identity
- `commissioning_baseline.json` — reference values
- `ct_qc_diagnostic_report.json` — drift detection results
- `certificate.json` — trust verdict for the QC run

### Step 3: View the QC results

```bash
cat /tmp/pwm_ct_qc_demo/certificate.json | python3 -c "
import sys, json
cert = json.load(sys.stdin)
print(f'Trust tier: {cert[\"trust_tier\"]}')
print(f'Domain flags: {json.dumps(cert.get(\"domain_flags\", {}), indent=2)}')
"
```

---

## Trust-Tier Promotion Workflow (all roles)

The full promotion path you can test end-to-end:

```
Step 1: Scaffold a claim                    → Draft (pending_review)
Step 2: Approve as Claim Curator            → Draft (approved, on leaderboard)
Step 3: Run independent reproduction        → produces matching RunBundle
Step 4: Mark Reproduced as Benchmark Reviewer → Reproduced
Step 5: Judge verifies all R1-R4 gates      → Certificate issued
Step 6: Reviewer signs off evidence package  → Certified
```

### Test the full flow via API

```bash
# 1. Scaffold
curl -X POST https://pwm.platformai.org/claims/scaffold \
    -H "Content-Type: application/json" \
    -d '{"arxiv_id": "2603.99999", "title": "Test Paper", "modality": "ct", "method": "FBP", "claimed_psnr": 25.0}'

# 2. Approve (copy the claim_id from response)
curl -X POST https://pwm.platformai.org/claims/CLAIM_ID/approve \
    -H "Content-Type: application/json" \
    -d '{"reviewer": "integrityyang", "reason": "Valid CT reconstruction claim"}'

# 3. Mark reproduced
curl -X POST https://pwm.platformai.org/claims/CLAIM_ID/reproduce \
    -H "Content-Type: application/json" \
    -d '{"reviewer": "integrityyang", "reason": "Independently verified via pwm evaluate"}'

# 4. Check trust promotion status
curl https://pwm.platformai.org/trust/status/SUBMISSION_ID

# 5. Check contributor economy activation
curl https://pwm.platformai.org/trust/contributor-economy-status
```

---

## Badge Milestones

| Badge | How to earn | Test it by |
|-------|------------|-----------|
| **First Certified** | Get one result to Certified tier | Complete the full trust promotion flow above |
| **Reproducer** | Reproduce 1 result independently | Run `pwm evaluate` and match a claim |
| **Reproducer x10** | Reproduce 10 results | Repeat for multiple modalities |
| **Red-team find** | Discover a gate vulnerability | Try the red-team attacks in Role 7 |
| **Modality maintainer** | Curate a modality's profile | Review and update a DomainProfile |
| **Dataset contributor** | Ingest one dataset | Run `pwm ingest` as in Role 4 |

---

## Quick Reference

| Action | URL / Command |
|--------|--------------|
| Claim review queue | `https://pwm.platformai.org/claims` |
| Benchmark leaderboard | `https://pwm.platformai.org/benchmark/cassi` |
| Modality detail | `https://pwm.platformai.org/modalities/ct` |
| Trust promotion API | `https://pwm.platformai.org/trust/status/{id}` |
| Contributor economy status | `https://pwm.platformai.org/trust/contributor-economy-status` |
| Run evaluation | `pwm evaluate --modality ct --emit-certificate` |
| View RunBundle | `pwm view run_ct_*/` |
| Reproduce a run | `pwm reproduce run_ct_*/` |
| System health | `pwm doctor` |
| Install plugin | `pwm install ./solver_dir --type solver` |
| Ingest dataset | `pwm ingest /data/dir --modality ct` |
| Generate synthetic data | `pwm synthesize --modality ct --n 10` |
