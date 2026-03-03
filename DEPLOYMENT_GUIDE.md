# Deployment Guide — Standard Benchmark Pages for All 168 Modalities

## Status
✅ **Code is ready for deployment to production**

## What Changed

### 1. **Comprehensive Check.md Documentation** (Committed)
- All 168 modalities now have standardized 6-point QA check.md files
- Files located in: `benchmarks/learn/{variant}/check.md`
- Includes: Benchmark Page Errors, Dataset Inspection, Algorithm Coverage, Action Items
- One hand-crafted file (CT) preserved; 167 auto-generated with domain-specific content

### 2. **Auto-Generation Pipeline** (Already in Production Code)
The following files provide the **standard tier metadata** for all modality pages:

#### Key Files (already committed):
- `platform/pwm_platform/services/benchmark_database/_challenge_data.py`
  - `generate_challenge_config()` — auto-generates tier configs with introduction sections
  - `_TIER_TEMPLATES` — generic tier templates with "What you get", "How to use", etc.
  - `_CATEGORY_TIER_DATA_SOURCES` — per-category data source routing

- `platform/pwm_platform/services/benchmark_database/_factory.py`
  - `build_variant()` — expands registry entries into full benchmark configs
  - `_make_b_challenge()` — builds challenge benchmarks with complete tier structures (lines 42-96)
  - Includes: introduction, spec_ranges, visible_data, dataset info for ALL tiers

- `platform/pwm_platform/services/benchmark_database/__init__.py`
  - `VARIANT_DATABASE` — builds all 168 variants with complete tier metadata
  - Lines 44-93: Pre-populates CHALLENGE_CONFIG for missing variants

- `platform/pwm_platform/templates/variant_benchmarks.html`
  - `tier_card` macro (lines 12-142) — displays introduction, leaderboard, spec ranges
  - Lines 29-41: Renders introduction sections if present
  - Lines 86-112: Renders spec_ranges tables

### 3. **Leaderboard Generation** (Already in Production Code)
- `platform/pwm_platform/services/benchmark_database/_leaderboard_generator.py`
  - `generate_full_leaderboard()` — creates challenge leaderboards for all variants
  - Generates synthetic scores for variants without hand-crafted data

## What Should Display on Live Site

For **every modality page** (e.g., https://pwm.platformai.org/benchmark/mri):

### Public Tier:
- ✓ "Full-access development tier..." summary
- ✓ Expandable "What you get & how to use" section
- ✓ Leaderboard table with scores, PSNR, SSIM
- ✓ Spec Ranges collapsible section with all parameters
- ✓ Download button for HDF5

### Dev Tier:
- ✓ "Blind evaluation tier..." summary
- ✓ Expandable section (no ground truth, use consistency)
- ✓ Leaderboard table
- ✓ Spec Ranges
- ✓ "Submit Reconstruction" button

### Hidden Tier:
- ✓ "Fully blind server-side..." summary
- ✓ Expandable section (no download, Docker container)
- ✓ Leaderboard (if available)
- ✓ Spec Ranges
- ✓ "Submit Algorithm" button

## Deployment Steps

### Step 1: Pull Latest Code
```bash
git pull origin master
```

### Step 2: Verify Data Structure
```bash
python3 -c "
import sys; sys.path.insert(0, 'platform')
from pwm_platform.services.benchmark_database import get_variant, list_all_variant_keys
variants = list_all_variant_keys()
print(f'Total variants: {len(variants)}')

# Spot-check a few
for key in ['ct', 'mri', 'ultrasound']:
    v = get_variant(key)
    if v and v.get('benchmarks'):
        bm = v['benchmarks'][0]
        print(f'{key}: {len(bm.get(\"leaderboard\", []))} leaderboard entries')
"
```

### Step 3: Clear Cache
```bash
# Clear Docker image cache
docker system prune -a

# Clear FastAPI cache
rm -rf ~/.cache/pytest
```

### Step 4: Rebuild & Deploy
```bash
cd platform
docker compose build
docker compose up -d
```

### Step 5: Verify Live Pages
Visit these URLs and verify they display the standard format:
- https://pwm.platformai.org/benchmark/ct (reference implementation)
- https://pwm.platformai.org/benchmark/mri (should match CT structure)
- https://pwm.platformai.org/benchmark/ultrasound (should match CT structure)
- https://pwm.platformai.org/benchmark/ptychography (should match CT structure)

Each should show:
1. Three tier cards (Public, Dev, Hidden)
2. Tier introduction summaries
3. Expandable "What you get & how to use" sections
4. Per-tier leaderboards
5. Spec Ranges tables
6. Download/Submit buttons

## What's Already in the Code

✅ **Challenge Config Generation**: Variants without hand-crafted configs auto-generate from category templates  
✅ **Tier Templates**: Generic introduction sections for all tiers  
✅ **Leaderboard Generation**: Synthetic scores for variants without published baselines  
✅ **Template Rendering**: HTML template properly displays all tier metadata  
✅ **Documentation**: All 168 modalities have comprehensive check.md files  

## Verification Checklist

After deployment, run these checks:

```bash
# 1. Test page loads without errors
curl -s https://pwm.platformai.org/benchmark/mri | grep -c "What you get"

# 2. Test leaderboard data is present
curl -s https://pwm.platformai.org/benchmark/mri | grep -c "public_score"

# 3. Test spec ranges are present
curl -s https://pwm.platformai.org/benchmark/mri | grep -c "Spec Ranges"

# 4. Test all tiers have introduction sections
curl -s https://pwm.platformai.org/benchmark/mri | grep -c "introduction.summary"
```

## Expected Results

All 168 modality pages should now display:
- **8 modalities with hand-crafted configs**: CT, MRI, SD-CASSI, CACTI, SPC_Block, SPC_Kronecker, + 2 others
- **160 modalities with auto-generated configs**: Powered by category templates

All should show consistent tier layouts, introduction sections, and leaderboards.

## Rollback Plan

If issues occur:
```bash
git revert <commit-hash>
docker compose down
docker compose build
docker compose up -d
```

## Questions?

Check:
- `platform/pwm_platform/services/benchmark_database/_challenge_data.py` — config generation
- `platform/pwm_platform/services/benchmark_database/_factory.py` — variant building
- `platform/pwm_platform/templates/variant_benchmarks.html` — page rendering
- `benchmarks/learn/*/check.md` — QA documentation
