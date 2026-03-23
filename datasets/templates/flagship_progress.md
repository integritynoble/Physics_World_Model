# Flagship Modality Template Progress (8 of 12)

**Skipped:** CASSI, CACTI, CT, MRI (per user request)

**Last updated:** 2026-03-21

## Progress Table

| Step | SPC | Lensless | Comp. Holo | Widefield | Ptychography | Cryo-EM | CBCT | Ultrasound |
|------|:---:|:--------:|:----------:|:---------:|:------------:|:-------:|:----:|:----------:|
| 1. Verify Dataset | DONE | DONE | DONE | DONE | DONE | DONE | DONE | DONE |
| 2. List Algorithms / README | DONE (38) | DONE (26) | DONE (14) | DONE (25) | DONE (25) | DONE (25) | DONE (30) | DONE (25) |
| 3. Update Solvers | DONE | DONE | DONE | DONE | DONE | DONE | DONE | DONE |
| 4. Verify (CPU) | DONE 22/22 | DONE 13/13 | DONE 8/8 | DONE 25/25 | DONE 14/14 | DONE 11/11 | DONE 20/20 | DONE 15/15 |
| 4. Verify (GPU) | DONE 16/16 | DONE 13/13 | FAIL 0/6 | DONE 12/12 | PASS 6/11, FAIL 5 | DONE 14/14 | PASS 5/10, FAIL 5 | DONE 10/10 |
| 5. Upload Checkpoints (GCS) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 6. Upload Dataset (GCS) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 7. Push to GitHub | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

## Verification Summary (CPU + GPU)

| Modality | Total | PASS | FAIL | Pass Rate | Best PSNR | Best Solver |
|----------|:-----:|:----:|:----:|:---------:|:---------:|-------------|
| SPC | 38 | 38 | 0 | 100% | 20.75 dB | BM3D-AMP |
| Lensless | 26 | 26 | 0 | 100% | 11.73 dB | ADMM-L1 |
| Comp. Holography | 14 | 8 | 6 | 57% | 27.94 dB | ADMM-TV |
| Widefield | 25 | 25 | 0 | 100% | 25.79 dB | Agard |
| Ptychography | 25 | 20 | 5 | 80% | 9.08 dB | ePIE |
| Cryo-EM | 25 | 25 | 0 | 100% | 17.27 dB | CryoStar |
| CBCT | 30 | 25 | 5 | 83% | 12.11 dB | FBP |
| Ultrasound | 25 | 25 | 0 | 100% | 7.85 dB | US-ViT |
| **Total** | **208** | **192** | **16** | **92.3%** | | |

## GPU Failures (16 total — same root cause)

All 16 failures are DRUNet tensor size mismatch: input image dimensions not divisible by the network's downsampling factor (stride 2^4=16).

| Modality | Failed Solvers | Error |
|----------|----------------|-------|
| Comp. Holography (6) | dl_hologan, dl_deepfresnel, dl_holonet_cs, dl_transformer, best_quality, famous_dl | tensor a (33) vs b (129) |
| Ptychography (5) | pnp_pgd_drunet, physics_nn, ptycho_dv, ptycho_flow, ptycho_foundation | tensor a (33) vs b (129) |
| CBCT (5) | pnp_hqs_drunet, cbct_gan, cbct_transformer, cbct_nerf, cbct_foundation | tensor a (182) vs b (129) |

**Root cause:** DRUNet expects input spatial dimensions divisible by 16 (due to 4-level U-Net). Fix: pad input to nearest multiple of 16 before inference, then crop.

## Known PSNR Issues

| Modality | Issue | Impact | Fix |
|----------|-------|--------|-----|
| Ptychography | fftshift missing in ePIE; probe stored as real (phase lost) | 9 dB vs 21 dB baseline | Add fftshift/ifftshift; reconstruct complex probe |
| CBCT | Solvers use Beer-Lambert inverse instead of proper FBP from sinogram; angles not forwarded | 12 dB vs 23 dB w/ ideal sino | Rewrite solvers to use iradon; forward H_ideal angles |
| Ultrasound | Forward model is depth-dependent PSF + speckle + log compression; deconvolution is inadequate | ~8 dB | Needs envelope-aware inverse model |

## Blockers

| Issue | Affects | Status |
|-------|---------|--------|
| gsutil requires python3.13 (unavailable) | Steps 5-6 (GCS upload) | NOT FIXED |
| Not a git repo | Step 7 (GitHub push) | NOT FIXED |
| DRUNet padding for non-16-divisible images | 16 GPU solvers | Fixable — pad/crop in dl_engine |

## Files Modified

| File | Changes |
|------|---------|
| `datasets/benchmark/verify_benchmark.py` | Added `bmode_measured` fallback for ultrasound; added ptychography metadata forwarding (probe, scan_positions); added H_ideal as angle source for radon family |
| `algorithm_base/ptychography/solvers.py` | Rewrote PtychographyOperator for dual-mode (3D ptycho + 2D PSF); added ePIE core; updated all 25 solvers |
| `algorithm_base/ptychography/README.md` | Complete rewrite: 25 solvers documented |
| `algorithm_base/compressive_holography/README.md` | Created: 14 solvers documented |
| `algorithm_base/spc/README.md` | Updated: 38 solvers documented |
| `algorithm_base/lensless/README.md` | Updated: 26 solvers documented |
| `algorithm_base/cryo_em/README.md` | Updated: 25 solvers documented |
| `algorithm_base/cbct/README.md` | Updated: 30 solvers documented |
| `algorithm_base/ultrasound/README.md` | Updated: 25 solvers documented |
| `algorithm_base/widefield_fluorescence/README.md` | Updated: 25 solvers documented |
| `datasets/benchmark/spc/public/spc_challenge_public.h5` | Created from 20 standard files |
| `datasets/benchmark/compressive_holography/public/compressive_holography_challenge_public.h5` | Created: synthetic 20 samples |
