"""Comprehensive verification of all modality algorithms in algorithm_base/.

Run from: /home/spiritai/pwm/Physics_World_Model/pwm/public/
"""
import sys, os, h5py, numpy as np, traceback
sys.path.insert(0, '.')
sys.path.insert(0, 'packages/pwm_core')

def psnr(x, y, maxval=None):
    if maxval is None:
        maxval = max(float(np.abs(y).max()), 1e-8)
    mse = np.mean((np.abs(x.astype(float) - y.astype(float)))**2)
    return float(20*np.log10(maxval / np.sqrt(mse))) if mse > 1e-10 else 100.0


from algorithm_base import list_modalities
mods = sorted(list_modalities())
print(f"Total modalities in algorithm_base: {len(mods)}")

benchmark_base = '/home/spiritai/pwm/Physics_World_Model/Physics_World_Model/datasets/benchmark'
print(f"Using benchmark base: {benchmark_base}")
print()

results = {}
for mod in mods:
    h5_dir = os.path.join(benchmark_base, mod, 'public')
    if not os.path.exists(h5_dir):
        results[mod] = ('skip', 'no_data_dir', None)
        continue
    h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
    if not h5_files:
        results[mod] = ('skip', 'no_h5', None)
        continue

    try:
        with h5py.File(os.path.join(h5_dir, h5_files[0]), 'r') as f:
            # Find sample key
            sample_keys = sorted([k for k in f.keys() if k.startswith('sample')])
            if not sample_keys:
                results[mod] = ('skip', 'no_samples', None)
                continue
            s = f[sample_keys[0]]
            available_keys = list(s.keys())

            x_true = np.array(s['x_true']).astype(np.float32)
            baseline = (np.array(s['reconstruction_baseline']).astype(np.float32)
                        if 'reconstruction_baseline' in s else None)

            # Special handling per modality
            if mod == 'ct' and 'sinogram_ideal' in s:
                y = np.array(s['sinogram_ideal']).astype(np.float32)
            elif mod == 'mri' and 'kspace_undersampled' in s:
                y = np.array(s['kspace_undersampled'])  # keep complex
            elif 'y' in s:
                y = np.array(s['y']).astype(np.float32)
            else:
                results[mod] = ('skip', f'no_y (keys: {available_keys})', None)
                continue

        from algorithm_base import run_solver
        x_hat = run_solver(mod, 'traditional_cpu', y)
        x_hat = np.asarray(x_hat, dtype=np.float32)

        if x_hat.shape != x_true.shape:
            results[mod] = ('shape_mismatch', f'{x_hat.shape} vs {x_true.shape}', None)
            print(f"  SHAPE {mod:40s}  {x_hat.shape} vs {x_true.shape}")
        else:
            ref = baseline if baseline is not None else x_true
            maxval = max(float(np.abs(ref).max()), 1e-8)
            p = psnr(x_hat, ref, maxval=maxval)
            results[mod] = ('ok', '', p)
            print(f"  OK    {mod:40s}  PSNR={p:.1f}dB  shape={x_hat.shape}")
    except Exception as e:
        short_msg = str(e)[:120]
        results[mod] = ('error', short_msg, None)
        print(f"  ERR   {mod:40s}  {short_msg}")

# ── Summary ──────────────────────────────────────────────────────────────────
ok = [(m, v) for m, v in results.items() if v[0] == 'ok']
errors = [(m, v) for m, v in results.items() if v[0] == 'error']
shape_mismatch = [(m, v) for m, v in results.items() if v[0] == 'shape_mismatch']
skipped = [(m, v) for m, v in results.items() if v[0] == 'skip']

print()
print("=" * 70)
print(f"PASS: {len(ok)}, ERRORS: {len(errors)}, SHAPE_MISMATCH: {len(shape_mismatch)}, SKIPPED: {len(skipped)}")
print("=" * 70)

if shape_mismatch:
    print()
    print("=== SHAPE MISMATCH ===")
    for m, (s, msg, p) in sorted(shape_mismatch):
        print(f"  {m}: {msg}")

if errors:
    print()
    print("=== ERRORS (full list) ===")
    for m, (s, msg, p) in sorted(errors):
        print(f"  {m}: {msg}")

skip_no_data = [(m, v) for m, v in skipped if v[1] == 'no_data_dir']
skip_no_y    = [(m, v) for m, v in skipped if 'no_y' in v[1]]
skip_no_h5   = [(m, v) for m, v in skipped if v[1] == 'no_h5']
skip_other   = [(m, v) for m, v in skipped if v[1] not in ('no_data_dir', 'no_h5') and 'no_y' not in v[1]]
print()
print(f"=== SKIPPED BREAKDOWN ===")
print(f"  no data dir: {len(skip_no_data)}")
print(f"  no h5 file:  {len(skip_no_h5)}")
print(f"  no y key:    {len(skip_no_y)}")
print(f"  other:       {len(skip_other)}")
if skip_no_y:
    print("  Modalities skipped (no y):")
    for m, (s, msg, p) in sorted(skip_no_y):
        print(f"    {m}: {msg}")
if skip_other:
    print("  Modalities skipped (other):")
    for m, (s, msg, p) in sorted(skip_other):
        print(f"    {m}: {msg}")

if ok:
    psnr_vals = [v[2] for _, v in ok if v[2] is not None]
    print()
    print(f"=== PASSING: {len(ok)} modalities ===")
    print(f"    median PSNR = {np.median(psnr_vals):.1f} dB")
    print(f"    mean   PSNR = {np.mean(psnr_vals):.1f} dB")
    print(f"    min    PSNR = {np.min(psnr_vals):.1f} dB")
    print(f"    max    PSNR = {np.max(psnr_vals):.1f} dB")
    print()
    print("Top 30 by PSNR:")
    for m, (s, msg, p) in sorted(ok, key=lambda x: x[1][2] or 0, reverse=True)[:30]:
        print(f"  {m:45s} PSNR={p:.1f}dB")
    print()
    print("Bottom 20 by PSNR:")
    for m, (s, msg, p) in sorted(ok, key=lambda x: x[1][2] or 0)[:20]:
        print(f"  {m:45s} PSNR={p:.1f}dB")
