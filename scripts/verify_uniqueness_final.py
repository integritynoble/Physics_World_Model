"""Verify all standard datasets have unique ground truth (x_true) data."""
import hashlib, os, h5py, glob
from collections import defaultdict

BASE = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark"

hashes = {}
missing = []
hash_to_mods = defaultdict(list)

for mod in sorted(os.listdir(BASE)):
    std_dir = os.path.join(BASE, mod, "standard")
    if not os.path.isdir(std_dir):
        continue
    # Find first h5 file (sample 00)
    h5_files = sorted(glob.glob(os.path.join(std_dir, "*.h5")))
    if not h5_files:
        missing.append(mod)
        continue
    h5_path = h5_files[0]  # first sample
    try:
        with h5py.File(h5_path, "r") as f:
            # Try different key patterns
            x_key = None
            for k in ["x_true", "sample_00/x_true"]:
                if k in f:
                    x_key = k
                    break
            # Also check top-level groups
            if x_key is None:
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        x_key = k
                        break
                    elif isinstance(f[k], h5py.Group) and "x_true" in f[k]:
                        x_key = f"{k}/x_true"
                        break
            if x_key is None:
                print(f"  WARNING: {mod} — no x_true found, keys={list(f.keys())}")
                missing.append(mod)
                continue
            data = f[x_key][:]
            h = hashlib.sha256(data.tobytes()).hexdigest()[:16]
            hashes[mod] = h
            hash_to_mods[h].append(mod)
    except Exception as e:
        print(f"  ERROR reading {mod}: {e}")
        missing.append(mod)

print(f"Total modalities with standard datasets: {len(hashes)}")
print(f"Unique x_true hashes: {len(hash_to_mods)}")
print()

dups = {h: mods for h, mods in hash_to_mods.items() if len(mods) > 1}
if dups:
    print("=== DUPLICATES FOUND ===")
    for h, mods in sorted(dups.items(), key=lambda x: -len(x[1])):
        print(f"  Hash {h}: {', '.join(mods)}")
    print(f"\nTotal duplicate groups: {len(dups)}")
else:
    print("ALL UNIQUE — no duplicate ground truth data found!")

if missing:
    print(f"\nMissing/unreadable ({len(missing)}): {', '.join(missing)}")
