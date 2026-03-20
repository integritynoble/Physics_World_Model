#!/usr/bin/env python3
"""Recover MiJUN checkpoint by rebuilding ZIP from all local file headers.

The file reference/cassi/our_mamba_5stg.pth has 98004 bytes prepended before
the ZIP archive. The central directory is truncated (only 265 of ~1655 entries).
But ALL tensor data exists as local file entries in the ZIP body.
This script rebuilds a proper ZIP archive from scratch.
"""
import struct
import os
import sys
import io
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg.pth")
DST = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_recovered.pth")

ZIP_START = 98004  # ZIP archive starts at this offset in the original file

def parse_local_file_header(data, offset):
    """Parse a ZIP local file header at the given offset in data.
    Returns (filename, compressed_data, compress_method, uncompressed_size, crc32, total_size) or None.
    """
    if offset + 30 > len(data):
        return None
    sig = struct.unpack_from('<I', data, offset)[0]
    if sig != 0x04034b50:  # PK\x03\x04
        return None

    (version, flags, method, mod_time, mod_date, crc, comp_size, uncomp_size,
     fname_len, extra_len) = struct.unpack_from('<HHHHHIBBIH', data, offset + 4)

    # Re-read with proper field sizes
    (version, flags, method, mod_time, mod_date, crc, comp_size, uncomp_size,
     fname_len, extra_len) = struct.unpack_from('<HHHHHIIIHH', data, offset + 4)

    header_end = offset + 30 + fname_len + extra_len
    if header_end > len(data):
        return None

    filename = data[offset + 30: offset + 30 + fname_len]
    try:
        filename_str = filename.decode('utf-8')
    except:
        return None

    data_start = header_end

    # Handle data descriptor (bit 3 of flags)
    if flags & 0x08:
        # Data descriptor follows - comp_size and crc are 0 in header
        # Need to find the data descriptor after the compressed data
        # For stored files (method 0), we can try to find next PK signature
        if method == 0 and comp_size == 0:
            # Search for next local file header or central directory
            next_pk = data_start
            while next_pk < len(data) - 4:
                if data[next_pk:next_pk+4] == b'PK\x03\x04' or data[next_pk:next_pk+4] == b'PK\x01\x02':
                    break
                next_pk += 1
            # Check for data descriptor before next PK
            # Data descriptor: optional sig (0x08074b50) + crc + comp_size + uncomp_size (12 or 16 bytes)
            if next_pk >= data_start + 16:
                dd_start = next_pk - 12
                if struct.unpack_from('<I', data, dd_start - 4)[0] == 0x08074b50:
                    dd_start -= 4
                comp_size = next_pk - data_start - 12
                if struct.unpack_from('<I', data, next_pk - 16)[0] == 0x08074b50:
                    comp_size = next_pk - data_start - 16
            else:
                comp_size = 0
        elif comp_size > 0:
            pass  # comp_size was actually set
        else:
            # Can't determine size
            return None

    if data_start + comp_size > len(data):
        # Try to recover - truncated entry
        comp_size = min(comp_size, len(data) - data_start)

    file_data = data[data_start: data_start + comp_size]
    total_entry_size = 30 + fname_len + extra_len + comp_size

    # If data descriptor present, add its size
    if flags & 0x08:
        total_entry_size += 16  # with signature

    return {
        'filename': filename_str,
        'data': file_data,
        'method': method,
        'uncomp_size': uncomp_size if uncomp_size > 0 else comp_size,
        'comp_size': comp_size,
        'crc': crc,
        'flags': flags,
        'offset': offset,
        'total_size': total_entry_size,
    }


def scan_all_local_headers(data):
    """Find all local file headers by scanning for PK\\x03\\x04 signatures."""
    entries = []
    pos = 0
    while pos < len(data) - 4:
        idx = data.find(b'PK\x03\x04', pos)
        if idx < 0:
            break
        entry = parse_local_file_header(data, idx)
        if entry and entry['filename']:
            entries.append(entry)
            # Move past this entry
            pos = idx + max(entry['total_size'], 1)
        else:
            pos = idx + 1
    return entries


def rebuild_zip(entries, output_path):
    """Rebuild a ZIP archive from parsed local file entries."""
    # Use zipfile module to create a clean archive
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        seen = set()
        for entry in entries:
            fname = entry['filename']
            if fname in seen:
                continue
            seen.add(fname)

            file_data = entry['data']

            # If compressed, we need to decompress first
            if entry['method'] == 8:  # deflate
                import zlib
                try:
                    file_data = zlib.decompress(file_data, -15)
                except zlib.error:
                    try:
                        file_data = zlib.decompress(file_data)
                    except:
                        print(f"  WARNING: Could not decompress {fname}, using raw data")

            zf.writestr(fname, file_data)

    return output_path


def main():
    print(f"Reading {SRC} ({os.path.getsize(SRC)} bytes)")
    with open(SRC, 'rb') as f:
        raw = f.read()

    # Extract ZIP portion
    zip_data = raw[ZIP_START:]
    print(f"ZIP data starts at offset {ZIP_START}, length {len(zip_data)}")

    # Scan all local file headers
    print("Scanning for local file headers...")
    entries = scan_all_local_headers(zip_data)
    print(f"Found {len(entries)} local file entries")

    # Categorize entries
    filenames = [e['filename'] for e in entries]
    unique_names = sorted(set(filenames))
    print(f"Unique filenames: {len(unique_names)}")

    # Show some examples
    data_entries = [f for f in unique_names if f.startswith('model_epoch_200/data/')]
    archive_entries = [f for f in unique_names if f.startswith('model_epoch_200/archive/')]
    other = [f for f in unique_names if not f.startswith('model_epoch_200/')]

    print(f"  data/ entries: {len(data_entries)}")
    print(f"  archive/ entries: {len(archive_entries)}")
    print(f"  other entries: {len(other)}")
    if other:
        for o in other[:5]:
            print(f"    {o}")

    # Check for required PyTorch files
    required = ['model_epoch_200/data.pkl', 'model_epoch_200/archive/data.pkl',
                 'model_epoch_200/archive/version']
    for r in required:
        if r in unique_names:
            print(f"  FOUND: {r}")
        else:
            print(f"  MISSING: {r}")

    # Show data index range
    if data_entries:
        indices = []
        for d in data_entries:
            try:
                idx = int(d.split('/')[-1])
                indices.append(idx)
            except:
                pass
        if indices:
            print(f"  Data indices: {min(indices)}-{max(indices)}, count={len(indices)}")
            missing = set(range(min(indices), max(indices)+1)) - set(indices)
            if missing:
                print(f"  Missing indices: {sorted(missing)[:20]}...")

    # Rebuild ZIP
    print(f"\nRebuilding ZIP archive to {DST}...")
    rebuild_zip(entries, DST)

    fsize = os.path.getsize(DST)
    print(f"Written {DST} ({fsize} bytes)")

    # Verify
    print("\nVerifying recovered archive...")
    try:
        zf = zipfile.ZipFile(DST, 'r')
        names = zf.namelist()
        print(f"  Valid ZIP with {len(names)} entries")

        # Check key files
        for r in ['model_epoch_200/data.pkl']:
            if r in names:
                data = zf.read(r)
                print(f"  {r}: {len(data)} bytes")

        # Count data entries
        data_names = [n for n in names if n.startswith('model_epoch_200/data/')]
        print(f"  data/ entries in rebuilt ZIP: {len(data_names)}")

        zf.close()
    except Exception as e:
        print(f"  ZIP verification failed: {e}")

    # Try torch.load
    print("\nAttempting torch.load...")
    try:
        import torch
        state = torch.load(DST, map_location='cpu', weights_only=False)
        if isinstance(state, dict):
            keys = list(state.keys())
            print(f"  Loaded! Top-level keys: {keys[:5]}")
            total_params = sum(v.numel() for v in state.values() if hasattr(v, 'numel'))
            print(f"  Total parameters: {total_params:,}")
        else:
            print(f"  Loaded type: {type(state)}")
    except Exception as e:
        print(f"  torch.load failed: {e}")

        # Try loading data.pkl directly
        print("\n  Trying manual unpickle of data.pkl...")
        try:
            import pickle
            zf = zipfile.ZipFile(DST, 'r')
            pkl_data = zf.read('model_epoch_200/data.pkl')

            # Create a custom unpickler that can load tensors
            import io

            class TensorLoader:
                def __init__(self, zipf):
                    self.zipf = zipf

                def get_data(self, key):
                    path = f'model_epoch_200/data/{key}'
                    return self.zipf.read(path)

            loader = TensorLoader(zf)

            # First just unpickle to see the structure
            result = pickle.loads(pkl_data)
            print(f"  data.pkl unpickled: type={type(result)}")
            if hasattr(result, 'keys'):
                keys = list(result.keys())
                print(f"  Keys ({len(keys)}): {keys[:10]}...")
            zf.close()
        except Exception as e2:
            print(f"  Manual unpickle failed: {e2}")


if __name__ == '__main__':
    main()
