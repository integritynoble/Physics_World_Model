#!/usr/bin/env python3
"""Recover MiJUN checkpoint v2: precise entry-gap based recovery.

The original our_mamba_5stg.pth has 98004 bytes prepended, and the ZIP
archive uses data descriptors (flag bit 3) so all local file headers
have comp_sz=uncomp_sz=0. We compute actual sizes from gaps between
consecutive local file headers, accounting for data descriptors.
"""
import struct
import os
import io
import zipfile
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "reference", "our_mamba_5stg.pth")
DST = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_v2.pth")

ZIP_START = 98004


def scan_local_headers(data):
    """Find all PK\\x03\\x04 local file headers and parse metadata."""
    entries = []
    pos = 0
    while pos < len(data) - 4:
        idx = data.find(b'PK\x03\x04', pos)
        if idx < 0:
            break

        if idx + 30 > len(data):
            break

        (ver, flags, method, mod_time, mod_date, crc,
         comp_sz, uncomp_sz, fname_len, extra_len) = struct.unpack_from(
            '<HHHHHIIIHH', data, idx + 4)

        header_end = idx + 30 + fname_len + extra_len
        if header_end > len(data):
            break

        fname = data[idx + 30: idx + 30 + fname_len].decode('utf-8', errors='replace')

        entries.append({
            'offset': idx,
            'fname': fname,
            'method': method,
            'flags': flags,
            'crc': crc,
            'comp_sz': comp_sz,
            'uncomp_sz': uncomp_sz,
            'fname_len': fname_len,
            'extra_len': extra_len,
            'data_offset': header_end,
        })

        pos = idx + 1  # move past this PK signature to find next

    return entries


def compute_data_sizes(entries, total_len):
    """Compute actual data sizes from gaps between consecutive entries.

    For data descriptor entries (flags & 0x08), the layout is:
    [local_file_header][file_data][data_descriptor]

    data_descriptor is either:
      - PK\\x07\\x08 + crc32 + comp_sz + uncomp_sz (16 bytes)
      - crc32 + comp_sz + uncomp_sz (12 bytes, no signature)
    """
    for i in range(len(entries)):
        if entries[i]['comp_sz'] > 0:
            # Size already known
            entries[i]['actual_data_end'] = entries[i]['data_offset'] + entries[i]['comp_sz']
            continue

        data_start = entries[i]['data_offset']

        if i + 1 < len(entries):
            next_offset = entries[i + 1]['offset']
        else:
            next_offset = total_len

        # The gap contains: file_data + data_descriptor (if flags & 0x08)
        gap = next_offset - data_start

        if entries[i]['flags'] & 0x08:
            # Check if data descriptor has signature
            # Try 16-byte descriptor first (with PK\x07\x08 sig)
            if gap >= 16:
                dd_start_16 = next_offset - 16
                dd_sig = struct.unpack_from('<I', entries[i:i+1] and b'\x00'*4 or
                                           b'\x00'*4, 0)[0]
                # Check for PK\x07\x08 signature at (next_offset - 16)
                if dd_start_16 >= data_start:
                    maybe_sig = struct.unpack_from('<I',
                        entries[0]['offset'].__class__.__mro__[0] and b'\x00'*4 or b'\x00'*4, 0)
                    # Simpler: just read from the raw data
                    pass

            # Use gap - 16 if PK\x07\x08 present, else gap - 12
            # Actually let's just check the raw bytes
            data_size_16 = gap - 16  # assuming 16-byte data descriptor
            data_size_12 = gap - 12  # assuming 12-byte data descriptor

            entries[i]['data_size_16'] = data_size_16
            entries[i]['data_size_12'] = data_size_12
        else:
            entries[i]['data_size_16'] = gap
            entries[i]['data_size_12'] = gap

    return entries


def main():
    print(f"Reading {SRC} ({os.path.getsize(SRC)} bytes)")
    with open(SRC, 'rb') as f:
        raw = f.read()

    zip_data = raw[ZIP_START:]
    total_len = len(zip_data)
    print(f"ZIP data: {total_len} bytes (offset {ZIP_START})")

    # Scan all local file headers
    entries = scan_local_headers(zip_data)
    print(f"Found {len(entries)} local file headers")

    # Determine data descriptor format by checking first entry
    # data.pkl is the first entry - we know its approximate size
    # The pickle for a 305-key state dict is typically ~5-100KB
    e0 = entries[0]
    e1 = entries[1]
    gap0 = e1['offset'] - e0['data_offset']
    print(f"First entry ({e0['fname']}): gap={gap0} bytes")

    # Check for PK\x07\x08 (data descriptor signature) before next entry
    # 16-byte DD: PK\x07\x08 (4) + crc32 (4) + comp_sz (4) + uncomp_sz (4)
    # 12-byte DD: crc32 (4) + comp_sz (4) + uncomp_sz (4)
    dd_check_16 = e1['offset'] - 16
    dd_check_12 = e1['offset'] - 12

    if dd_check_16 >= e0['data_offset']:
        sig_at_16 = struct.unpack_from('<I', zip_data, dd_check_16)[0]
        if sig_at_16 == 0x08074b50:
            dd_size = 16
            print("Data descriptor format: 16 bytes (with PK\\x07\\x08 signature)")
        else:
            dd_size = 12
            print("Data descriptor format: 12 bytes (no signature)")
    else:
        dd_size = 0
        print("No data descriptor")

    # Verify by checking data descriptor fields
    if dd_size == 16:
        crc, csz, usz = struct.unpack_from('<III', zip_data, dd_check_16 + 4)
        file_data_size = gap0 - 16
        print(f"  DD: crc=0x{crc:08x}, comp_sz={csz}, uncomp_sz={usz}, computed_file_sz={file_data_size}")
        # The comp_sz in DD should match the computed file size
        if csz == file_data_size:
            print("  DD comp_sz matches computed size!")
        else:
            print(f"  DD comp_sz MISMATCH: DD says {csz}, computed {file_data_size}")
    elif dd_size == 12:
        crc, csz, usz = struct.unpack_from('<III', zip_data, dd_check_12)
        file_data_size = gap0 - 12
        print(f"  DD: crc=0x{crc:08x}, comp_sz={csz}, uncomp_sz={usz}, computed_file_sz={file_data_size}")
        if csz == file_data_size:
            print("  DD comp_sz matches computed size!")

    # Now rebuild the ZIP with correct sizes
    print(f"\nRebuilding ZIP with dd_size={dd_size}...")

    with zipfile.ZipFile(DST, 'w', compression=zipfile.ZIP_STORED) as zf:
        seen = set()
        for i, entry in enumerate(entries):
            fname = entry['fname']
            if fname in seen:
                continue
            seen.add(fname)

            data_start = entry['data_offset']

            # Compute actual data size
            if i + 1 < len(entries):
                next_start = entries[i + 1]['offset']
            else:
                next_start = total_len

            # Check for data descriptor at end
            if entry['flags'] & 0x08 and dd_size > 0:
                file_data_size = next_start - data_start - dd_size
            else:
                file_data_size = next_start - data_start

            if file_data_size < 0:
                print(f"  WARNING: negative data size for {fname}, skipping")
                continue

            file_data = zip_data[data_start: data_start + file_data_size]

            # Decompress if needed
            if entry['method'] == 8:  # deflate
                import zlib
                try:
                    file_data = zlib.decompress(file_data, -15)
                except:
                    pass

            zf.writestr(fname, file_data)

    fsize = os.path.getsize(DST)
    print(f"Written {DST} ({fsize:,} bytes)")

    # Verify with torch.load
    print("\nVerifying with torch.load...")
    try:
        import torch
        state = torch.load(DST, map_location='cpu', weights_only=False)
        if isinstance(state, dict):
            print(f"Keys: {list(state.keys())}")
            if 'model' in state:
                sd = state['model']
                n_keys = len(sd)
                n_params = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
                print(f"Model: {n_keys} keys, {n_params:,} params")

                # Check DP.mlp output
                if 'DP.mlp.4.bias' in sd:
                    print(f"DP.mlp.4.bias: {sd['DP.mlp.4.bias']}")
        print("SUCCESS!")
    except Exception as e:
        print(f"torch.load FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
