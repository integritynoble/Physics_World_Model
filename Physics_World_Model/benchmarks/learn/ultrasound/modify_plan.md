# Modify Plan: Ultrasound B-mode

**Created:** 2026-03-03
**Status:** Done (fixed via carrier-based routing)

## Changes

Ultrasound was previously getting CT algorithms (FBP, FBPConvNet) via the generic "medical" category.
Fixed by carrier-based routing: (medical, Acoustic) → medical_ultrasound pool.

Now correctly shows: DAS, PnP-ADMM, ABLE, MU-Net.

No additional changes needed.
