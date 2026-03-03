# Modify Plan: SPECT

**Created:** 2026-03-03
**Status:** Done (fixed via carrier-based routing)

## Changes

SPECT was previously getting CT algorithms (FBP, FBPConvNet) via the generic "medical" category.
Fixed by carrier-based routing: (medical, Gamma) → particle_imaging pool.

Now correctly shows: OSEM, MAPEM-RDP, DeepPET, TransEM.

No additional changes needed.
