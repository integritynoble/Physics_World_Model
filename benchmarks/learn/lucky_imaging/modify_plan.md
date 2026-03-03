# Modify Plan — lucky_imaging

## Current State

- **Category:** astronomy
- **Carrier:** Photon
- **Score key:** astronomy
- **Algorithms (from catalog):**
  1. CLEAN (Classical) -- Hogbom, A&AS 1974
  2. AIRI (PnP) -- Terris et al., MNRAS 2022
  3. R2D2 (Deep Learning) -- Aghabiglou et al., ApJS 2024
  4. PRIMO (Deep Learning) -- Medeiros et al., ApJL 2023
- **Leaderboard (live):** CLEAN, AIRI, R2D2, PRIMO (4 entries)

## Assessment

The algorithms are **partially appropriate** but could be improved for Lucky Imaging specifically.

- **CLEAN** is a radio interferometry algorithm (deconvolution of dirty beams from aperture synthesis). Lucky imaging is an optical technique that selects and stacks the sharpest short-exposure frames to beat atmospheric seeing. CLEAN is not typically used in lucky imaging pipelines. A more appropriate classical method would be **Shift-and-Add** or **Drizzle** (Fruchter & Hook 2002).
- **AIRI** is a PnP method for radio interferometric imaging. While PnP methods are general, AIRI specifically targets radio aperture synthesis, not optical lucky imaging.
- **R2D2** is a deep learned reconstruction for radio interferometry. Same concern as AIRI.
- **PRIMO** is another deep learning method specifically for radio/EHT imaging.

Lucky imaging operates in the optical domain with frame selection and registration, not radio interferometric deconvolution. The entire algorithm set is radio-interferometry-focused because the category is "astronomy" and all astronomy algorithms map to radio methods. Lucky imaging would benefit from its own variant override or a sub-category split.

## Recommended Changes

1. **Add a variant override or sub-category** for optical astronomy (lucky imaging, speckle imaging) in `_algorithm_catalog.py`:
   - Classical: **Shift-and-Add** (Bates & Cady 1980)
   - PnP: **PnP-BM3D** or **ADMM-Deblur** (general deblurring PnP)
   - Deep Learning: **DeepLucky** or **Multi-Frame SR-Net** (deep multi-frame super-resolution)
   - Transformer: **Restormer** (Zamir et al., CVPR 2022) for frame deblurring/restoration

2. Alternatively, add `("astronomy", "Photon")` to `_CARRIER_ROUTING` to route optical astronomy to a different pool.

## Verdict

**Changes recommended** -- the current radio-interferometry algorithms (CLEAN, AIRI, R2D2, PRIMO) are not appropriate for optical lucky imaging. The modality needs either a variant override or carrier-based routing to an optical astronomy algorithm pool.
