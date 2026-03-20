# Solver Summary — PWM5 Benchmark

Generated: 2026-03-16

## Quick Reference

### Usage
```bash
# CLI usage
pwm evaluate --method traditional_cpu --modality cassi --track correct

# Python usage
from pwm_core.recon.gap_tv import run_gap_tv
x_hat = run_gap_tv(y, operator, {'iterations': 50, 'lam': 0.05})

# Modal GPU usage (for gpu=true solvers)
# Deploy solver function to Modal with T4 GPU
# See benchmarks/modal/ for Modal wrapper templates
```

## Per-Modality Solver Table

| Modality | Solver Key | Name | Module | Function | GPU | Importable |
|----------|------------|------|--------|----------|-----|------------|
| acoustic_emission | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| acoustic_emission | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| acoustic_emission | dl_localizer | DeepAE-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| acoustic_microscopy | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| acoustic_microscopy | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| acoustic_microscopy | saft_dl | SAFT-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| active_thermography | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| active_thermography | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| active_thermography | pulsed_phase_tv | Pulsed-Phase TV [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| adaptive_optics | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| adaptive_optics | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| adaptive_optics | deep_ao | Deep-AO [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| afm | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| afm | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| afm | afm_dl | AFM-UNet | pwm_core.recon.afm_solvers | afm_unet_recon | yes | yes |
| angiography | traditional_cpu | FBP (DSA baseline) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| angiography | best_quality | DSA-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| angiography | famous_dl | VesselSegNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| asl_mri | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| asl_mri | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| asl_mri | asl_dl | ASL-Net [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| atom_probe | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| atom_probe | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| atom_probe | apt_dl | APT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| bioluminescence_tomo | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| bioluminescence_tomo | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| bioluminescence_tomo | blt_dl | BLT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| brachytherapy_img | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| brachytherapy_img | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| brachytherapy_img | brachy_dl | BrachyNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| brillouin | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| brillouin | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| brillouin | brillouin_dl | Brillouin-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cacti | traditional_cpu | GAP-TV | pwm_core.recon.gap_tv | run_gap_tv | no | yes |
| cacti | best_quality | EfficientSCI | pwm_core.recon.efficientsci | run_efficientsci | no | yes |
| cacti | famous_dl | ELP-Unfolding | pwm_core.recon.elp_unfolding | run_elp_unfolding | no | yes |
| cacti | small_gpu | EfficientSCI-T | pwm_core.recon.efficientsci | run_efficientsci | no | yes |
| cacti | pnp_ffdnet | PnP-FFDNet | pwm_core.recon.cacti_solvers | pnp_ffdnet_cacti | no | yes |
| cacti | hisvit9 | HiSViT-9 | pwm_core.recon.cacti_solvers | hisvit_cacti | yes | yes |
| cacti | hisvit13 | HiSViT-13 | pwm_core.recon.cacti_solvers | hisvit_cacti | yes | yes |
| cars | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cars | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cars | cars_dl | CARS-DeepSpec [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cassi | traditional_cpu | GAP-TV | pwm_core.recon.gap_tv | run_gap_tv | no | yes |
| cassi | best_quality | GAP-TV (guided) | pwm_core.recon.gap_tv | run_gap_tv | no | yes |
| cassi | famous_dl | GAP-TV (fast) | pwm_core.recon.gap_tv | run_gap_tv | no | yes |
| cassi | small_gpu | GAP-TV (small) | pwm_core.recon.gap_tv | run_gap_tv | no | yes |
| cassi | mst_l | MST-L | pwm_core.recon.mst | mst_recon_cassi | no | yes |
| cassi | hdnet | HDNet | pwm_core.recon.hdnet | run_hdnet | no | yes |
| cassi | hsi_sdecnn | HSI-SDeCNN | pwm_core.recon.hsi_sdecnn | run_hsi_sdecnn | no | yes |
| cathodoluminescence | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cathodoluminescence | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cathodoluminescence | cl_dl | CL-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cbct | traditional_cpu | FDK / FBP | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| cbct | best_quality | FDK-DL | pwm_core.recon.cbct_solvers | fdk_dl_recon | yes | yes |
| cbct | famous_dl | CBCT-UNet | pwm_core.recon.cbct_solvers | cbct_unet_recon | yes | yes |
| cest_mri | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cest_mri | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cest_mri | cest_dl | CEST-Net [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| ceus | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ceus | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ceus | us_dl_enhance | US-DeepSight [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| clem | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| clem | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| clem | clem_dl | CLEM-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| coded_exposure | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| coded_exposure | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| coded_exposure | coded_dl | FlowNet-Coded [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| confocal_3d | traditional_cpu | 3D Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| confocal_3d | best_quality | 3D CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| confocal_3d | famous_dl | CARE-3D | pwm_core.recon.care_unet | run_care | no | yes |
| confocal_3d | small_gpu | CARE-3D (slice-wise) | pwm_core.recon.care_unet | run_care | no | yes |
| confocal_endomicroscopy | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| confocal_endomicroscopy | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| confocal_endomicroscopy | cle_dl | CLE-Net (CARE) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| confocal_livecell | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| confocal_livecell | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| confocal_livecell | famous_dl | CARE | pwm_core.recon.care_unet | run_care | no | yes |
| confocal_livecell | small_gpu | CARE | pwm_core.recon.care_unet | run_care | no | yes |
| coronagraphy | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| coronagraphy | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| coronagraphy | speckle_null_dl | DL-SpeckleNull [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cryo_em | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cryo_em | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cryo_em | relion_dl | CryoDRGN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cryo_et | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cryo_et | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| cryo_et | cryo_et_dl | CryoCARE | pwm_core.recon.cryoet_solvers | cryocare_recon | yes | yes |
| ct | traditional_cpu | FBP | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| ct | best_quality | PnP-HQS + NLM | pwm_core.recon.pnp | run_pnp | no | yes |
| ct | famous_dl | RED-CNN | pwm_core.recon.redcnn | run_redcnn | no | yes |
| ct | small_gpu | RED-CNN | pwm_core.recon.redcnn | run_redcnn | no | yes |
| ct_fluorescence | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ct_fluorescence | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ct_fluorescence | xfct_dl | XFCT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cup | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cup | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| cup | e2e_cup | E2E-CUP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dark_field | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dark_field | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| dark_field | df_unet | DF-UNet | pwm_core.recon.darkfield_solvers | df_unet_recon | yes | yes |
| desi | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| desi | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| desi | desi_dl | DESI-SegNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dexa | traditional_cpu | FISTA-L2 (dual-energy) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| dexa | best_quality | DXA-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dexa | famous_dl | DEXA-UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dic | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dic | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| dic | dic_dl | DIC-Net | pwm_core.recon.dic_solvers | dic_dl_recon | yes | yes |
| diffusion_mri | traditional_cpu | SENSE (WLS tensor fit) | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| diffusion_mri | best_quality | q-DL (qDiffusion) [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| diffusion_mri | famous_dl | SHORE-Net [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| digital_breast_tomo | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| digital_breast_tomo | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| digital_breast_tomo | dbt_dl | DBT-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dna_paint | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dna_paint | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| dna_paint | dna_paint_dl | DECODE-PAINT | pwm_core.recon.smlm_solvers | decode_smlm_recon | yes | yes |
| doppler_ultrasound | traditional_cpu | Back-Projection (Doppler) | pwm_core.recon.photoacoustic_solver | run_photoacoustic | no | yes |
| doppler_ultrasound | best_quality | UDoppler-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| doppler_ultrasound | famous_dl | Doppler CFAR [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dot | traditional_cpu | Born Approximation | pwm_core.recon.dot_solver | run_dot | no | yes |
| dot | best_quality | L-BFGS-TV [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| dot | dot_dl | DOT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ebsd | traditional_cpu | FISTA-L2 (Hough baseline) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| ebsd | best_quality | EBSD-DL (DictIndex) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ebsd | famous_dl | EMsoft-EBSD [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eddy_current | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eddy_current | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eddy_current | ec_dl | ECT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| edx_mapping | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| edx_mapping | best_quality | Richardson-Lucy (high quality) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| edx_mapping | edx_dl | Richardson-Lucy (DL baseline) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eels | traditional_cpu | FISTA-L2 (Fourier ratio) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eels | best_quality | EELS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eels | famous_dl | MLLS-EELS [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eels | eels_dl | EELS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eht_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eht_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| eht_imaging | eht_dl | EHT-PRIMO [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| elastography | traditional_cpu | SENSE (displacement field) | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| elastography | best_quality | MRE-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| elastography | famous_dl | NLSI-Solver [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| electron_diffraction | traditional_cpu | ePIE (electron ptychography) | pwm_core.recon.ptychography_solver | run_epie | no | yes |
| electron_diffraction | best_quality | ED-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| electron_diffraction | famous_dl | CRISP-ED [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| electron_holography | traditional_cpu | Phase Retrieval (HIO) | pwm_core.recon.phase_retrieval_solver | run_phase_retrieval | no | yes |
| electron_holography | best_quality | EH-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| electron_holography | famous_dl | Phase-Sideband [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| electron_tomography | traditional_cpu | FBP (SIRT baseline) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| electron_tomography | best_quality | IMOD-SIRT-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| electron_tomography | famous_dl | SIRT-3D [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| endoscopy | traditional_cpu | FISTA-L2 (endoscopy) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| endoscopy | best_quality | EndoMapper-Net | pwm_core.recon.endoscopy_solvers | endomapper_recon | yes | yes |
| endoscopy | famous_dl | AF-SfMLearner | pwm_core.recon.endoscopy_solvers | af_sfm_learner_recon | yes | yes |
| entangled_photon | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| entangled_photon | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| entangled_photon | qgi_dl | QGI-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| event_camera | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| event_camera | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| event_camera | event_dl | E2VID+ [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| expansion | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| expansion | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| expansion | exm_dl | EXpansionNet | pwm_core.recon.expansion_solvers | expansion_dl_recon | yes | yes |
| fib_sem | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fib_sem | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| fib_sem | fibsem_dl | FIB-SEM-Net | pwm_core.recon.fibsem_solvers | fibsem_dl_recon | yes | yes |
| flash_lidar | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| flash_lidar | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| flash_lidar | flash_dl | FlashLiDAR-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| flim | traditional_cpu | Phasor Analysis | pwm_core.recon.flim_solver | run_flim | no | yes |
| flim | best_quality | MLE Fit | pwm_core.recon.flim_solver | run_flim | no | yes |
| flim | famous_dl | MLE Fit (iterative) | pwm_core.recon.flim_solver | run_flim | no | yes |
| flim | small_gpu | Phasor Analysis | pwm_core.recon.flim_solver | run_flim | no | yes |
| fluoroscopy | traditional_cpu | FBP (fluoroscopy) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| fluoroscopy | best_quality | FluoroNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fluoroscopy | famous_dl | X-ray CNN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fmri | traditional_cpu | SENSE (fMRI) | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| fmri | best_quality | fMRI-Transformer [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| fmri | famous_dl | DeepBold [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| fpm | traditional_cpu | Sequential Phase Retrieval | pwm_core.recon.fpm_solver | run_fpm | no | yes |
| fpm | best_quality | Gradient Descent FPM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fpm | famous_dl | Fourier Ptychnet | pwm_core.recon.fpm_solver | run_fpm | no | yes |
| fpm | small_gpu | Fourier Ptychnet | pwm_core.recon.fpm_solver | run_fpm | no | yes |
| ftir_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ftir_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ftir_imaging | ftir_dl | FTIR-UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fundus | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fundus | best_quality | RETFound | pwm_core.recon.fundus_solvers | retfound_recon | yes | yes |
| fundus | famous_dl | DR-Grade-Net | pwm_core.recon.fundus_solvers | dr_grade_net_recon | yes | yes |
| fwi | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fwi | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| fwi | fwi_dl | InversionNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gaussian_splatting | traditional_cpu | EWA Splatting | pwm_core.recon.gaussian_splatting_solver | run_gaussian_splatting | no | yes |
| gaussian_splatting | best_quality | 3DGS (full) | pwm_core.recon.gaussian_splatting_solver | run_gaussian_splatting | yes | yes |
| gaussian_splatting | famous_dl | NeRF (baseline comparison) | pwm_core.recon.nerf_solver | run_nerf | no | yes |
| gaussian_splatting | small_gpu | 3DGS (compact) | pwm_core.recon.gaussian_splatting_solver | run_gaussian_splatting | no | yes |
| ghost_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ghost_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ghost_imaging | gi_dl | GI-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gpr | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gpr | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gpr | gpr_dl | GPR-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gravitational_wave | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gravitational_wave | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| gravitational_wave | matched_filter_dl | GW-DL (PyCBC-ML) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| hdr_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| hdr_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| hdr_imaging | hdr_dl | HDR-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| holography | traditional_cpu | Angular Spectrum | pwm_core.recon.holography_solver | run_holography_reconstruction | no | yes |
| holography | best_quality | PhaseNet | pwm_core.recon.phasenet | run_phasenet | no | yes |
| holography | famous_dl | PhaseNet | pwm_core.recon.phasenet | run_phasenet | no | yes |
| holography | small_gpu | PhaseNet | pwm_core.recon.phasenet | run_phasenet | no | yes |
| hyperspectral_remote | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| hyperspectral_remote | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| hyperspectral_remote | hyper_dl | SST-USRNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| impedance_tomo | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| impedance_tomo | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| impedance_tomo | eit_dl | EIT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| industrial_ct | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| industrial_ct | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| industrial_ct | ict_dl | IndustrialCT-Net [proxy] | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| insar | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| insar | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| insar | insar_dl | InSAR-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| integral | traditional_cpu | Depth Estimation | pwm_core.recon.integral_solver | run_integral | no | yes |
| integral | best_quality | DIBR [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| integral | famous_dl | EPINet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| integral | small_gpu | EPINet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ism | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ism | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| ism | ism_dl | ISM-Reassignment-Net | pwm_core.recon.ism_solvers | ism_dl_recon | yes | yes |
| ivus | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ivus | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ivus | ivus_dl | IVUS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| lattice_lightsheet | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| lattice_lightsheet | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| lattice_lightsheet | llsm_dl | LLSM-CARE | pwm_core.recon.llsm_solvers | llsm_care_recon | yes | yes |
| lensless | traditional_cpu | ADMM-TV | pwm_core.recon.lensless_solver | run_lensless | no | yes |
| lensless | best_quality | FlatNet | pwm_core.recon.flatnet | run_flatnet | no | yes |
| lensless | famous_dl | FlatNet | pwm_core.recon.flatnet | run_flatnet | no | yes |
| lensless | small_gpu | FlatNet-Lite | pwm_core.recon.flatnet | run_flatnet | no | yes |
| libs | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| libs | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| libs | libs_dl | LIBS-CNN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| lidar | traditional_cpu | FISTA-L2 (depth) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| lidar | best_quality | PointNeXt [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| lidar | famous_dl | PointNet++ [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| light_field | traditional_cpu | Shift-and-Sum | pwm_core.recon.light_field_solver | run_light_field | no | yes |
| light_field | best_quality | LFBM5D | pwm_core.recon.light_field_solver | lfbm5d_recon | no | yes |
| light_field | famous_dl | LFSSR | pwm_core.recon.light_field_solver | lfssr_recon | no | yes |
| light_field | small_gpu | LFSSR | pwm_core.recon.light_field_solver | lfssr_recon | no | yes |
| lightsheet | traditional_cpu | Fourier Notch Filter | pwm_core.recon.lightsheet_solver | run_lightsheet | no | yes |
| lightsheet | best_quality | VSNR | pwm_core.recon.lightsheet_solver | run_lightsheet | no | yes |
| lightsheet | famous_dl | DeStripe | pwm_core.recon.destripe_net | run_destripe | no | yes |
| lightsheet | small_gpu | DeStripe | pwm_core.recon.destripe_net | run_destripe | no | yes |
| lucky_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| lucky_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| lucky_imaging | lucky_dl | Lucky-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| machine_vision | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| machine_vision | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| machine_vision | mv_dl | PatchCore [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| magnetic_particle | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| magnetic_particle | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| magnetic_particle | mpi_dl | MPI-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| maldi_msi | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| maldi_msi | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| maldi_msi | msi_dl | MSI-UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mammography | traditional_cpu | FBP (mammography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| mammography | best_quality | MammoNet (GatorTron) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mammography | famous_dl | Mammo-ResNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| matrix | traditional_cpu | FISTA-L1 | pwm_core.recon.classical | run_fista_l2 | no | yes |
| matrix | best_quality | FISTA-L1 (high quality) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| matrix | famous_dl | LISTA | pwm_core.recon.lista | run_lista | no | yes |
| matrix | small_gpu | LISTA | pwm_core.recon.lista | run_lista | no | yes |
| mfm | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mfm | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| mfm | mfm_dl | MFM-UNet | pwm_core.recon.mfm_solvers | mfm_dl_recon | yes | yes |
| minflux | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| minflux | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| minflux | minflux_dl | MINFLUX-Net | pwm_core.recon.minflux_solvers | minflux_dl_recon | yes | yes |
| mr_elastography | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mr_elastography | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mr_elastography | mre_dl | MRE-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mr_fingerprinting | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mr_fingerprinting | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mr_fingerprinting | mrf_dl | MRF-Net [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| mra | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mra | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mra | mra_dl | MRA-VesselNet [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| mri | traditional_cpu | Zero-Filled IFFT | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| mri | best_quality | CS-MRI (Wavelet) | pwm_core.recon.mri_solvers | run_cs_mri | no | yes |
| mri | famous_dl | MoDL | pwm_core.recon.modl | run_modl | no | yes |
| mri | small_gpu | MoDL (5 unrolls) | pwm_core.recon.modl | run_modl | no | yes |
| mri | sense | SENSE | pwm_core.recon.mri_solvers | run_sense | no | yes |
| mrs | traditional_cpu | SENSE (spectroscopy) | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| mrs | best_quality | MRS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| mrs | famous_dl | HLSVD-MRS [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| multispectral_sat | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| multispectral_sat | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| multispectral_sat | ms_dl | MS-Pansharpening-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| muon_tomo | traditional_cpu | FBP (muon tomography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| muon_tomo | best_quality | POCA-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| muon_tomo | famous_dl | EM-POCA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nerf | traditional_cpu | SfM + MVS | pwm_core.recon.nerf_solver | run_nerf | no | yes |
| nerf | best_quality | Mip-NeRF 360 | pwm_core.recon.nerf_solver | run_nerf | yes | yes |
| nerf | famous_dl | NeRF (original MLP) | pwm_core.recon.nerf_solver | run_nerf | no | yes |
| nerf | small_gpu | Instant-NGP | pwm_core.recon.nerf_solver | run_nerf | no | yes |
| nerf | rl_proxy | Richardson-Lucy (proxy baseline) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nerf | fista_proxy | FISTA-TV (proxy baseline) | pwm_core.recon.cs_solvers | run_ista | no | yes |
| neutron_diffraction | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| neutron_diffraction | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| neutron_diffraction | nd_dl | NeutronDiff-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| neutron_tomo | traditional_cpu | FBP (neutron tomography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| neutron_tomo | best_quality | NeuTomo-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| neutron_tomo | famous_dl | GRIDREC-Neutron [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nirs_brain | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nirs_brain | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nirs_brain | nirs_dl | fNIRS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nsom | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| nsom | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| nsom | nsom_dl | NSOM-Net | pwm_core.recon.nsom_solvers | nsom_dl_recon | yes | yes |
| ocean_acoustic_tomo | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ocean_acoustic_tomo | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ocean_acoustic_tomo | oat_dl | OAT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ocean_color | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ocean_color | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ocean_color | oc_dl | OC-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| oct | traditional_cpu | FFT Recon | pwm_core.recon.oct_solver | run_oct | no | yes |
| oct | best_quality | Spectral Estimation | pwm_core.recon.oct_solver | spectral_estimation_recon | no | yes |
| oct | famous_dl | OCT Denoising Net | pwm_core.recon.oct_solver | oct_denoising_net_recon | no | yes |
| oct | small_gpu | OCT Denoising Net | pwm_core.recon.oct_solver | oct_denoising_net_recon | no | yes |
| octa | traditional_cpu | FFT Recon (OCTA) | pwm_core.recon.oct_solver | run_oct | no | yes |
| octa | best_quality | OCTA-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| octa | famous_dl | OCTA-FF [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| odt | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| odt | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| odt | odt_dl | ODT-Net (PhaseNet) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| palm_storm | traditional_cpu | Richardson-Lucy (STORM/PALM) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| palm_storm | best_quality | DECODE-SMLM | pwm_core.recon.smlm_solvers | decode_smlm_recon | yes | yes |
| palm_storm | famous_dl | DeepSTORM | pwm_core.recon.smlm_solvers | deep_storm_recon | yes | yes |
| panorama | traditional_cpu | Laplacian Pyramid Fusion | pwm_core.recon.panorama_solver | run_panorama_fusion | no | yes |
| panorama | best_quality | Guided Filter Fusion | pwm_core.recon.panorama_solver | run_panorama_fusion | no | yes |
| panorama | famous_dl | IFCNN | pwm_core.recon.ifcnn | run_ifcnn | no | yes |
| panorama | small_gpu | IFCNN | pwm_core.recon.ifcnn | run_ifcnn | no | yes |
| particle_calorimetry | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| particle_calorimetry | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| particle_calorimetry | cal_dl | CaloDiffusion [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| passive_microwave | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| passive_microwave | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| passive_microwave | pm_dl | PM-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pet | traditional_cpu | FBP (emission tomography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| pet | best_quality | NeuroLF-PET | pwm_core.recon.pet_solvers | neurolF_pet_recon | yes | yes |
| pet | famous_dl | PET-DL (U-Net) | pwm_core.recon.pet_solvers | pet_unet_recon | yes | yes |
| pet_ct | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pet_ct | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pet_ct | petct_dl | PET-CT-Fusion-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pet_mr | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pet_mr | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pet_mr | petmr_dl | PET-MR-DeepJoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| phase_contrast | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| phase_contrast | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| phase_contrast | pc_dl | PhaseNet | pwm_core.recon.phase_contrast_solvers | phase_net_recon | yes | yes |
| phase_retrieval | traditional_cpu | HIO | pwm_core.recon.phase_retrieval_solver | run_phase_retrieval | no | yes |
| phase_retrieval | best_quality | RAAR [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| phase_retrieval | famous_dl | prDeep [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| phase_retrieval | small_gpu | prDeep [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| photoacoustic | traditional_cpu | Back Projection | pwm_core.recon.photoacoustic_solver | run_photoacoustic | no | yes |
| photoacoustic | best_quality | Time Reversal [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| photoacoustic | famous_dl | Deep-PAT [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| photoacoustic | small_gpu | Deep-PAT [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| photometric_stereo | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| photometric_stereo | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| photometric_stereo | ps_dl | PS-FCN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| polarization | traditional_cpu | PnP-HQS | pwm_core.recon.pnp | run_pnp | no | yes |
| polarization | best_quality | PolarNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| polarization | famous_dl | Stokes-NN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| polsar | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| polsar | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| polsar | polsar_dl | PolSAR-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| portal_imaging | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| portal_imaging | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| portal_imaging | portal_dl | PortalDL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| proton_radiography | traditional_cpu | FBP (proton radiography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| proton_radiography | best_quality | ProtonRecon-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| proton_radiography | famous_dl | FBP-Proton [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| proton_therapy_img | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| proton_therapy_img | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| proton_therapy_img | proton_therapy_dl | ProtonTherapy-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ptychography | traditional_cpu | ePIE | pwm_core.recon.ptychography_solver | run_epie | no | yes |
| ptychography | best_quality | PtychoNN | pwm_core.recon.ptychonn | run_ptychonn | no | yes |
| ptychography | famous_dl | PtychoNN | pwm_core.recon.ptychonn | run_ptychonn | no | yes |
| ptychography | small_gpu | PtychoNN 2.0 | pwm_core.recon.ptychonn | run_ptychonn | no | yes |
| pump_probe | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pump_probe | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| pump_probe | pp_dl | PumpProbe-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| quantum_illumination | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| quantum_illumination | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| quantum_illumination | qi_dl | QI-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| radio_astronomy | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| radio_astronomy | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| radio_astronomy | clean_dl | RadioAST-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| radio_interferometry | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| radio_interferometry | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| radio_interferometry | ri_dl | R2D2 (interferometry) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| raman_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| raman_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| raman_imaging | raman_dl | RamanNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sar | traditional_cpu | FBP (SAR backprojection) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| sar | best_quality | SAR-DL (PolSF) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sar | famous_dl | SAR-CNN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| saxs | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| saxs | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| saxs | saxs_dl | SAXS-VAE [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| seismic_tomo | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| seismic_tomo | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| seismic_tomo | seismic_dl | SeisInversion-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sem | traditional_cpu | Richardson-Lucy (SEM) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sem | best_quality | SEM-DL (SegNet) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sem | famous_dl | SEM-UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| shearography | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| shearography | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| shearography | shear_dl | ShearNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| shg | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| shg | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| shg | shg_dl | SHG-CARE | pwm_core.recon.shg_solvers | shg_care_recon | yes | yes |
| sim | traditional_cpu | Wiener-SIM | pwm_core.recon.sim_solver | run_sim_reconstruction | no | yes |
| sim | best_quality | HiFi-SIM | pwm_core.recon.sim_solver | run_sim_reconstruction | no | yes |
| sim | famous_dl | fairSIM (open-source) | pwm_core.recon.sim_solver | run_sim_reconstruction | no | yes |
| sim | small_gpu | Wiener-SIM (fast) | pwm_core.recon.sim_solver | run_sim_reconstruction | no | yes |
| sims | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sims | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sims | sims_dl | SIMS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| solar_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| solar_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| solar_imaging | solar_dl | SolarNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sonar | traditional_cpu | FISTA-L2 (DAS) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sonar | best_quality | SonarSR-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sonar | famous_dl | Sonar-CNN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spc | traditional_cpu | TVAL3 | pwm_core.recon.cs_solvers | run_tval3 | no | yes |
| spc | best_quality | ADMM-L1 | pwm_core.recon.spc_solvers | run_admm_spc | no | yes |
| spc | famous_dl | FISTA-L1 | pwm_core.recon.spc_solvers | run_fista_l1_spc | no | yes |
| spc | small_gpu | FISTA-L1 | pwm_core.recon.spc_solvers | run_fista_l1_spc | no | yes |
| spc | ista_net_plus | ISTA-Net+ | pwm_core.recon.spc_solvers | run_admm_spc | no | yes |
| spc | hatnet | HATNet | pwm_core.recon.spc_solvers | run_fista_l1_spc | no | yes |
| spect | traditional_cpu | FBP (emission tomography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| spect | best_quality | SPECT-DL (OSEM+) | pwm_core.recon.spect_solvers | spect_dl_recon | yes | yes |
| spect | famous_dl | SPECT-UNet | pwm_core.recon.spect_solvers | spect_unet_recon | yes | yes |
| spect_ct | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spect_ct | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spect_ct | spectct_dl | SPECT-CT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spectral_ct | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spectral_ct | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spectral_ct | spectral_ct_dl | SpectralCT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spinning_disk | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| spinning_disk | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| spinning_disk | sd_dl | SD-CARE | pwm_core.recon.spinning_disk_solvers | sd_care_recon | yes | yes |
| srs | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| srs | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| srs | srs_dl | SRS-DeepSpec [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sted | traditional_cpu | Richardson-Lucy (STED) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| sted | best_quality | STED-Net (CARE) | pwm_core.recon.sted_solvers | sted_care_recon | yes | yes |
| sted | famous_dl | RCAN-STED | pwm_core.recon.sted_solvers | rcan_sted_recon | yes | yes |
| stem | traditional_cpu | Richardson-Lucy (STEM) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| stem | best_quality | STEM-DL (AtomSegNet) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| stem | famous_dl | STEM-UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| stm | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| stm | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| stm | stm_dl | STM-Net | pwm_core.recon.stm_solvers | stm_dl_recon | yes | yes |
| streak_camera | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| streak_camera | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| streak_camera | streak_dl | StreakNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| structured_light | traditional_cpu | FISTA-L2 (phase unwrap) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| structured_light | best_quality | SL-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| structured_light | famous_dl | FTPD [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| swi | traditional_cpu | FBP [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| swi | best_quality | DL-Recon [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| swi | swi_dl | SWI-Net [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| talbot_lau | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| talbot_lau | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| talbot_lau | talbot_dl | Talbot-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| tem | traditional_cpu | FISTA-L2 (CTF correction) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| tem | best_quality | TEM-DL (ePIE-Net) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| tem | famous_dl | TEM-UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| terahertz | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| terahertz | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| terahertz | thz_dl | THz-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| three_photon | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| three_photon | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| three_photon | 3p_dl | 3P-Net (CARE) | pwm_core.recon.three_photon_solvers | three_photon_care_recon | yes | yes |
| tirf | traditional_cpu | Richardson-Lucy (TIRF) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| tirf | best_quality | TIRF-Net (CARE) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| tirf | famous_dl | TIRF-SRRF [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| tof_camera | traditional_cpu | FISTA-L2 (depth) | pwm_core.recon.classical | run_fista_l2 | no | yes |
| tof_camera | best_quality | ToF-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| tof_camera | famous_dl | ToF-MPI Deconv [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| two_photon | traditional_cpu | Richardson-Lucy (2P) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| two_photon | best_quality | 2P-Net (CARE) | pwm_core.recon.two_photon_solvers | two_photon_care_recon | yes | yes |
| two_photon | famous_dl | 2P-DeepInterp | pwm_core.recon.two_photon_solvers | deep_interp_recon | yes | yes |
| ultrasonic_phased_array | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ultrasonic_phased_array | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ultrasonic_phased_array | upa_dl | TFM-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ultrasound | traditional_cpu | Richardson-Lucy (ultrasound) | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ultrasound | best_quality | US-UNet (DeepUS) [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| ultrasound | famous_dl | US-CNN [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| us_mri | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| us_mri | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| us_mri | us_mri_dl | US-MRI-Net [proxy] | pwm_core.recon.mri_solvers | run_zero_filled | no | yes |
| waxs | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| waxs | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| waxs | waxs_dl | WAXS-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| weather_radar | traditional_cpu | RDA [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| weather_radar | best_quality | SAR-DL [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| weather_radar | weather_dl | NowcastNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| widefield | traditional_cpu | Richardson-Lucy | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| widefield | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| widefield | famous_dl | CARE | pwm_core.recon.care_unet | run_care | no | yes |
| widefield | small_gpu | CARE | pwm_core.recon.care_unet | run_care | no | yes |
| widefield_lowdose | traditional_cpu | BM3D + RL | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| widefield_lowdose | best_quality | CARE | pwm_core.recon.care_unet | run_care | yes | yes |
| widefield_lowdose | famous_dl | Noise2Void | pwm_core.recon.noise2void | noise2void_denoise | no | yes |
| widefield_lowdose | small_gpu | Noise2Void | pwm_core.recon.noise2void | noise2void_denoise | no | yes |
| xfel_sfx | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xfel_sfx | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xfel_sfx | sfx_dl | SFX-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_crystallography | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_crystallography | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_crystallography | xtal_dl | AlphaFold-SF [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_ndt | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_ndt | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_ndt | ndt_dl | NDT-DefectNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_radiography | traditional_cpu | FBP (X-ray radiography) | pwm_core.recon.ct_solvers | run_fbp | no | yes |
| xray_radiography | best_quality | CheXNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xray_radiography | famous_dl | X-ray UNet [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xrf_imaging | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xrf_imaging | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xrf_imaging | xrf_dl | XRF-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xrf_tomo | traditional_cpu | Adjoint [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xrf_tomo | best_quality | PnP-ADMM [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |
| xrf_tomo | xrft_dl | XRFT-Net [proxy] | pwm_core.recon.richardson_lucy | run_richardson_lucy | no | yes |

## GPU Solvers (need Modal T4)

| Modality | Solver | Module | Function |
|----------|--------|--------|----------|
| afm | CARE | pwm_core.recon.care_unet | run_care |
| afm | AFM-UNet | pwm_core.recon.afm_solvers | afm_unet_recon |
| cacti | HiSViT-9 | pwm_core.recon.cacti_solvers | hisvit_cacti |
| cacti | HiSViT-13 | pwm_core.recon.cacti_solvers | hisvit_cacti |
| cbct | FDK-DL | pwm_core.recon.cbct_solvers | fdk_dl_recon |
| cbct | CBCT-UNet | pwm_core.recon.cbct_solvers | cbct_unet_recon |
| confocal_3d | 3D CARE | pwm_core.recon.care_unet | run_care |
| confocal_livecell | CARE | pwm_core.recon.care_unet | run_care |
| cryo_et | CARE | pwm_core.recon.care_unet | run_care |
| cryo_et | CryoCARE | pwm_core.recon.cryoet_solvers | cryocare_recon |
| dark_field | CARE | pwm_core.recon.care_unet | run_care |
| dark_field | DF-UNet | pwm_core.recon.darkfield_solvers | df_unet_recon |
| dic | CARE | pwm_core.recon.care_unet | run_care |
| dic | DIC-Net | pwm_core.recon.dic_solvers | dic_dl_recon |
| dna_paint | CARE | pwm_core.recon.care_unet | run_care |
| dna_paint | DECODE-PAINT | pwm_core.recon.smlm_solvers | decode_smlm_recon |
| endoscopy | EndoMapper-Net | pwm_core.recon.endoscopy_solvers | endomapper_recon |
| endoscopy | AF-SfMLearner | pwm_core.recon.endoscopy_solvers | af_sfm_learner_recon |
| expansion | CARE | pwm_core.recon.care_unet | run_care |
| expansion | EXpansionNet | pwm_core.recon.expansion_solvers | expansion_dl_recon |
| fib_sem | CARE | pwm_core.recon.care_unet | run_care |
| fib_sem | FIB-SEM-Net | pwm_core.recon.fibsem_solvers | fibsem_dl_recon |
| fundus | RETFound | pwm_core.recon.fundus_solvers | retfound_recon |
| fundus | DR-Grade-Net | pwm_core.recon.fundus_solvers | dr_grade_net_recon |
| gaussian_splatting | 3DGS (full) | pwm_core.recon.gaussian_splatting_solver | run_gaussian_splatting |
| ism | CARE | pwm_core.recon.care_unet | run_care |
| ism | ISM-Reassignment-Net | pwm_core.recon.ism_solvers | ism_dl_recon |
| lattice_lightsheet | CARE | pwm_core.recon.care_unet | run_care |
| lattice_lightsheet | LLSM-CARE | pwm_core.recon.llsm_solvers | llsm_care_recon |
| mfm | CARE | pwm_core.recon.care_unet | run_care |
| mfm | MFM-UNet | pwm_core.recon.mfm_solvers | mfm_dl_recon |
| minflux | CARE | pwm_core.recon.care_unet | run_care |
| minflux | MINFLUX-Net | pwm_core.recon.minflux_solvers | minflux_dl_recon |
| nerf | Mip-NeRF 360 | pwm_core.recon.nerf_solver | run_nerf |
| nsom | CARE | pwm_core.recon.care_unet | run_care |
| nsom | NSOM-Net | pwm_core.recon.nsom_solvers | nsom_dl_recon |
| palm_storm | DECODE-SMLM | pwm_core.recon.smlm_solvers | decode_smlm_recon |
| palm_storm | DeepSTORM | pwm_core.recon.smlm_solvers | deep_storm_recon |
| pet | NeuroLF-PET | pwm_core.recon.pet_solvers | neurolF_pet_recon |
| pet | PET-DL (U-Net) | pwm_core.recon.pet_solvers | pet_unet_recon |
| phase_contrast | CARE | pwm_core.recon.care_unet | run_care |
| phase_contrast | PhaseNet | pwm_core.recon.phase_contrast_solvers | phase_net_recon |
| shg | CARE | pwm_core.recon.care_unet | run_care |
| shg | SHG-CARE | pwm_core.recon.shg_solvers | shg_care_recon |
| spect | SPECT-DL (OSEM+) | pwm_core.recon.spect_solvers | spect_dl_recon |
| spect | SPECT-UNet | pwm_core.recon.spect_solvers | spect_unet_recon |
| spinning_disk | CARE | pwm_core.recon.care_unet | run_care |
| spinning_disk | SD-CARE | pwm_core.recon.spinning_disk_solvers | sd_care_recon |
| sted | STED-Net (CARE) | pwm_core.recon.sted_solvers | sted_care_recon |
| sted | RCAN-STED | pwm_core.recon.sted_solvers | rcan_sted_recon |
| stm | CARE | pwm_core.recon.care_unet | run_care |
| stm | STM-Net | pwm_core.recon.stm_solvers | stm_dl_recon |
| three_photon | CARE | pwm_core.recon.care_unet | run_care |
| three_photon | 3P-Net (CARE) | pwm_core.recon.three_photon_solvers | three_photon_care_recon |
| two_photon | 2P-Net (CARE) | pwm_core.recon.two_photon_solvers | two_photon_care_recon |
| two_photon | 2P-DeepInterp | pwm_core.recon.two_photon_solvers | deep_interp_recon |
| widefield | CARE | pwm_core.recon.care_unet | run_care |
| widefield_lowdose | CARE | pwm_core.recon.care_unet | run_care |