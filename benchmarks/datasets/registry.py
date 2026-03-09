"""Dataset registry – central mapping of modalities to real benchmark datasets.

Each ``DatasetEntry`` describes a publicly available dataset: where to
download it, what format it comes in, which converter to use, and which
benchmark modalities it serves.

The registry follows the priority chain:
  1. web           – real public datasets
  2. experimental  – local real measurements
  3. synthetic_web – synthetic data from online sources
  4. generated     – create programmatically (last resort)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetEntry:
    """Metadata for one downloadable benchmark dataset."""

    id: str                         # e.g., "cave_multispectral"
    name: str                       # Human-readable name
    source_type: str                # "web" | "experimental" | "synthetic_web" | "generated"
    url: str                        # Direct download URL (or landing page)
    format: str                     # "mat" | "hdf5" | "tiff" | "npy" | "nifti" | "png" | "zip"
    citation: str                   # Citation string
    license: str                    # License type
    size_mb: float                  # Approximate download size in MB
    storage: str                    # "local" | "gcs"
    applies_to: List[str]           # modality_ids this dataset serves
    converter: str                  # function name in downloaders.py
    x_shape: List[int]              # expected output shape after conversion
    mat_key: Optional[str] = None   # key inside .mat/.hdf5 file
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — covers all 168 benchmark modalities
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, DatasetEntry] = {

    # ==================================================================
    # 1. Compressive Imaging  (9 modalities)
    # ==================================================================
    "indian_pines_hs": DatasetEntry(
        id="indian_pines_hs",
        name="Indian Pines Hyperspectral",
        source_type="web",
        url="https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        format="mat",
        citation="Purdue University, Indian Pines, 1992",
        license="Public domain",
        size_mb=6.0,
        storage="local",
        applies_to=[
            "cassi", "cacti", "spc", "coded_exposure", "cup",
            "dcchi", "ghost_imaging", "matrix_completion", "one_pixel",
            # Also serves compressive modality "matrix"
            "matrix",
        ],
        converter="convert_mat",
        x_shape=[145, 145, 200],
        mat_key="indian_pines_corrected",
    ),
    "pavia_university_hs": DatasetEntry(
        id="pavia_university_hs",
        name="Pavia University Hyperspectral",
        source_type="web",
        url="https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        format="mat",
        citation="Pavia University, ROSIS sensor, 2003",
        license="Public domain",
        size_mb=33.0,
        storage="local",
        applies_to=[
            "cassi", "cacti", "spc", "coded_exposure", "cup",
            "dcchi", "ghost_imaging", "hyperspectral_rs",
            "sar", "insar", "polsar",
            # Spectral imaging modalities
            "ftir_imaging", "raman_imaging", "cars", "srs",
            "brillouin", "libs", "sims", "desi",
        ],
        converter="convert_mat",
        x_shape=[610, 340, 103],
        mat_key="paviaU",
    ),
    "salinas_hs": DatasetEntry(
        id="salinas_hs",
        name="Salinas Hyperspectral",
        source_type="web",
        url="https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        format="mat",
        citation="Salinas Valley, AVIRIS sensor, 1998",
        license="Public domain",
        size_mb=30.0,
        storage="local",
        applies_to=[
            "cassi", "cacti", "spc", "coded_exposure",
            "hyperspectral_rs",
            # Remote sensing
            "hyperspectral_remote", "multispectral_sat", "ocean_color",
        ],
        converter="convert_mat",
        x_shape=[512, 217, 204],
        mat_key="salinas_corrected",
    ),
    "cacti_video_generated": DatasetEntry(
        id="cacti_video_generated",
        name="CACTI Video Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Llull et al., Optica 2015",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cacti"],
        converter="generate_cacti_video_phantom",
        x_shape=[128, 128],
        notes="Synthetic CACTI coded aperture compressive video phantom (B=8 frames, moving objects)",
    ),

    # ==================================================================
    # 2. Microscopy  (26 modalities)
    # ==================================================================
    "bsd68_microscopy": DatasetEntry(
        id="bsd68_microscopy",
        name="BSD68 - Natural Images for Denoising",
        source_type="web",
        url="https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip",
        format="zip",
        citation="Martin et al., Berkeley Segmentation Dataset, ICCV 2001",
        license="Research use",
        size_mb=15.0,
        storage="local",
        applies_to=[
            # Core microscopy
            "widefield", "confocal_3d", "confocal_livecell",
            "two_photon", "lightsheet", "sted", "sim",
            # Extended microscopy
            "dark_field", "dic", "phase_contrast", "polarization",
            "spinning_disk", "lattice_lightsheet", "widefield_lowdose",
            "flim", "shg", "three_photon", "tirf",
            "fpm", "ism", "expansion",
            # Lensless / computational
            "lensless",
            # Confocal endomicroscopy (medical micro)
            "confocal_endomicroscopy",
            # Correlation microscopy
            "clem",
        ],
        converter="convert_png_stack",
        x_shape=[256, 256],
        notes="68 natural test images, widely used as denoising benchmark",
    ),
    "biosr_sim_microtubules": DatasetEntry(
        id="biosr_sim_microtubules",
        name="BioSR Microtubules (SIM)",
        source_type="web",
        url="https://ndownloader.figshare.com/files/25714514",
        format="zip",
        citation="Qiao et al., BioSR, Nature Methods 2021",
        license="CC BY 4.0",
        size_mb=1600.0,
        storage="gcs",
        applies_to=["sim", "widefield"],
        converter="convert_png_stack",
        x_shape=[256, 256],
        notes="Paired WF/SIM TIFF images; microtubules subset from figshare 13264793",
    ),
    "smlm_generated": DatasetEntry(
        id="smlm_generated",
        name="Generated SMLM Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated SMLM emitter field",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "palm", "storm", "palm_storm",
            "dna_paint", "minflux",
        ],
        converter="generate_smlm_phantom",
        x_shape=[256, 256],
        notes="Sparse point emitters with Gaussian PSFs; SMLM benchmark fallback",
    ),

    # ==================================================================
    # 3. Medical CT / X-ray  (18 modalities)
    # ==================================================================
    "lodopab_ct_sample": DatasetEntry(
        id="lodopab_ct_sample",
        name="LoDoPaB-CT - Low-Dose Parallel Beam CT",
        source_type="web",
        url="https://zenodo.org/records/3384092/files/ground_truth_train_000.hdf5",
        format="hdf5",
        citation="Leuschner et al., LoDoPaB-CT, Scientific Data 2021",
        license="CC BY 4.0",
        size_mb=950.0,
        storage="gcs",
        applies_to=[
            "ct", "cbct", "helical_ct", "dual_energy_ct",
            "photon_counting_ct", "spectral_ct",
            # Extended CT/X-ray
            "industrial_ct", "digital_breast_tomo", "mammography",
            "fluoroscopy", "xray_radiography", "xray_ndt",
            "dexa", "ct_fluorescence",  # angiography → dedicated generator
            "talbot_lau", "portal_imaging",
            # Neutron/muon tomography (similar reconstruction)
            "neutron_tomo", "muon_tomo",
        ],
        converter="convert_hdf5",
        x_shape=[362, 362],
        mat_key="data",
        notes="128 ground truth slices from training set 0",
    ),
    "covid_ct_lung_seg": DatasetEntry(
        id="covid_ct_lung_seg",
        name="COVID-19 CT Lung Segmentation",
        source_type="web",
        url="https://zenodo.org/records/3757476/files/COVID-19-CT-Seg_20cases.zip",
        format="zip",
        citation="COVID-19 CT Lung and Infection Segmentation, Zenodo 3757476",
        license="CC BY 4.0",
        size_mb=500.0,
        storage="local",
        applies_to=[
            "ct", "cbct", "helical_ct",
            # Extended CT/X-ray (also in GCS lodopab_ct_sample)
            "industrial_ct", "digital_breast_tomo", "mammography",
            "fluoroscopy", "xray_radiography", "xray_ndt",
            "dexa", "ct_fluorescence",  # angiography → dedicated generator
            "talbot_lau", "portal_imaging", "spectral_ct",
            "neutron_tomo", "muon_tomo",
        ],
        converter="convert_nifti_from_zip",
        x_shape=[256, 256],
        notes="20 COVID-19 CT volumes with lung/infection segmentations (NIfTI in ZIP)",
    ),

    # Dedicated angiography vessel phantom (DSA / 3DRA iodine map)
    "angiography_vessel_generated": DatasetEntry(
        id="angiography_vessel_generated",
        name="Generated X-ray Angiography Vessel Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated vascular tree phantom (fractal bifurcation, Murray's law)",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["angiography"],
        converter="generate_angiography_vessel_phantom",
        x_shape=[256, 256],
        notes=(
            "Fractal vascular tree with main trunk + 2-order branches; "
            "iodine concentration map calibrated to DSA/3DRA physics. "
            "Ref: Shen et al. Med. Image Anal. 2024; Wang et al. IEEE TMI 2024."
        ),
    ),

    # ==================================================================
    # 4. Medical MRI  (16 modalities)
    # ==================================================================
    # Dedicated ASL perfusion phantom (CBF map, brain compartments)
    "asl_mri_perfusion_generated": DatasetEntry(
        id="asl_mri_perfusion_generated",
        name="Generated ASL MRI Perfusion Phantom (CBF Map)",
        source_type="generated",
        url="",
        format="npy",
        citation=(
            "PWM generated ASL perfusion phantom. "
            "Calibrated to Alsop et al. MRM 2015 (pCASL recommended protocol); "
            "Mutsaerts et al. NeuroImage 2020 (ExploreASL population atlas)."
        ),
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["asl_mri"],
        converter="generate_asl_perfusion_phantom",
        x_shape=[128, 128],
        notes=(
            "Brain CBF map with cortical grey matter (~0.60), white matter (~0.35), "
            "basal ganglia/thalami (~0.90), CSF/ventricles (0), and vascular "
            "territory heterogeneity. Calibrated to pCASL perfusion physiology."
        ),
    ),
    "ixi_t1_sample": DatasetEntry(
        id="ixi_t1_sample",
        name="IXI Brain T1 MRI",
        source_type="web",
        url="https://brain-development.org/ixi-dataset/IXI-T1.tar",
        format="nifti",
        citation="IXI Dataset, Imperial College London",
        license="CC BY-SA 3.0",
        size_mb=1200.0,
        storage="gcs",
        applies_to=[
            "mri", "cest_mri", "dti",
            "fmri", "mrsi", "mrf", "mre",
            # Extended MRI  (asl_mri → dedicated generate_asl_perfusion_phantom)
            "diffusion_mri", "mr_elastography", "mr_fingerprinting",
            "mra", "mrs", "swi",
        ],
        converter="convert_nifti",
        x_shape=[256, 256],
        notes="T1-weighted brain images; extract central slices",
    ),
    "fastmri_knee_sample": DatasetEntry(
        id="fastmri_knee_sample",
        name="fastMRI Knee - Single-Coil k-space",
        source_type="web",
        url="https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xz",
        format="hdf5",
        citation="Zbontar et al., fastMRI, arXiv 2018",
        license="fastMRI License (research only)",
        size_mb=2500.0,
        storage="gcs",
        applies_to=["mri"],
        converter="convert_hdf5",
        x_shape=[320, 320],
        mat_key="reconstruction_esc",
        notes="Requires fastMRI data use agreement",
    ),

    # ==================================================================
    # 5. Medical Imaging – Other  (12 modalities)
    #    (Ultrasound, endoscopy, photoacoustic, DOT, etc.)
    # ==================================================================
    "medical_phantom_generated": DatasetEntry(
        id="medical_phantom_generated",
        name="Generated Medical Imaging Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated medical phantom (Shepp-Logan derived)",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            # Ultrasound family
            "ultrasound", "doppler_ultrasound", "ceus",
            "elastography", "ivus", "us_mri",
            # Endoscopy
            "endoscopy",
            # Optical medical
            "fundus", "octa",
            "dot", "nirs_brain",
            "photoacoustic",
            # Brachytherapy / proton
            "brachytherapy_img", "proton_therapy_img", "proton_radiography",
            # MRI fallback (also in GCS ixi_t1_sample)
            # asl_mri → dedicated generate_asl_perfusion_phantom
            "mri", "cest_mri", "dti", "fmri",
            "diffusion_mri", "mr_elastography", "mr_fingerprinting",
            "mra", "mrs", "swi", "mrsi", "mrf", "mre",
        ],
        converter="generate_medical_phantom",
        x_shape=[256, 256],
        notes="Shepp-Logan variant with tissue contrast for general medical imaging",
    ),

    # ==================================================================
    # 6. Electron Microscopy  (13 modalities)
    # ==================================================================
    "empiar_10146_apoferritin": DatasetEntry(
        id="empiar_10146_apoferritin",
        name="EMPIAR-10146 Apoferritin Micrographs",
        source_type="web",
        url="https://ftp.ebi.ac.uk/empiar/world_availability/10146/",
        format="mrc",
        citation="EMPIAR-10146, Apoferritin tutorial dataset",
        license="CC0",
        size_mb=5000.0,
        storage="gcs",
        applies_to=["cryo_em", "cryo_et", "tem"],
        converter="convert_mrc",
        x_shape=[256, 256],
        notes="Large dataset; download subset of micrographs",
    ),
    "electron_microscopy_generated": DatasetEntry(
        id="electron_microscopy_generated",
        name="Generated Electron Microscopy Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated EM phantom (nanoparticle field)",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "cryo_em", "cryo_et", "tem", "sem",
            "eels", "stem", "fib_sem",
            "ebsd", "edx_mapping", "electron_diffraction",
            "electron_holography", "electron_tomography",
            "cathodoluminescence",
        ],
        converter="generate_em_phantom",
        x_shape=[256, 256],
        notes="Nanoparticles on carbon film; fallback for all EM modalities",
    ),

    # ==================================================================
    # 7. Nuclear Emission  (5 modalities)
    # ==================================================================
    "brainweb_pet": DatasetEntry(
        id="brainweb_pet",
        name="BrainWeb PET Phantom",
        source_type="synthetic_web",
        url="https://brainweb.bic.mni.mcgill.ca/brainweb/",
        format="raw",
        citation="Collins et al., BrainWeb, IEEE TMI 1998",
        license="Non-commercial research",
        size_mb=50.0,
        storage="local",
        applies_to=[
            "pet", "spect",
            # Multi-modal nuclear
            "pet_ct", "pet_mr", "spect_ct",
            # Magnetic particle (similar reconstruction)
            "magnetic_particle",
        ],
        converter="convert_brainweb",
        x_shape=[256, 256],
        notes="Synthetic brain PET phantom with anatomical priors",
    ),

    # ==================================================================
    # 8. Remote Sensing / SAR  (10 modalities)
    # ==================================================================
    "kennedy_space_center_hs": DatasetEntry(
        id="kennedy_space_center_hs",
        name="Kennedy Space Center Hyperspectral",
        source_type="web",
        url="https://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat",
        format="mat_v73",
        citation="Kennedy Space Center, AVIRIS sensor",
        license="Public domain",
        size_mb=57.0,
        storage="local",
        applies_to=[
            "hyperspectral_rs", "sar", "insar", "polsar",
            "multispectral_rs",
            # Extended remote sensing
            "hyperspectral_remote", "multispectral_sat",
            "ocean_color", "passive_microwave",
            "weather_radar", "gpr",
        ],
        converter="convert_mat_v73",
        x_shape=[512, 614, 176],
        mat_key="KSC",
        notes="MATLAB v7.3 (HDF5) format; requires h5py or mat73",
    ),

    # ==================================================================
    # 9. Scanning Probe  (7 modalities)
    # ==================================================================
    "afm_synthetic_surface": DatasetEntry(
        id="afm_synthetic_surface",
        name="Synthetic AFM Surface Topography",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated surface topography",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["stm", "nsom", "mfm", "skpm", "sicm", "shpm"],
        converter="generate_surface",
        x_shape=[256, 256],
        notes="Generated fractal surface + step edges; fallback for scanning probe",
    ),

    # ==================================================================
    # 10. Optical / Coherent Imaging  (10 modalities)
    # ==================================================================
    "oct_generated": DatasetEntry(
        id="oct_generated",
        name="Generated OCT B-scan Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated OCT-like layered retinal structure",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["oct", "octa"],
        converter="generate_oct_phantom",
        x_shape=[256, 256],
        notes="Multi-layer retinal structure with speckle; fallback for OCT",
    ),
    "hologram_usaf_zenodo": DatasetEntry(
        id="hologram_usaf_zenodo",
        name="USAF Hologram (Zenodo)",
        source_type="web",
        url="https://zenodo.org/records/8059636/files/USAFamp_17.09mm.mat",
        format="mat",
        citation="Rogalski et al., Physics-driven twin-image removal, Zenodo 2023",
        license="CC0 1.0",
        size_mb=630.0,
        storage="gcs",
        applies_to=[
            "holography", "digital_holography", "doi",
            "phase_retrieval", "odt",
        ],
        converter="convert_mat",
        x_shape=[256, 256],
        notes="USAF resolution target hologram from digital in-line holographic microscopy",
    ),
    "coherent_imaging_generated": DatasetEntry(
        id="coherent_imaging_generated",
        name="Generated Coherent Imaging Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated resolution target with phase",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "holography", "ptychography", "phase_retrieval", "odt",
            "phase_contrast",
            # Interferometric
            "electron_holography",
        ],
        converter="generate_resolution_target",
        x_shape=[256, 256],
        notes="USAF-like resolution target with phase component",
    ),

    # ==================================================================
    # 11. Depth Imaging / LiDAR  (5 modalities)
    # ==================================================================
    "lidar_kitti_sample": DatasetEntry(
        id="lidar_kitti_sample",
        name="KITTI LiDAR Point Cloud Sample",
        source_type="web",
        url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
        format="zip",
        citation="Geiger et al., KITTI, CVPR 2012",
        license="CC BY-NC-SA 3.0",
        size_mb=5000.0,
        storage="gcs",
        applies_to=["lidar", "flash_lidar"],
        converter="convert_lidar_bin",
        x_shape=[64, 256],
        notes="Velodyne HDL-64E point clouds; project to range image",
    ),
    "depth_imaging_generated": DatasetEntry(
        id="depth_imaging_generated",
        name="Generated Depth Map",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated depth scene",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "lidar", "flash_lidar", "tof_camera",
            "structured_light", "photometric_stereo",
            "sonar",
        ],
        converter="generate_depth_map",
        x_shape=[256, 256],
        notes="Synthetic room scene with smooth surfaces and depth discontinuities",
    ),

    # ==================================================================
    # 12. Computational Photography  (6 modalities)
    # ==================================================================
    "computational_photo_generated": DatasetEntry(
        id="computational_photo_generated",
        name="Generated Scene for Computational Photography",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated test scene",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "event_camera", "hdr_imaging", "lensless", "panorama",
            "light_field", "integral",
            # Machine vision / industrial
            "machine_vision",
        ],
        converter="generate_test_scene",
        x_shape=[256, 256],
        notes="High-contrast scene with edges, gradients, and textures",
    ),

    # ==================================================================
    # 13. Astronomy / Space  (6 modalities)
    # ==================================================================
    "astronomy_generated": DatasetEntry(
        id="astronomy_generated",
        name="Generated Astronomical Field",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated star field with extended source",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "coronagraphy", "eht_imaging", "lucky_imaging",
            "solar_imaging",
            "radio_astronomy", "radio_interferometry",
            "gravitational_wave",
        ],
        converter="generate_star_field",
        x_shape=[256, 256],
        notes="Point sources + extended emission with Airy PSFs",
    ),

    # ==================================================================
    # 14. Ultrafast Imaging  (3 modalities)
    # ==================================================================
    "ultrafast_generated": DatasetEntry(
        id="ultrafast_generated",
        name="Generated Ultrafast Imaging Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated ultrafast dynamics phantom",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "pump_probe", "streak_camera", "xfel_sfx",
            "entangled_photon", "quantum_illumination",
        ],
        converter="generate_test_scene",
        x_shape=[256, 256],
        notes="Dynamic scene snapshot for ultrafast/quantum imaging",
    ),

    # ==================================================================
    # 15. Scientific Instrumentation  (10 modalities)
    #     (X-ray, neutron, atom probe, SAXS/WAXS, etc.)
    # ==================================================================
    "xray_diffraction_generated": DatasetEntry(
        id="xray_diffraction_generated",
        name="Generated Diffraction Pattern",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated crystallographic diffraction pattern",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "xray_crystallography", "saxs", "waxs",
            "neutron_diffraction", "electron_diffraction",
            "xfel_sfx",
        ],
        converter="generate_diffraction_pattern",
        x_shape=[256, 256],
        notes="Simulated Debye-Scherrer rings + Bragg peaks",
    ),
    "xrf_generated": DatasetEntry(
        id="xrf_generated",
        name="Generated XRF/MALDI Elemental Map",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated elemental distribution map",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "xrf_imaging", "xrf_tomo",
            "maldi_msi",
        ],
        converter="generate_elemental_map",
        x_shape=[256, 256],
        notes="Multi-element spatial distribution for spectroscopic imaging",
    ),
    "bioluminescence_tomo_generated": DatasetEntry(
        id="bioluminescence_tomo_generated",
        name="Generated Bioluminescence Tomography (BLT) Source Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation=(
            "PWM generated BLT source phantom. "
            "Calibrated to Lv et al., Phys. Med. Biol. 51:1479, 2006 (BLT phantom geometry); "
            "Han et al., Opt. Express 14(8):3673, 2006 (diffusion theory); "
            "Cong & Wang, J. Biomed. Opt. 11(2):020503, 2006 (boundary integral BLT); "
            "Jacques, Phys. Med. Biol. 58(11):R37, 2013 (tissue optical properties)."
        ),
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["bioluminescence_tomo"],
        converter="generate_blt_source_phantom",
        x_shape=[128, 128],
        notes=(
            "2-D projected bioluminescent source map: tissue background (0.02-0.05), "
            "2-5 primary tumour foci (0.70-1.0) with Gaussian fall-off at varying depths, "
            "1-3 satellite lesions (0.35-0.65), depth-attenuation gradient from diffusion "
            "approximation (μ_eff ≈ 0.46 cm⁻¹), and CCD Poisson shot noise (σ ≈ 0.03). "
            "Physically faithful to small-animal BLT phantom experiments."
        ),
    ),

    "brachytherapy_seed_generated": DatasetEntry(
        id="brachytherapy_seed_generated",
        name="Generated Brachytherapy I-125 Prostate Seed Implant Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation=(
            "PWM generated brachytherapy seed phantom. "
            "Calibrated to TG-43 prostate implant template geometry (ABS, 2012); "
            "I-125 seed attenuation from Nath et al., Med. Phys. 22(2):209, 1995; "
            "Dose-volume histogram validation per Potters et al., Int. J. Radiat. Oncol. 2001."
        ),
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["brachytherapy_img"],
        converter="generate_brachytherapy_seed_phantom",
        x_shape=[128, 128],
        notes=(
            "2-D attenuation map: soft-tissue prostate ellipsoid (mu=0.20/cm), urethra "
            "(mu=0.05/cm), pubic bone arc (mu=0.8-1.2/cm), and 70-110 I-125 seeds "
            "(mu~8.0/cm) on a TG-43 template grid with +/-2mm placement uncertainty. "
            "Multi-view Radon projections (18 angles) with quantum noise. "
            "Physically faithful to post-implant prostate brachytherapy verification imaging."
        ),
    ),

    "atom_probe_apt_generated": DatasetEntry(
        id="atom_probe_apt_generated",
        name="Generated Atom Probe Tomography (APT) Composition Map",
        source_type="generated",
        url="",
        format="npy",
        citation=(
            "PWM generated APT composition phantom. "
            "Calibrated to Hellman et al., Microsc. Microanal. 2000 (precipitate sizes); "
            "Blavette et al., Science 1999 (grain boundary segregation); "
            "Bas et al., Appl. Surf. Sci. 1995 (Bas reconstruction protocol); "
            "Larson et al., Local Electrode Atom Probe Tomography, Springer 2013."
        ),
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["atom_probe"],
        converter="generate_apt_composition_map",
        x_shape=[128, 128],
        notes=(
            "2-D elemental composition map with matrix (~0.25), gamma-prime precipitates "
            "(0.7-1.0, log-normal size distribution), grain boundary segregation bands "
            "(0.55-0.80, 1-2 px wide), dislocation loops, and trajectory aberration "
            "artefacts. Physically faithful to LEAP 5000 field-evaporation APT datasets."
        ),
    ),

    "active_thermography_generated": DatasetEntry(
        id="active_thermography_generated",
        name="Generated Pulsed Thermography Defect Map",
        source_type="generated",
        url="",
        format="npy",
        citation="Maldague (2001) Theory and Practice of Infrared Technology for NDE, Wiley",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["active_thermography"],
        converter="generate_thermography_phantom",
        x_shape=[256, 256],
        notes="Thermal diffusivity map with subsurface circular defects of varying depth and size. Forward model: 1-D heat diffusion PSF approximation.",
    ),
    "adaptive_optics_generated": DatasetEntry(
        id="adaptive_optics_generated",
        name="Generated Kolmogorov Turbulence Wavefront",
        source_type="generated",
        url="",
        format="npy",
        citation="Noll (1976) JOSA 66:207; Fried (1966) JOSA 56:1372",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["adaptive_optics"],
        converter="generate_ao_wavefront",
        x_shape=[256, 256],
        notes="Kolmogorov turbulence wavefront phase map (Zernike decomposition, modes 2-21). Represents the wavefront to be corrected by a deformable mirror.",
    ),
    "afm_surface_generated": DatasetEntry(
        id="afm_surface_generated",
        name="Generated AFM Surface Topography",
        source_type="generated",
        url="",
        format="npy",
        citation="Nečas & Klapetek (2012) Gwyddion; Jalili & Laxminarayana, Mechatronics 2004",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["afm"],
        converter="generate_afm_surface",
        x_shape=[256, 256],
        notes="AFM surface topography with three scene types: crystalline (periodic lattice), amorphous (layered rough), biological (cell-like features).",
    ),

    # ==================================================================
    # 15a. Scanning Acoustic Microscopy — dedicated C-scan phantom
    # ==================================================================
    "acoustic_microscopy_generated": DatasetEntry(
        id="acoustic_microscopy_generated",
        name="Generated SAM C-Scan Reflectivity Map",
        source_type="generated",
        url="",
        format="npy",
        citation=(
            "Guo et al. (2022) Ultrasonics 122:106679; "
            "Rigby et al. (2023) NDT&E Int. 138:102871"
        ),
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["acoustic_microscopy"],
        converter="generate_sam_phantom",
        x_shape=[256, 256],
        notes=(
            "Physics-calibrated SAM C-scan reflectivity map: die-attach boundary, "
            "elliptical delaminations (R≈−0.6), voids (R≈−0.9), wire-bond inclusions "
            "(R≈+0.5). Calibrated to microelectronic package and CFRP laminate SAM images. "
            "Forward model: 2-D PSF convolution (acoustic lens diffraction limited)."
        ),
    ),

    # ==================================================================
    # 15b. Acoustic Emission — dedicated source-energy-map generator
    # ==================================================================
    "acoustic_emission_generated": DatasetEntry(
        id="acoustic_emission_generated",
        name="Generated Acoustic Emission Source Energy Map",
        source_type="generated",
        url="",
        format="npy",
        citation=(
            "Grosse & Ohtsu (2008) Acoustic Emission Testing, Springer; "
            "Ebrahimkhanlou & Salamone (2019) Struct. Health Monit. 18(2):636-651"
        ),
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["acoustic_emission"],
        converter="generate_ae_source_map",
        x_shape=[256, 256],
        notes=(
            "Physics-accurate AE source intensity map: sparse point sources "
            "(crack-initiation hits, power-law amplitude) + line sources "
            "(crack propagation fronts) + diffuse background (dislocations). "
            "Models a 2-D panel monitored by a surface sensor array. "
            "Forward model: convolutional approximation to Green's function "
            "propagation (valid in the diffraction-limited far field of each sensor)."
        ),
    ),

    # ==================================================================
    # 16. Industrial Inspection  (4 modalities not covered above)
    # ==================================================================
    "industrial_ndt_generated": DatasetEntry(
        id="industrial_ndt_generated",
        name="Generated NDT Inspection Phantom",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated NDT inspection phantom",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "eddy_current",
            "shearography",
            "terahertz", "ultrasonic_phased_array",
        ],
        converter="generate_ndt_phantom",
        x_shape=[256, 256],
        notes="Material with embedded defects (voids, cracks, inclusions)",
    ),

    # ==================================================================
    # 17. Seismic / Geophysical  (3 modalities)
    # ==================================================================
    "seismic_generated": DatasetEntry(
        id="seismic_generated",
        name="Generated Seismic Velocity Model",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated layered velocity model",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=[
            "seismic_tomo", "fwi", "ocean_acoustic_tomo",
            "impedance_tomo",
        ],
        converter="generate_velocity_model",
        x_shape=[256, 256],
        notes="Layered earth model with velocity contrasts",
    ),

    # ==================================================================
    # 18. Neural Rendering  (2 modalities)
    # ==================================================================
    "nerf_generated": DatasetEntry(
        id="nerf_generated",
        name="Generated Multi-View Scene",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated multi-view test scene",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["nerf", "gaussian_splatting"],
        converter="generate_test_scene",
        x_shape=[256, 256],
        notes="Synthetic scene for neural rendering benchmarks",
    ),

    # ==================================================================
    # 19. Particle Physics  (1 modality)
    # ==================================================================

    # Brillouin microscopy — VIPA spectrometer phantom for viscoelastic mapping
    "brillouin_vipa_generated": DatasetEntry(
        id="brillouin_vipa_generated",
        name="Brillouin VIPA Spectral Maps (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Prevedel et al., Nat. Methods 2019",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["brillouin"],
        converter="generate_brillouin_vipa_phantom",
        x_shape=[64, 64],
        notes="Synthetic VIPA Brillouin spectral maps of biological cell monolayers (Lorentzian peak model)",
    ),

    "calorimeter_generated": DatasetEntry(
        id="calorimeter_generated",
        name="Generated Calorimeter Shower",
        source_type="generated",
        url="",
        format="npy",
        citation="PWM generated particle shower pattern",
        license="N/A",
        size_mb=1.0,
        storage="local",
        applies_to=["particle_calorimetry"],
        converter="generate_test_scene",
        x_shape=[256, 256],
        notes="Simplified electromagnetic shower energy deposition",
    ),

    # CARS microscopy — coherent anti-Stokes Raman scattering phantom
    "cars_raman_generated": DatasetEntry(
        id="cars_raman_generated",
        name="CARS Raman Microscopy (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Camp & Cicerone, Nat. Photon. 2015",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cars"],
        converter="generate_cars_raman_phantom",
        x_shape=[64, 64],
        notes="Synthetic CARS hyperspectral cell phantom with lipid droplets and NRB background",
    ),
    "cathodoluminescence_generated": DatasetEntry(
        id="cathodoluminescence_generated",
        name="Cathodoluminescence SEM Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Zagonel et al., Nano Lett. 2011",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cathodoluminescence"],
        converter="generate_cathodoluminescence_phantom",
        x_shape=[128, 128],
        notes="Synthetic CL map of semiconductor nanostructures with plasmonic nanoparticles",
    ),
    "cbct_head_generated": DatasetEntry(
        id="cbct_head_generated",
        name="CBCT Head/Dental Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Feldkamp et al., J. Opt. Soc. Am. A 1984",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cbct"],
        converter="generate_cbct_head_phantom",
        x_shape=[128, 128],
        notes="Synthetic CBCT dental/maxillofacial phantom with teeth, bone, air cavities, optional metal implant",
    ),
    "cest_mri_generated": DatasetEntry(
        id="cest_mri_generated",
        name="CEST MRI APT Brain Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Zhou et al., Nat. Med. 2003",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cest_mri"],
        converter="generate_cest_mri_phantom",
        x_shape=[64, 64],
        notes="Synthetic CEST z-spectrum brain phantom with tumour and stroke regions",
    ),
    "ceus_microbubble_generated": DatasetEntry(
        id="ceus_microbubble_generated",
        name="CEUS Microbubble Liver Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Errico et al., Nature 2015",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["ceus"],
        converter="generate_ceus_phantom",
        x_shape=[128, 128],
        notes="Synthetic CEUS liver vasculature phantom with microbubble perfusion and speckle noise",
    ),
    "clem_generated": DatasetEntry(
        id="clem_generated",
        name="CLEM FM+EM Paired Cell Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Bharat et al., Nat. Methods 2018",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["clem"],
        converter="generate_clem_phantom",
        x_shape=[128, 128],
        notes="Synthetic CLEM paired FM+EM cell phantom for super-resolution fusion",
    ),
    "coded_exposure_generated": DatasetEntry(
        id="coded_exposure_generated",
        name="Coded Exposure Flutter Shutter Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Raskar et al., SIGGRAPH 2006",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["coded_exposure"],
        converter="generate_coded_exposure_phantom",
        x_shape=[128, 128],
        notes="Synthetic flutter shutter coded exposure phantom for motion deblurring",
    ),
    "confocal_livecell_generated": DatasetEntry(
        id="confocal_livecell_generated",
        name="Confocal Live-Cell Fluorescence Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Weigert et al., Nat. Methods 2018",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["confocal_livecell"],
        converter="generate_confocal_livecell_phantom",
        x_shape=[128, 128],
        notes="Low-dose live-cell confocal phantom with mitochondria and endosomes",
    ),
    "confocal_3d_generated": DatasetEntry(
        id="confocal_3d_generated",
        name="Confocal 3D Cell Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Conchello & Lichtman, Nat. Methods 2005",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["confocal_3d"],
        converter="generate_confocal_3d_phantom",
        x_shape=[64, 64],
        notes="Synthetic 3D confocal cell phantom with nucleus, mitochondria, actin filaments",
    ),
    "confocal_endomicroscopy_generated": DatasetEntry(
        id="confocal_endomicroscopy_generated",
        name="Confocal Endomicroscopy Crypt Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Kiesslich et al., Gastroenterology 2004",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["confocal_endomicroscopy"],
        converter="generate_confocal_endomicroscopy_phantom",
        x_shape=[128, 128],
        notes="Synthetic CLE colonic crypt phantom with fibre bundle honeycomb artefacts",
    ),
    "coronagraphy_generated": DatasetEntry(
        id="coronagraphy_generated",
        name="Coronagraphic Exoplanet Detection Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Soummer et al., ApJ 2012",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["coronagraphy"],
        converter="generate_coronagraphy_phantom",
        x_shape=[64, 64],
        notes="Synthetic coronagraphic focal-plane phantom with stellar speckles and planet companions",
    ),
    "cryo_em_generated": DatasetEntry(
        id="cryo_em_generated",
        name="Cryo-EM Single-Particle Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Frank, Three-Dimensional Electron Microscopy, 2006",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cryo_em"],
        converter="generate_cryo_em_phantom",
        x_shape=[64, 64],
        notes="Synthetic cryo-EM single-particle phantom with CTF corruption and low-dose noise",
    ),
    "cryo_et_generated": DatasetEntry(
        id="cryo_et_generated",
        name="Cryo-ET Cellular Tomogram Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Bharat & Bharat, Nat. Methods 2015",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cryo_et"],
        converter="generate_cryo_et_phantom",
        x_shape=[64, 64],
        notes="Synthetic cryo-ET phantom with membranes, ribosomes, missing-wedge corruption",
    ),
    "ct_generated": DatasetEntry(
        id="ct_generated",
        name="CT Shepp-Logan Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Shepp & Logan, IEEE Trans. Nucl. Sci. 1974",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["ct"],
        converter="generate_ct_phantom",
        x_shape=[64, 64],
        notes="Synthetic CT Shepp-Logan phantom with Poisson sinogram noise",
    ),
    "ct_fluorescence_generated": DatasetEntry(
        id="ct_fluorescence_generated",
        name="X-ray Fluorescence CT Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Larsson et al., Phys. Med. Biol. 2020",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["ct_fluorescence"],
        converter="generate_ct_fluorescence_phantom",
        x_shape=[64, 64],
        notes="Synthetic XRF-CT phantom with fluorescent marker clusters and Compton background",
    ),
    "cup_generated": DatasetEntry(
        id="cup_generated",
        name="CUP Ultrafast Photography Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Gao et al., Nature 2014",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["cup"],
        converter="generate_cup_phantom",
        x_shape=[64, 64],
        notes="Synthetic CUP phantom with light pulse propagation and compressed measurement",
    ),
    "dark_field_generated": DatasetEntry(
        id="dark_field_generated",
        name="Dark-Field Microscopy Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Siedentopf & Zsigmondy, Ann. Physik 1902",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["dark_field"],
        converter="generate_dark_field_phantom",
        x_shape=[64, 64],
        notes="Synthetic dark-field phantom with sparse sub-wavelength particle scattering",
    ),
    "desi_generated": DatasetEntry(
        id="desi_generated",
        name="DESI-MSI Lipid Distribution Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Takats et al., Science 2004",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["desi"],
        converter="generate_desi_phantom",
        x_shape=[64, 64],
        notes="Synthetic DESI-MSI phantom with tissue region lipid/metabolite distributions",
    ),
    "dexa_generated": DatasetEntry(
        id="dexa_generated",
        name="DEXA Bone Density Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Blake & Fogelman, J. Clin. Densitom. 1997",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["dexa"],
        converter="generate_dexa_phantom",
        x_shape=[64, 64],
        notes="Synthetic DEXA phantom with bone mineral density and soft tissue regions",
    ),
    "dic_generated": DatasetEntry(
        id="dic_generated",
        name="DIC Microscopy Phase Phantom (Synthetic)",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic phantom based on Mehta & Sheppard, Nat. Photonics 2009",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["dic"],
        converter="generate_dic_phantom",
        x_shape=[64, 64],
        notes="Synthetic DIC phase phantom with cell nucleus, cytoplasm, differential gradient imaging",
    ),
    "diffusion_mri_generated": DatasetEntry(
        id="diffusion_mri_generated",
        name="Diffusion MRI Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic DTI/DWI phantom with white matter fiber tract structure",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["diffusion_mri"],
        converter="generate_diffusion_mri_phantom",
        x_shape=[64, 64],
        notes="Synthetic diffusion MRI phantom for benchmarking DTI reconstruction algorithms",
    ),
    "digital_breast_tomo_generated": DatasetEntry(
        id="digital_breast_tomo_generated",
        name="Digital Breast Tomosynthesis Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic DBT phantom with adipose/glandular tissue and lesion",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["digital_breast_tomo"],
        converter="generate_digital_breast_tomo_phantom",
        x_shape=[64, 64],
        notes="Synthetic DBT phantom for benchmarking limited-angle reconstruction algorithms",
    ),
    "dna_paint_generated": DatasetEntry(
        id="dna_paint_generated",
        name="DNA-PAINT Super-Resolution Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic DNA-PAINT phantom with stochastic blinking and PSF model",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["dna_paint"],
        converter="generate_dna_paint_phantom",
        x_shape=[64, 64],
        notes="Synthetic DNA-PAINT phantom for benchmarking super-resolution reconstruction",
    ),
    "doppler_ultrasound_generated": DatasetEntry(
        id="doppler_ultrasound_generated",
        name="Doppler Ultrasound Flow Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic Doppler ultrasound phantom with parabolic flow and speckle noise",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["doppler_ultrasound"],
        converter="generate_doppler_ultrasound_phantom",
        x_shape=[64, 64],
        notes="Synthetic Doppler US phantom for benchmarking flow velocity reconstruction",
    ),
    "dot_generated": DatasetEntry(
        id="dot_generated",
        name="Diffuse Optical Tomography Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic DOT phantom with absorption coefficient inclusions and boundary measurements",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["dot"],
        converter="generate_dot_phantom",
        x_shape=[64, 64],
        notes="Synthetic DOT phantom for benchmarking optical property reconstruction algorithms",
    ),
    "ebsd_generated": DatasetEntry(
        id="ebsd_generated",
        name="EBSD Grain Orientation Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic EBSD phantom with Voronoi polycrystalline microstructure and Kikuchi noise",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["ebsd"],
        converter="generate_ebsd_phantom",
        x_shape=[64, 64],
        notes="Synthetic EBSD phantom for benchmarking grain orientation reconstruction algorithms",
    ),
    "eddy_current_generated": DatasetEntry(
        id="eddy_current_generated",
        name="Eddy Current NDT Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic eddy current NDT phantom with conductivity defects and electromagnetic forward model",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["eddy_current"],
        converter="generate_eddy_current_phantom",
        x_shape=[64, 64],
        notes="Synthetic eddy current phantom for benchmarking defect reconstruction algorithms",
    ),
    "edx_mapping_generated": DatasetEntry(
        id="edx_mapping_generated",
        name="EDX Elemental Mapping Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic EDX elemental map phantom with Poisson counting statistics and X-ray background",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["edx_mapping"],
        converter="generate_edx_mapping_phantom",
        x_shape=[64, 64],
        notes="Synthetic EDX phantom for benchmarking elemental map denoising/reconstruction algorithms",
    ),
    "eels_generated": DatasetEntry(
        id="eels_generated",
        name="EELS Chemical Map Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic EELS phantom with chemical phase distributions and Poisson/multiple-scattering noise",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["eels"],
        converter="generate_eels_phantom",
        x_shape=[64, 64],
        notes="Synthetic EELS phantom for benchmarking chemical map reconstruction algorithms",
    ),
    "eht_imaging_generated": DatasetEntry(
        id="eht_imaging_generated",
        name="EHT Black Hole Imaging Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic EHT/VLBI phantom with accretion disk brightness distribution and sparse u-v coverage",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["eht_imaging"],
        converter="generate_eht_imaging_phantom",
        x_shape=[64, 64],
        notes="Synthetic EHT phantom for benchmarking sparse interferometric image reconstruction",
    ),
    "elastography_generated": DatasetEntry(
        id="elastography_generated",
        name="Elastography Stiffness Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic elastography phantom with shear modulus inclusions and shear wave displacement model",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["elastography"],
        converter="generate_elastography_phantom",
        x_shape=[64, 64],
        notes="Synthetic elastography phantom for benchmarking tissue stiffness reconstruction algorithms",
    ),
    "electron_diffraction_generated": DatasetEntry(
        id="electron_diffraction_generated",
        name="Electron Diffraction Pattern Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic electron diffraction phantom with Debye-Scherrer rings and dynamic scattering noise",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["electron_diffraction"],
        converter="generate_electron_diffraction_phantom",
        x_shape=[64, 64],
        notes="Synthetic electron diffraction phantom for benchmarking structure determination algorithms",
    ),
    "electron_holography_generated": DatasetEntry(
        id="electron_holography_generated",
        name="Electron Holography Phase Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic electron holography phantom with electrostatic potential and off-axis fringe model",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["electron_holography"],
        converter="generate_electron_holography_phantom",
        x_shape=[64, 64],
        notes="Synthetic electron holography phantom for benchmarking phase reconstruction algorithms",
    ),
    "electron_tomography_generated": DatasetEntry(
        id="electron_tomography_generated",
        name="Electron Tomography Reconstruction Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic ET phantom with macromolecular density blobs, limited-angle tilt series, and missing wedge",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["electron_tomography"],
        converter="generate_electron_tomography_phantom",
        x_shape=[64, 64],
        notes="Synthetic ET phantom for benchmarking missing-wedge compensation reconstruction algorithms",
    ),
    "endoscopy_generated": DatasetEntry(
        id="endoscopy_generated",
        name="Endoscopy Tissue Image Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic endoscopy phantom with mucosal texture, vignetting, and specular highlight model",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["endoscopy"],
        converter="generate_endoscopy_phantom",
        x_shape=[64, 64],
        notes="Synthetic endoscopy phantom for benchmarking image enhancement and restoration algorithms",
    ),
    "entangled_photon_generated": DatasetEntry(
        id="entangled_photon_generated",
        name="Entangled Photon Ghost Imaging Phantom Dataset",
        source_type="generated",
        url="",
        format="npy",
        citation="Synthetic entangled photon phantom with SPDC coincidence imaging and quantum noise model",
        license="synthetic",
        size_mb=1.0,
        storage="local",
        applies_to=["entangled_photon"],
        converter="generate_entangled_photon_phantom",
        x_shape=[64, 64],
        notes="Synthetic entangled photon phantom for benchmarking quantum ghost imaging reconstruction",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_datasets_for_modality(modality_id: str) -> List[DatasetEntry]:
    """Return all registry entries that apply to *modality_id*."""
    return [
        entry for entry in DATASET_REGISTRY.values()
        if modality_id in entry.applies_to
    ]


def get_datasets_for_category(category: str) -> List[DatasetEntry]:
    """Return all registry entries matching a category name (fuzzy)."""
    category_lower = category.lower().replace(" ", "_")
    _CATEGORY_MODALITIES = {
        "compressive_imaging": [
            "cassi", "cacti", "spc", "coded_exposure", "cup",
            "dcchi", "ghost_imaging", "matrix_completion", "one_pixel", "matrix",
        ],
        "microscopy": [
            "widefield", "confocal_3d", "confocal_livecell", "sim", "sted",
            "palm", "storm", "lightsheet", "two_photon", "minflux",
            "ism", "sim_3d", "sofi", "dark_field", "dic", "phase_contrast",
            "polarization", "spinning_disk", "lattice_lightsheet", "flim",
            "shg", "three_photon", "tirf", "fpm", "expansion",
            "dna_paint", "widefield_lowdose", "palm_storm",
        ],
        "medical_ct": [
            "ct", "cbct", "helical_ct", "dual_energy_ct",
            "photon_counting_ct", "spectral_ct", "industrial_ct",
            "digital_breast_tomo", "mammography", "fluoroscopy",
            "xray_radiography", "xray_ndt", "dexa", "angiography",
            "talbot_lau", "portal_imaging",
        ],
        "medical_mri": [
            "mri", "asl_mri", "cest_mri", "dti", "fmri",
            "mrsi", "mrf", "mre", "diffusion_mri", "mr_elastography",
            "mr_fingerprinting", "mra", "mrs", "swi",
        ],
        "electron_microscopy": [
            "cryo_em", "cryo_et", "tem", "sem", "eels",
            "stem", "fib_sem", "ebsd", "edx_mapping",
            "electron_diffraction", "electron_holography", "electron_tomography",
            "cathodoluminescence",
        ],
        "nuclear_emission": ["pet", "spect", "pet_ct", "pet_mr", "spect_ct"],
        "remote_sensing": [
            "sar", "insar", "polsar", "hyperspectral_rs", "multispectral_rs",
            "hyperspectral_remote", "multispectral_sat", "ocean_color",
            "passive_microwave", "weather_radar", "gpr",
        ],
        "scanning_probe": [
            "afm", "stm", "nsom", "mfm", "skpm", "sicm", "shpm",
        ],
        "optical": [
            "oct", "holography", "digital_holography", "doi",
            "ptychography", "lidar", "sonar", "phase_retrieval", "odt",
        ],
    }
    modalities = _CATEGORY_MODALITIES.get(category_lower, [])
    results = []
    seen = set()
    for entry in DATASET_REGISTRY.values():
        if entry.id in seen:
            continue
        if any(m in entry.applies_to for m in modalities):
            results.append(entry)
            seen.add(entry.id)
    return results


def get_all_local_entries() -> List[DatasetEntry]:
    """Return entries that should be stored locally (< 100 MB)."""
    return [e for e in DATASET_REGISTRY.values() if e.storage == "local"]


def get_all_gcs_entries() -> List[DatasetEntry]:
    """Return entries that should be stored on GCS (> 100 MB)."""
    return [e for e in DATASET_REGISTRY.values() if e.storage == "gcs"]
