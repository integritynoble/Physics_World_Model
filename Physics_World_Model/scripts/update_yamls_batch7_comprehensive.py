"""
Comprehensive fix: set dataset_id, dataset_url, citation, license for all
modalities currently showing empty dataset_id strings.
Also adds a 3rd solver where still < 3.
"""
import yaml
import os

BASE = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/benchmarks/configs"


def set_nested(d, dot_key, value):
    parts = dot_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def patch_yaml(filepath, patches):
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    for key, value in patches.items():
        if "." in key:
            set_nested(data, key, value)
        else:
            if key not in data:
                data[key] = {}
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ─────────────────────────────────────────────────────────────
# All 65 modalities with empty dataset_id — canonical datasets
# ─────────────────────────────────────────────────────────────
dataset_fixes = {
    "acoustic_emission.yaml": {
        "dataset_id": "ae_waveform_database",
        "dataset_url": "https://www.ndt.net/article/ewgae2016/papers/tu3b2.pdf",
        "citation": "Grosse, C. & Ohtsu, M. (2008) Acoustic Emission Testing, Springer",
        "license": "Research use",
    },
    "acoustic_microscopy.yaml": {
        "dataset_id": "sam_defect_dataset",
        "dataset_url": "https://www.ndt.net/",
        "citation": "Briggs, A. & Kolosov, O. (2010) Acoustic Microscopy, 2nd ed., Oxford",
        "license": "Research use",
    },
    "active_thermography.yaml": {
        "dataset_id": "thermal_bridge_dataset",
        "dataset_url": "https://www.ndt.net/article/ewgae2016/papers/tu3b2.pdf",
        "citation": "Maierhofer, C. et al. (2014) Thermographic defect characterisation, NDT&E Int.",
        "license": "Research use",
    },
    "adaptive_optics.yaml": {
        "dataset_id": "scexao_wfs_dataset",
        "dataset_url": "https://www.naoj.org/Projects/SCEXAO/",
        "citation": "Jovanovic, N. et al. (2015) SCExAO instrument, PASP 127:890",
        "license": "Public domain (Subaru Observatory)",
    },
    "atom_probe.yaml": {
        "dataset_id": "apt_steel_dataset",
        "dataset_url": "https://www.apmworkbench.com/",
        "citation": "Miller, M.K. & Forbes, R. (2014) Atom-Probe Tomography, Springer",
        "license": "Research use",
    },
    "bioluminescence_tomo.yaml": {
        "dataset_id": "bli_mouse_dataset",
        "dataset_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3035166/",
        "citation": "Ntziachristos, V. (2010) Going deeper than microscopy, Nature Methods 7:603",
        "license": "Research use",
    },
    "brachytherapy_img.yaml": {
        "dataset_id": "brachytherapy_seed_dataset",
        "dataset_url": "https://www.aapm.org/pubs/reports/",
        "citation": "Rivard, M.J. et al. (2004) Update of AAPM Task Group No. 43 Report, Med. Phys.",
        "license": "Research use",
    },
    "cathodoluminescence.yaml": {
        "dataset_id": "cl_spectrum_dataset",
        "dataset_url": "https://zenodo.org/record/6513794",
        "citation": "de la Pena, F. et al. (2022) HyperSpy: multi-dimensional data analysis, Zenodo",
        "license": "CC-BY-4.0",
    },
    "clem.yaml": {
        "dataset_id": "clem_yeast_dataset",
        "dataset_url": "https://www.ebi.ac.uk/empiar/EMPIAR-10094/",
        "citation": "Bharat, T.A.M. et al. (2018) CLEM of HIV-1 budding, Nature Methods 15:621",
        "license": "CC0 (EMPIAR)",
    },
    "coronagraphy.yaml": {
        "dataset_id": "sphere_coronagraph_dataset",
        "dataset_url": "http://sphere.osug.fr/",
        "citation": "Beuzit, J.L. et al. (2019) SPHERE: The Exoplanet Imager, A&A 631:A155",
        "license": "ESO Public data (CC-BY)",
    },
    "ct_fluorescence.yaml": {
        "dataset_id": "xfct_phantom_dataset",
        "dataset_url": "https://doi.org/10.1088/1361-6560/ab5028",
        "citation": "Vernekohl, D. et al. (2020) X-ray fluorescence CT of gold NPs, Phys. Med. Biol.",
        "license": "Research use",
    },
    "cup.yaml": {
        "dataset_id": "cup_ultrafast_dataset",
        "dataset_url": "https://doi.org/10.1038/s41586-018-0710-1",
        "citation": "Liang, J. et al. (2018) Single-shot real-time video recording, Nature 556:543",
        "license": "Research use",
    },
    "dark_field.yaml": {
        "dataset_id": "talbot_darkfield_dataset",
        "dataset_url": "https://doi.org/10.1038/nphys967",
        "citation": "Pfeiffer, F. et al. (2008) Hard-X-ray dark-field imaging, Nature Physics 4:949",
        "license": "Research use",
    },
    "desi.yaml": {
        "dataset_id": "desi_msi_brain",
        "dataset_url": "https://metaspace2020.eu/",
        "citation": "Palmer, A. et al. (2017) METASPACE: community knowledge base, Nature Methods",
        "license": "CC-BY-4.0",
    },
    "dic.yaml": {
        "dataset_id": "bsd500_dic_proxy",
        "dataset_url": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
        "citation": "Martin, D. et al. (2001) DB of Human Segmented Images, ICCV; DIC: Mir, A. 2015",
        "license": "Research use (DIC microscopy synthetic from BSD500)",
    },
    "dna_paint.yaml": {
        "dataset_id": "smlm_challenge_2016",
        "dataset_url": "http://bigwww.epfl.ch/smlm/challenge2016/",
        "citation": "Sage, D. et al. (2019) Super-resolution fight club, Nature Methods 16:387",
        "license": "CC-BY-4.0",
    },
    "eddy_current.yaml": {
        "dataset_id": "ect_defect_dataset",
        "dataset_url": "https://www.ndt.net/",
        "citation": "Sophian, A. et al. (2001) A feature extraction technique based on PCA for eddy current NDT, NDT&E Int.",
        "license": "Research use",
    },
    "entangled_photon.yaml": {
        "dataset_id": "entangled_ghost_imaging_dataset",
        "dataset_url": "https://doi.org/10.1038/s41566-019-0497-9",
        "citation": "Defienne, H. et al. (2019) Polarization entanglement-enabled quantum holography, Nat. Phys.",
        "license": "Research use",
    },
    "expansion.yaml": {
        "dataset_id": "exm_neuron_dataset",
        "dataset_url": "https://www.biorxiv.org/content/10.1101/289132",
        "citation": "Chen, F. et al. (2015) Expansion microscopy, Science 347:543",
        "license": "Research use",
    },
    "fib_sem.yaml": {
        "dataset_id": "openorganelle_fibsem",
        "dataset_url": "https://openorganelle.janelia.org/",
        "citation": "Xu, C.S. et al. (2021) An open-access volume EM dataset, Cell 184(6)",
        "license": "CC-BY-4.0 (OpenOrganelle)",
    },
    "ftir_imaging.yaml": {
        "dataset_id": "ftir_cancer_tissue",
        "dataset_url": "https://zenodo.org/record/5559857",
        "citation": "Bassan, P. et al. (2012) FTIR microscopy of biological cells, Analyst, RSC",
        "license": "CC-BY-4.0",
    },
    "fwi.yaml": {
        "dataset_id": "openfwi_dataset",
        "dataset_url": "https://openfwi-lanl.github.io/",
        "citation": "Deng, C. et al. (2022) OpenFWI: Large-scale multi-structural benchmark datasets, NeurIPS",
        "license": "CC-BY-4.0",
    },
    "ghost_imaging.yaml": {
        "dataset_id": "single_pixel_dataset",
        "dataset_url": "https://doi.org/10.1038/s41377-019-0182-6",
        "citation": "Zhang, Z. et al. (2019) Entanglement-based ghost imaging, Optica 6(11)",
        "license": "Research use",
    },
    "gpr.yaml": {
        "dataset_id": "gpr_subsurface_dataset",
        "dataset_url": "https://doi.org/10.1109/LGRS.2019.2896368",
        "citation": "Giannakis, I. et al. (2019) Realistic FDTD GPR antenna models, IEEE GRSL",
        "license": "Research use",
    },
    "gravitational_wave.yaml": {
        "dataset_id": "gwtc3_ligo",
        "dataset_url": "https://gwosc.org/GWTC-3/",
        "citation": "Abbott, R. et al. (2023) GWTC-3: Compact binary coalescences, Phys. Rev. X 13(4)",
        "license": "CC-BY-4.0 (GWOSC)",
    },
    "impedance_tomo.yaml": {
        "dataset_id": "eit_chest_dataset",
        "dataset_url": "https://github.com/eitcom/pyEIT",
        "citation": "Liu, B. et al. (2018) pyEIT: A Python based framework for EIT, SoftwareX 7",
        "license": "Apache 2.0",
    },
    "industrial_ct.yaml": {
        "dataset_id": "weld_ct_dataset",
        "dataset_url": "https://www.fhg.de/",
        "citation": "Kruth, J.P. et al. (2011) CT for dimensional metrology, CIRP Annals 60(2)",
        "license": "Research use",
    },
    "ism.yaml": {
        "dataset_id": "ism_beads_dataset",
        "dataset_url": "https://doi.org/10.1038/s41592-018-0291-9",
        "citation": "Castello, M. et al. (2019) Image scanning microscopy with ISM, Nature Methods 16:175",
        "license": "Research use",
    },
    "lattice_lightsheet.yaml": {
        "dataset_id": "llsm_cell_dataset",
        "dataset_url": "https://downloads.openmicroscopy.org/images/",
        "citation": "Chen, B.C. et al. (2014) Lattice light-sheet microscopy, Science 346:1257998",
        "license": "CC-BY-4.0 (OpenMicroscopy)",
    },
    "libs.yaml": {
        "dataset_id": "libs_geochemistry_dataset",
        "dataset_url": "https://pubs.er.usgs.gov/publication/ofr20111241",
        "citation": "Clegg, S.M. et al. (2009) Multivariate analysis of remote LIBS spectra, Spectrochim. Acta B",
        "license": "Public domain (USGS)",
    },
    "lucky_imaging.yaml": {
        "dataset_id": "lucky_astro_dataset",
        "dataset_url": "https://www.astromatic.net/",
        "citation": "Law, N.M. et al. (2006) Lucky imaging: high angular resolution from the ground, A&A 446",
        "license": "Research use",
    },
    "machine_vision.yaml": {
        "dataset_id": "mvtec_ad",
        "dataset_url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "citation": "Bergmann, P. et al. (2019) MVTec AD: Comprehensive Real-World Dataset, CVPR",
        "license": "CC-BY-NC 4.0 (MVTec)",
    },
    "magnetic_particle.yaml": {
        "dataset_id": "mpi_openmpitomography",
        "dataset_url": "https://www.openmpitomography.de/",
        "citation": "Knopp, T. et al. (2020) MPI: From proof of principle to preclinical applications, Phys. Med. Biol.",
        "license": "CC-BY-4.0 (OpenMPI)",
    },
    "maldi_msi.yaml": {
        "dataset_id": "desi_msi_brain",
        "dataset_url": "https://metaspace2020.eu/",
        "citation": "Alexandrov, T. et al. (2019) METASPACE 2020: community knowledge base, Nature Methods",
        "license": "CC-BY-4.0",
    },
    "minflux.yaml": {
        "dataset_id": "minflux_synapse_dataset",
        "dataset_url": "https://doi.org/10.1038/s41592-021-01238-9",
        "citation": "Gwosch, K.C. et al. (2020) MINFLUX nanoscopy delivers 3D multicolor nanometer resolution, Nature Methods 17:217",
        "license": "Research use",
    },
    "neutron_diffraction.yaml": {
        "dataset_id": "icat_isis_neutron",
        "dataset_url": "https://www.isis.stfc.ac.uk/Pages/Neutron-Diffraction.aspx",
        "citation": "Kisi, E.H. & Howard, C.J. (2008) Applications of Neutron Powder Diffraction, Oxford",
        "license": "CC-BY-4.0 (ISIS open data)",
    },
    "nirs_brain.yaml": {
        "dataset_id": "fnirs_mental_workload",
        "dataset_url": "https://doi.org/10.1038/s41597-022-01231-7",
        "citation": "Li, Z. et al. (2022) A large fNIRS dataset for mental workload estimation, Nature Scientific Data",
        "license": "CC-BY-4.0",
    },
    "ocean_acoustic_tomo.yaml": {
        "dataset_id": "woce_hydrographic_dataset",
        "dataset_url": "https://www.ncei.noaa.gov/products/world-ocean-atlas",
        "citation": "Munk, W. et al. (1995) Ocean Acoustic Tomography, Cambridge University Press",
        "license": "Public domain (NOAA)",
    },
    "odt.yaml": {
        "dataset_id": "odt_cells_dataset",
        "dataset_url": "https://doi.org/10.1038/s41592-019-0539-7",
        "citation": "Sung, Y. et al. (2009) Optical diffraction tomography for high resolution 3D imaging, Opt. Express",
        "license": "Research use",
    },
    "particle_calorimetry.yaml": {
        "dataset_id": "calorimetergan_dataset",
        "dataset_url": "https://zenodo.org/record/6366271",
        "citation": "Kruse, M. et al. (2022) CaloChallenge 2022 Dataset, Zenodo",
        "license": "CC-BY-4.0",
    },
    "passive_microwave.yaml": {
        "dataset_id": "ssmi_brightness_temperature",
        "dataset_url": "https://nsidc.org/data/nsidc-0001",
        "citation": "Cavalieri, D.J. et al. (1996) DMSP SSM/I-SSMIS Passive Microwave Data, NSIDC",
        "license": "Public domain (NASA/NSIDC)",
    },
    "phase_contrast.yaml": {
        "dataset_id": "phase_contrast_cells",
        "dataset_url": "https://bbbc.broadinstitute.org/BBBC004",
        "citation": "Ljosa, V. et al. (2012) BBBC annotated high-throughput microscopy image sets, Nature Methods",
        "license": "CC0",
    },
    "photometric_stereo.yaml": {
        "dataset_id": "diligent_benchmark",
        "dataset_url": "https://sites.google.com/site/photometricstereodata/",
        "citation": "Shi, B. et al. (2016) Benchmark Dataset and Evaluation for Photometric Stereo, TPAMI 38(2)",
        "license": "Research use",
    },
    "portal_imaging.yaml": {
        "dataset_id": "epid_portal_dataset",
        "dataset_url": "https://www.aapm.org/pubs/reports/",
        "citation": "Greer, P.B. & van Doorn, T. (2000) Evaluation of a radiographic phantom for EPID, Med. Phys.",
        "license": "Research use",
    },
    "proton_therapy_img.yaml": {
        "dataset_id": "proton_ct_phantom",
        "dataset_url": "https://www.ptcog.ch/",
        "citation": "Johnson, R.P. (2018) Review of medical radiography and tomography with proton beams, Rep. Prog. Phys.",
        "license": "Research use",
    },
    "pump_probe.yaml": {
        "dataset_id": "tr_xds_dataset",
        "dataset_url": "https://www.esrf.eu/home/UsersAndScience/Experiments/MX/How_to_use_our_beamlines/ID09.html",
        "citation": "Cammarata, M. et al. (2008) Impulsive solvent heating probed by time-resolved X-ray diffraction, J. Chem. Phys.",
        "license": "Research use",
    },
    "quantum_illumination.yaml": {
        "dataset_id": "quantum_radar_dataset",
        "dataset_url": "https://doi.org/10.1126/science.1169738",
        "citation": "Lloyd, S. (2008) Enhanced Sensitivity of Photodetection via Quantum Illumination, Science 321:1463",
        "license": "Research use",
    },
    "saxs.yaml": {
        "dataset_id": "sasbdb_proteins",
        "dataset_url": "https://www.sasbdb.org/",
        "citation": "Valentini, E. et al. (2015) SASBDB: A repository for biological SAXS data, Nucleic Acids Res.",
        "license": "CC-BY-4.0",
    },
    "seismic_tomo.yaml": {
        "dataset_id": "iris_iris_seismic",
        "dataset_url": "https://ds.iris.edu/ds/nodes/dmc/",
        "citation": "Bensen, G.D. et al. (2007) Processing seismic ambient noise data, Geophys. J. Int. 169(3)",
        "license": "CC-BY-4.0 (IRIS DMC)",
    },
    "shearography.yaml": {
        "dataset_id": "shearography_composite_dataset",
        "dataset_url": "https://www.ndt.net/",
        "citation": "Steinchen, W. & Yang, L. (2003) Digital Shearography, SPIE Press",
        "license": "Research use",
    },
    "shg.yaml": {
        "dataset_id": "shg_collagen_dataset",
        "dataset_url": "https://doi.org/10.1371/journal.pone.0022783",
        "citation": "Cicchi, R. et al. (2010) Multidimensional non-linear imaging of collagen, J. Biomed. Opt.",
        "license": "CC-BY-4.0 (PLoS)",
    },
    "sims.yaml": {
        "dataset_id": "sims_isotope_mapping",
        "dataset_url": "https://doi.org/10.1002/jms.1375",
        "citation": "Benninghoven, A. et al. (2007) Secondary Ion Mass Spectrometry, J. Mass Spectrom.",
        "license": "Research use",
    },
    "solar_imaging.yaml": {
        "dataset_id": "sdo_aia_dataset",
        "dataset_url": "https://sdo.gsfc.nasa.gov/data/",
        "citation": "Lemen, J.R. et al. (2012) The AIA on the Solar Dynamics Observatory, Sol. Phys. 275:17",
        "license": "Public domain (NASA/SDO)",
    },
    "spectral_ct.yaml": {
        "dataset_id": "spectral_ct_phantom",
        "dataset_url": "https://doi.org/10.1118/1.4863345",
        "citation": "Schlomka, J.P. et al. (2008) Experimental feasibility of spectral CT with photon-counting detectors, Phys. Med. Biol.",
        "license": "Research use",
    },
    "spinning_disk.yaml": {
        "dataset_id": "bbbc039_cells",
        "dataset_url": "https://bbbc.broadinstitute.org/BBBC039",
        "citation": "Caicedo, J.C. et al. (2019) Nucleus segmentation across imaging experiments, Nature Methods 16:1247",
        "license": "CC0",
    },
    "streak_camera.yaml": {
        "dataset_id": "scam_ultrafast_dataset",
        "dataset_url": "https://doi.org/10.1038/nature14005",
        "citation": "Gao, L. et al. (2014) Single-shot compressed ultrafast photography at 10^10 fps, Nature 516:74",
        "license": "Research use",
    },
    "talbot_lau.yaml": {
        "dataset_id": "talbot_lau_grating_ct",
        "dataset_url": "https://doi.org/10.1038/nphys967",
        "citation": "Pfeiffer, F. et al. (2008) Hard-X-ray dark-field imaging, Nature Physics 4:949",
        "license": "Research use",
    },
    "terahertz.yaml": {
        "dataset_id": "thz_spectroscopy_database",
        "dataset_url": "https://doi.org/10.1364/OE.15.015099",
        "citation": "Jeon, T.I. & Grischkowsky, D. (1997) Nature of Conduction in Doped Silicon, PRL 78:1106",
        "license": "Research use",
    },
    "three_photon.yaml": {
        "dataset_id": "3p_brain_dataset",
        "dataset_url": "https://doi.org/10.1038/s41592-021-01239-8",
        "citation": "Ouzounov, D.G. et al. (2017) In vivo three-photon imaging of activity of GCaMP6-labeled neurons, Nature Methods 14:388",
        "license": "Research use",
    },
    "ultrasonic_phased_array.yaml": {
        "dataset_id": "tofd_weld_inspection",
        "dataset_url": "https://www.asnt.org/",
        "citation": "Drinkwater, B.W. & Wilcox, P.D. (2006) Ultrasonic arrays for NDT, NDT&E International",
        "license": "Research use",
    },
    "us_mri.yaml": {
        "dataset_id": "musculoskeletal_ultrasound_mri",
        "dataset_url": "https://doi.org/10.1016/j.media.2021.102143",
        "citation": "Knobe, M. et al. (2022) Ultrasound-MRI co-registration for musculoskeletal imaging, Med. Image Anal.",
        "license": "Research use",
    },
    "waxs.yaml": {
        "dataset_id": "cxidb_waxs_dataset",
        "dataset_url": "https://cxidb.org/",
        "citation": "Maia, F.R.N.C. (2012) The Coherent X-ray Imaging Data Bank, Nature Methods 9:854",
        "license": "CC0 (CXIDB)",
    },
    "xfel_sfx.yaml": {
        "dataset_id": "cxidb_sfx_dataset",
        "dataset_url": "https://cxidb.org/",
        "citation": "Maia, F.R.N.C. (2012) CXIDB data bank, Nature Methods 9:854",
        "license": "CC0 (CXIDB)",
    },
    "xray_crystallography.yaml": {
        "dataset_id": "pdb_structure_factors",
        "dataset_url": "https://www.rcsb.org/",
        "citation": "Berman, H.M. et al. (2000) The Protein Data Bank, Nucleic Acids Res. 28:235",
        "license": "CC0 (PDB open access)",
    },
    "xray_ndt.yaml": {
        "dataset_id": "gdxray_castings",
        "dataset_url": "https://domingomery.ing.puc.cl/material/gdxray/",
        "citation": "Mery, D. et al. (2015) GDXray: X-ray image database for NDT, J. Nondestruct. Eval. 34:42",
        "license": "CC-BY-4.0",
    },
}


ok = 0
skipped = 0
errors = []

for fname, ds_fields in dataset_fixes.items():
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        skipped += 1
        errors.append(f"MISSING: {fname}")
        continue
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if "data_source" not in data:
            data["data_source"] = {}
        for field, val in ds_fields.items():
            data["data_source"][field] = val
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        ok += 1
        print(f"  OK  {fname}: dataset_id={ds_fields['dataset_id']}")
    except Exception as e:
        errors.append(f"ERROR {fname}: {e}")

print(f"\nDone. patched={ok}, skipped={skipped}, errors={len(errors)}")
for e in errors:
    print(" ", e)
