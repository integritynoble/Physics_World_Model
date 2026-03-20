# Cover Letter

**To:** The Editors, *Nature*
**Re:** Submission of "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"
**From:** Chengshuai Yang (corresponding author)

---

Dear Editors,

We submit for your consideration our manuscript "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging," which establishes the universal mathematical structure underlying all computational imaging forward models and identifies operator mismatch as the dominant cross-modality reconstruction bottleneck.

### Why Nature

This work addresses a foundational question that spans multiple scientific disciplines: What is the structure of the space of imaging forward models, and what limits reconstruction quality in practice? The answer --- 11 primitives suffice to represent every imaging forward model, and calibration errors dominate reconstruction quality across all validated modalities --- has direct implications for optics, medical imaging (CT and MRI), electron microscopy, materials science, and compressed sensing theory. The cross-disciplinary scope, spanning four physical carrier families (optical photons, X-ray photons, electrons, and nuclear spins) and seven imaging modalities, makes Nature the natural venue. No existing specialist journal reaches all affected communities simultaneously.

### The advance

We prove two complementary theorems.

1. **The Finite Primitive Basis Theorem** proves that every imaging forward model in a broad operator class admits an approximate representation as a directed acyclic graph over exactly 11 physically typed primitives --- a library that is both sufficient and minimal. This is the first universal representation result for imaging forward models, reducing the apparent diversity of dozens of imaging modalities to a compact, finite basis.

2. **The Triad Decomposition** proves that every reconstruction failure decomposes into exactly three root causes --- information deficiency, carrier noise, and operator mismatch --- with a formal condition under which mismatch dominates. The two results are complementary: the DAG structure localizes mismatch to specific physical operators, making correction actionable.

### The surprise

The central empirical finding is counterintuitive: **in 5 of 7 validated modalities, correcting the forward model with a classical solver recovers more reconstruction quality than replacing the solver with a state-of-the-art deep network operating on the same mismatched operator.** The computational imaging community has invested a decade progressing from compressed sensing to vision transformers, yet calibration errors --- not algorithmic limitations --- are the dominant bottleneck. A sub-pixel mask perturbation in coded aperture imaging erases twice the reconstruction gains achieved by this entire decade of solver innovation. This finding reframes the field's investment priorities.

### Evidence strength

The evidence is comprehensive and rigorous:

- **7 modalities** validated: coded aperture spectral imaging (CASSI), coded aperture temporal imaging (CACTI), single-pixel camera (SPC), lensless imaging, X-ray computed tomography (CT), electron ptychography, and magnetic resonance imaging (MRI).
- **4 carrier families** spanning the physical spectrum: optical photons, X-ray photons, electrons, and nuclear spins.
- **5 real instruments** providing hardware validation: CASSI (TSA real data, 5 scenes), CACTI (EfficientSCI real data, 4 scenes), CT (2 public sinogram datasets), electron ptychography (4D-STEM SrTiO3), and MRI (M4Raw multi-coil brain k-space).
- **Strong effect sizes**: correction recovers +0.8 to +10.7 dB of PSNR, with 95% bootstrap confidence intervals of +/-0.3 dB and Cohen's d > 2.0 for every modality.
- **Held-out closure test**: 8 additional modalities (including 3 exotic: quantum ghost imaging, THz-TDS, Compton scatter) confirm basis completeness under a frozen protocol.
- **26 registered modality templates** spanning 5 carrier families, with 90 total templates in the open-source release.
- **Full reproducibility**: open-source codebase with 139 experiment bundles, RunBundle manifests with SHA-256 provenance, and ~4-hour total reproduction time on a single GPU.

### Suggested referees

We suggest the following referees, chosen to cover the breadth of disciplines addressed by this work:

1. **Emmanuel Candes** (Stanford University, Department of Statistics)
   *Justification:* Pioneer of compressed sensing theory and convex optimization for inverse problems. His foundational work on restricted isometry and sparsity-based recovery provides the theoretical context for our Gate 1 (recoverability) analysis. He can evaluate the mathematical rigor of both theorems and their implications for the field.

2. **Klaas Enno Stephan** (University of Zurich, Translational Neuromodeling Unit) or **Florian Knoll** (FAU Erlangen-Nuremberg, Department of Artificial Intelligence in Biomedical Engineering)
   *Justification:* Expert in computational MRI reconstruction and machine learning for medical imaging. Can evaluate the MRI validation results, the clinical relevance of coil sensitivity mismatch findings, and the comparison with ESPIRiT.

3. **John Miao** (UCLA, Department of Physics and Astronomy)
   *Justification:* Pioneer of coherent diffractive imaging and ptychography. His work on phase retrieval algorithms provides direct context for the electron ptychography validation. Can evaluate the 4D-STEM position jitter results and the claim that probe position errors trigger Gate 3 dominance.

4. **Rebecca Willett** (University of Chicago, Departments of Statistics and Computer Science)
   *Justification:* Leading researcher in computational imaging, inverse problems, and machine learning. Her work spans multiple imaging modalities and bridges theory with practice. Can evaluate the cross-modality generality claims and the OperatorGraph formalism.

5. **Laura Waller** (UC Berkeley, Department of Electrical Engineering and Computer Sciences)
   *Justification:* Expert in computational microscopy, phase retrieval, and lensless imaging. Her group has extensive experience with forward model calibration in optical systems. Can evaluate the lensless imaging validation, the practical implications of the calibration-vs.-solver finding, and the broader impact on computational microscopy.

### Excluded referees

We request that the following individuals not serve as referees due to potential conflicts of interest:

- Researchers at NextGen PlatformAI C Corp (employer of the corresponding author)
- Researchers at Westlake University who are direct collaborators of Xin Yuan

### Manuscript details

- Main manuscript: ~4,500 words (excluding Methods, References, and figure legends), 8 display items (5 figures, 3 tables)
- Online Methods: ~4,000 words
- Supplementary Information: ~15,000 words, 15 Supplementary Notes, 13 Supplementary Tables
- The manuscript has not been submitted elsewhere and is not under consideration at another journal.
- A companion paper establishing the formal semantics of the 11 primitives is under preparation for SIAM Journal on Imaging Sciences (Yang, 2026, "Finite Primitive Theorem").
- A companion paper on the InverseNet benchmark for CASSI/CACTI mismatch analysis is under review at ECCV 2026 (Yang and Yuan, 2026).

We believe this work will be of broad interest to the Nature readership and look forward to your consideration.

Sincerely,

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com
