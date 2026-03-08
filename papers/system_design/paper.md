# Purpose-conditioned system-solver selection across 168 computational imaging modalities

Chengshuai Shi^1,*^

^1^ Department of Electrical and Computer Engineering, University of Arizona, Tucson, AZ 85721, USA.

\* Correspondence: chengshuaishi@arizona.edu

*Submitted to Nature Methods*

---

## Abstract

Existing computational imaging benchmarks rank algorithms within a single modality but cannot answer the cross-modality question practitioners face: given application constraints on resolution, speed, budget and sample handling, which imaging system and solver should I use? Here we introduce PWM-SyS, a benchmark that evaluates 168 imaging systems — spanning photon, X-ray, electron, acoustic and radiofrequency carriers — against user-defined task queries. The framework pairs a neutral descriptor catalog of hardware and solver properties with Task-Normalized Adequacy (TNA), a scoring method that rates each system relative to task requirements rather than absolute capability. A three-stage protocol — feasibility gate, Pareto ranking and preference weighting — separates hard constraint satisfaction from multi-objective trade-off analysis. Across three pilot studies we show that PWM-SyS reproduces expert-consensus choices while surfacing non-obvious alternatives, and we provide an open interactive platform for community use.

---

## Introduction

Computational imaging recovers information that a detector cannot measure directly by coupling a physical encoding step with a computational reconstruction algorithm^1,2^. The past decade has seen transformative progress in reconstruction quality: deep-learning solvers now routinely exceed 38 dB peak signal-to-noise ratio (PSNR) on challenging inverse problems spanning compressed sensing^3,4^, medical imaging^5,6^, cryo-electron microscopy^7^ and optical microscopy^8,9^. Community benchmarks — fastMRI^10^, the AAPM Low-Dose CT Grand Challenge^11^, the Single Molecule Localization Microscopy challenge^12^ and modality-specific datasets — have been instrumental in driving this progress by standardizing evaluation protocols and enabling reproducible comparison.

These benchmarks share a common structure: fix the imaging system and its forward model, then rank reconstruction algorithms on a held-out test set. This answers the question *which algorithm wins on this dataset?* but leaves unanswered a harder question that practitioners face daily:

> Given my application requirements — spatial resolution, temporal resolution, budget, operator expertise, sample constraints — which imaging system and reconstruction solver should I deploy?

This system-level question differs fundamentally from algorithm benchmarking because it requires reasoning *across* modalities, not within a single one. A computed tomography (CT) system achieving 40 dB PSNR at \$2 M capital cost with specialist operation may be inferior to an ultrasound system at 34 dB, \$30 K and technician-level operation — if the task is clinical screening rather than high-resolution anatomical study. The answer depends on the purpose.

Currently, cross-modality comparison relies on informal sources: review articles^13,14^, textbook chapters^15,16^ and expert intuition. No standardized, quantitative framework exists for purpose-conditioned system selection across the full diversity of computational imaging. This gap has practical consequences. Researchers default to familiar modalities when superior alternatives exist for their specific constraints^17^. Hospital procurement committees evaluate imaging systems using ad hoc checklists rather than systematic multi-criteria analysis^18^. And the community lacks a common language to study what makes one system-solver combination preferable to another for a given task.

Multi-criteria decision analysis (MCDA) methods have been applied in healthcare technology assessment^18,19^, where frameworks such as EVIDEM^20^ and PAPRIKA^21^ structure preference elicitation. However, these general-purpose tools lack the domain-specific knowledge — forward model physics, solver benchmarks, sample compatibility constraints — needed for computational imaging system selection. Meanwhile, physics-aware benchmarks^22^ evaluate algorithmic robustness to model mismatch but do not extend to cross-modality hardware-solver trade-offs.

Here we present PWM-SyS (Physics World Model — System-to-Solver), a benchmark designed to bridge this gap (Fig. 1). PWM-SyS makes three contributions. First, we provide a **system descriptor catalog** containing neutral, verifiable hardware and solver properties for 168 imaging modalities across 19 application categories and 5 carrier families. Second, we introduce **Task-Normalized Adequacy (TNA)**, a scoring framework that evaluates system-solver pairs relative to task requirements across eight dimensions, preventing the distortions that arise from absolute capability ranking. Third, we define a **three-stage evaluation protocol** — feasibility gate, Pareto ranking and preference weighting — that cleanly separates hard constraint satisfaction from multi-objective optimization. PWM-SyS is implemented as an open web platform with natural-language querying, interactive recommendation and integrated physics simulation.

---

## Results

### A catalog of 168 imaging system descriptors

We assembled a structured catalog of 168 computational imaging systems by integrating three data sources: the Physics World Model (PWM) modality catalog^23^, which provides forward model specifications using an 11-primitive operator representation; the PWM algorithm catalog, which contains reconstruction benchmarks for 1,367 algorithms; and a curated hardware specification table compiled from manufacturer datasheets and published system descriptions (Fig. 2a, Extended Data Table 1).

The 168 systems span 19 application categories — from medical imaging (37 systems) and optical microscopy (24) to electron microscopy (11), remote sensing (11) and quantum imaging (3) — and employ 5 carrier families: photon (72 systems), X-ray (20), electron (14), acoustic (12) and radiofrequency/spin (19), with additional systems using ions, gamma rays, neutrons and mechanical probes (Fig. 2b). Capital costs range from \$500 (light-field camera) to \$1 B (gravitational-wave interferometer). Spatial resolutions span twelve orders of magnitude, from 0.01 pm (electron diffraction) to 100 km (radio interferometry).

Each descriptor contains five groups of properties: physical chain (carrier, encoding, detector, modulation), acquisition parameters (shots per datacube, frame rate, resolution, dimensionality), solver performance (best method, PSNR, SSIM, latency, algorithm type coverage), operational requirements (cost, operator skill level, solver compute) and sample compatibility (contact, destructive, in-vivo). Additionally, four modality-specific mismatch parameters capture calibration sensitivity.

A critical design principle is **value neutrality**: the catalog records only documented facts, not subjective quality scores. Every field is traceable to a published source. This separation between data (Layer A) and evaluation (Layer B) ensures that the catalog remains useful under different evaluation criteria.

### Task-Normalized Adequacy scoring

Standard benchmarks rank systems by absolute metrics such as PSNR, which implicitly assumes that higher is always better and that scores are commensurable across modalities. This creates two distortions: a system with 100 nm resolution receives a higher spatial score than one with 10 µm, even if the task requires only 50 µm; and a 36 dB PSNR on MRI and 36 dB on cryo-EM appear equivalent despite representing fundamentally different reconstruction difficulties.

TNA addresses both problems by scoring each system relative to user-specified task requirements (Fig. 3a). For each of eight adequacy dimensions *d*, TNA maps a system's capability to a [0, 10] scale using three reference points defined by the task query:

*r*_min (hard floor, below which the system is infeasible → TNA = 0),
*r*_target (meets the requirement → TNA = 5), and
*r*_comfort (exceeds with margin → TNA = 10).

Interpolation between reference points is piecewise linear. Systems that vastly exceed requirements score the same as those that comfortably meet them, because overkill is not rewarded — a key property that prevents expensive high-end systems from dominating recommendations for modest tasks.

The eight dimensions (Fig. 3b) are: acquisition feasibility (D1), temporal adequacy (D2), spatial adequacy (D3), observable sufficiency (D4), output recovery quality (D5), budget feasibility (D6), deployment burden (D7) and sample compatibility (D8). For D5, scores are **modality-normalized**: a system's PSNR is evaluated relative to the best and worst known results for that specific modality in the algorithm catalog, rather than on an absolute scale. This ensures fair comparison between inherently easy and inherently difficult inverse problems. For direct-readout systems with no reconstruction step (for example, high-speed CMOS cameras), D5 defaults to detector signal-to-noise adequacy.

### Three-stage evaluation protocol

The evaluation protocol separates three logically distinct operations (Fig. 4).

**Stage 1: Feasibility gate.** Each task query defines hard constraints — budget ceiling, minimum resolution, required frame rate, acquisition mode, sample handling restrictions. A system that violates any hard constraint is rejected with a diagnostic indicating which constraint(s) failed. This binary gate ensures that infeasible systems never appear in rankings regardless of their other merits.

**Stage 2: Pareto ranking.** Among feasible systems, we identify the Pareto frontier: the set of systems for which no other feasible system is equal or better on all eight TNA dimensions and strictly better on at least one. This stage requires no scalar aggregation or weight specification, preserving the multi-objective nature of the decision. In our experiments, the Pareto set typically contained 4–12 systems (Extended Data Fig. 1), a manageable number for human review.

**Stage 3: Preference weighting.** When the Pareto set is large or the user has clear priorities, optional application-specific weights produce a scalar preference score: *S*_pref = Σ_d *w*_d · TNA_d. Weights are user-specified or drawn from predefined profiles (for example, clinical-screening weights emphasize D6 and D7; research weights emphasize D3 and D5). Importantly, weighted scores serve only as tie-breakers within the Pareto set — they never override Pareto dominance.

### Pilot studies

We evaluated all 168 systems against three application profiles representing distinct use cases (Fig. 5).

**Clinical screening.** A hospital seeks an imaging system for emergency triage, requiring: budget ≤ \$500 K, operator skill ≤ technician, in-vivo capability, non-contact measurement and reconstruction latency ≤ 10 s. Of 168 systems, 41 pass the feasibility gate. The Pareto frontier contains 8 systems, with B-mode ultrasound ranking first in preference-weighted score owing to real-time operation, low cost (\$30 K) and technician-level workflow. CT appears on the frontier but ranks lower because of higher cost and the requirement for radiological expertise in interpretation. Non-obvious alternatives include optical coherence tomography (OCT), which is Pareto-optimal owing to its high spatial resolution at moderate cost, and fundus photography, optimized for ophthalmic screening (Fig. 5a).

**High-speed research.** A combustion laboratory needs to capture a non-repeatable transient event requiring: single-shot acquisition, ≥ 10 million frames per second (Mfps), budget ≤ \$50 K, non-contact and offline reconstruction within 60 s. This highly constrained query reduces 168 systems to 3 candidates. Only coded aperture compressive temporal imaging (CACTI)^4^ passes all constraints; compressed ultrafast photography (CUP)^24^ fails on budget (\$80 K) and streak cameras fail on both budget and single-shot 2D capability. Among solvers for CACTI, EfficientSCI^25^ provides the best quality–speed trade-off (33.1 dB, 2.1 s), whereas DiffusionSCI^26^ achieves higher quality (35.8 dB) but violates the latency constraint at 180 s (Fig. 5b). This demonstrates how PWM-SyS integrates solver selection with system selection.

**Industrial non-destructive testing (NDT).** A manufacturer requires defect detection in composite panels: non-destructive, non-contact, operator ≤ technician, budget ≤ \$50 K, throughput ≥ 1 part per minute. Twelve systems are feasible; 4 are Pareto-optimal. Active thermography^27^ ranks first, combining full-field infrared capture with fast reconstruction (\$30 K, 2 s per inspection). Shearography and eddy current testing are Pareto-optimal alternatives offering different resolution–cost–depth trade-offs. Notably, ultrasonic phased array, the dominant NDT technology, is Pareto-optimal but requires contact coupling — if the manufacturer later adds a non-contact hard constraint, the Pareto set shrinks to 3 systems (Fig. 5c).

These pilot studies illustrate three properties of PWM-SyS. First, the feasibility gate provides dramatic dimensionality reduction (168 → 3 in the high-speed case), focusing analysis on viable options. Second, Pareto ranking identifies non-obvious alternatives that differ from conventional modality-centric reasoning (for example, OCT for clinical screening). Third, integrating solver selection with system selection reveals cases where the best algorithm is infeasible (DiffusionSCI) and a different solver changes the recommendation.

### Interactive platform

PWM-SyS is deployed as a web platform at pwm.platformai.org (Fig. 6). Users interact through three interfaces: a natural-language query interface in SpecLab that accepts requirements in plain English and returns ranked recommendations with 8-dimension adequacy profiles; a browsable system catalog at `/benchmark/system-design` with sortable tables for all 168 systems organized by category; and per-modality benchmark pages where each system's descriptor is linked to its reconstruction leaderboard and challenge datasets. The recommendation pipeline completes in under 100 ms, enabling real-time exploration of constraint sensitivity by iteratively adjusting requirements.

---

## Discussion

PWM-SyS introduces purpose-conditioned evaluation to computational imaging — a field where benchmarking has historically been algorithm-centric. Three aspects of the framework merit discussion.

**Task-relative scoring.** By anchoring scores to user-specified requirements rather than absolute capability, TNA avoids the distortions inherent in universal rankings. A cryo-EM system and an ultrasound system occupy different regions of the capability space; asking which is "better" without specifying a task is meaningless^28^. TNA formalizes this intuition and makes the dependence on task specification explicit. This design parallels the shift from absolute to relative fitness evaluation in evolutionary optimization^29^ and from universal to task-specific evaluation in machine learning^30^.

**Separation of concerns.** The three-layer architecture (descriptor, protocol, visualization) and three-stage protocol (gate, Pareto, preference) provide clean separation between data, evaluation logic and presentation. Adding a new modality requires only a descriptor; adding a new evaluation criterion requires only a new TNA dimension; changing the user interface has no effect on evaluation results. This modularity makes the framework extensible by the community, similar to how modular benchmark designs in genomics^31^ and natural-language processing^32^ have outlived their original implementations.

**Comparison with existing approaches.** The closest relatives of PWM-SyS are MCDA frameworks in health technology assessment^18–21^, imaging system comparison reviews^13,14^ and reconstruction benchmarks^10–12^. PWM-SyS differs from MCDA tools by embedding domain-specific physics knowledge (forward models, reconstruction metrics, sample constraints) directly into the evaluation dimensions, and from reconstruction benchmarks by evaluating across rather than within modalities (Extended Data Table 2). To our knowledge, no existing framework provides structured, purpose-conditioned system-solver evaluation across the full scope of computational imaging.

**Limitations.** The catalog currently relies on published specifications and PWM benchmark results on synthetic data. For modalities where empirical benchmark data is sparse, D5 scores may be less well calibrated. The TNA reference points (*r*_min, *r*_target, *r*_comfort) are derived from task specifications; poorly specified tasks yield less discriminative evaluations. Future versions will incorporate empirically measured robustness scores using PWM's mismatch perturbation framework^23^, expand solver operating-point data through systematic latency profiling on standardized hardware, and add community contribution mechanisms for system descriptors, task queries and real-world validation case studies.

**Broader impact.** PWM-SyS complements the flagship PWM framework^23^, whose 11-primitive basis and 3-gate decomposition provide the theoretical foundation for system descriptors. Together they establish a unified framework spanning from physics-based forward modeling to purpose-conditioned system selection. We anticipate that PWM-SyS will be most immediately useful for three communities: imaging system designers evaluating hardware–algorithm trade-offs, procurement committees selecting clinical or industrial imaging equipment, and researchers exploring unfamiliar modalities outside their domain expertise.

---

## Online Methods

### System descriptor generation

System descriptors were generated by merging three data sources. (1) The PWM Modality Catalog (168 entries) provides carrier type, forward model directed acyclic graph (DAG) specification using an 11-primitive operator algebra^23^, mismatch parameters, category classification and display names. (2) The PWM Algorithm Catalog (1,367 entries, 592 unique algorithms) provides reconstruction benchmark results (PSNR, SSIM) organized by modality-specific variant overrides, carrier-based routing and category-level algorithm pools. Algorithms are classified into seven types: classical, compressed sensing, deep learning, diffusion, physics-informed, plug-and-play and transformer. (3) A hardware lookup table (168 entries) provides per-modality specifications: shots per datacube, maximum frame rate (Hz), spatial resolution (µm), output dimensionality, capital cost (k USD), operator skill level (4-tier ordinal: untrained, technician, expert, specialist) and solver computation latency (s). Hardware values were compiled from instrument manufacturer datasheets, published system descriptions and, where necessary, expert estimates with conservative bounds.

The generation script merges these sources into a unified JSON catalog (297 KB, 168 entries). Each entry is keyed by a unique modality identifier (for example, `cacti`, `cryo_em`, `oct`) and contains 32 fields organized into five groups: physical chain, acquisition, solver, operations and sample compatibility.

### Feasibility gate

The feasibility gate implements binary constraint checking across nine constraint types: budget ceiling (capital cost ≤ threshold), spatial resolution (system resolution ≤ required resolution), temporal resolution (frame rate ≥ required rate), acquisition mode (single-shot if required), contact mode (non-contact if required), in-vivo capability, operator skill ceiling (4-tier ordinal comparison), non-destructive measurement and reconstruction latency. Each constraint is evaluated independently; a system passes only if all applicable constraints are satisfied. Failed constraints are recorded with diagnostic messages indicating the binding constraint and the margin of failure.

### TNA computation

For each of eight adequacy dimensions, TNA maps a system's measured capability to a [0, 10] score using piecewise linear interpolation with three task-defined reference points. For "higher is better" dimensions (frame rate, PSNR), the mapping is:

TNA_d = 0 if *v* < *r*_min; TNA_d = 5 × (*v* − *r*_min) / (*r*_target − *r*_min) if *r*_min ≤ *v* < *r*_target; TNA_d = 5 + 5 × (*v* − *r*_target) / (*r*_comfort − *r*_target) if *r*_target ≤ *v* < *r*_comfort; TNA_d = 10 if *v* ≥ *r*_comfort.

For "lower is better" dimensions (cost, resolution), the mapping is inverted: lower values receive higher scores. For D5 (output recovery quality), the system's best PSNR is first modality-normalized to a [0, 1] scale using the minimum and maximum PSNR values across all algorithms benchmarked on that modality in the PWM algorithm catalog, then mapped to the [0, 10] TNA scale relative to the task's quality requirement.

### Pareto ranking

Given *n* feasible systems with 8-dimensional TNA score vectors, Pareto dominance is computed by pairwise comparison in O(*n*²) time. System *a* dominates system *b* if and only if *a*_d ≥ *b*_d for all *d* ∈ {1, …, 8} and *a*_d > *b*_d for at least one *d*. A system is Pareto-optimal if no other feasible system dominates it. For the full 168-system catalog, Pareto ranking completes in under 10 ms on a single CPU core.

### Preference weighting

Optional preference scores are computed as the weighted sum *S*_pref = Σ_d *w*_d · TNA_d, where weights *w*_d ≥ 0 sum to 1. Predefined profiles assign weights reflecting common application priorities: clinical screening (D6 = 0.25, D7 = 0.25, D8 = 0.20, others uniform), research (D3 = 0.25, D5 = 0.30, others uniform) and industrial (D1 = 0.20, D2 = 0.20, D6 = 0.20, D7 = 0.20, others uniform). Users can override weights interactively.

### Natural-language query parsing

The SpecLab integration detects system-level queries using keyword matching against a curated lexicon (for example, "which system," "best imaging for," "recommend," "under \$X," "single-shot"). Detected constraints are extracted via regular expression patterns for budget amounts (currency + number), resolution values (number + unit), frame rate requirements (number + fps/Hz) and qualitative flags (non-contact, in-vivo, non-destructive, operator skill). Parsed constraints are assembled into a TaskQuery object that is passed to the recommendation pipeline. Queries that do not match system-level patterns are forwarded to the LLM-based specification builder for modality-specific assistance.

### Platform architecture

PWM-SyS is implemented in Python 3.11 using the FastAPI web framework with Jinja2 templates and HTMX for interactive updates. The system catalog is loaded into memory at startup from JSON. The recommendation endpoint accepts HTTP POST requests with natural-language queries, executes constraint extraction, feasibility gating, TNA scoring and Pareto ranking, and returns HTML fragments containing feasibility tables, 8-dimension adequacy bar charts, solver operating-point comparisons and physics simulation buttons. The full pipeline executes in under 100 ms. The platform is deployed on Google Cloud Platform with PostgreSQL for user management and an Nginx reverse proxy.

### Statistics and reproducibility

All TNA scores are deterministic functions of system descriptors and task queries; no random sampling is involved in the evaluation protocol. Pareto ranking is a deterministic set operation. PSNR and SSIM values in the algorithm catalog are computed on standardized benchmark datasets using standard definitions (PSNR = 10 log₁₀(MAX² / MSE); SSIM per Wang et al.^33^). The pilot study results are fully reproducible by submitting the specified task queries to the platform.

---

## Data Availability

The system descriptor catalog (`system_catalog.json`, 168 entries) and all benchmark data are available at https://pwm.platformai.org/benchmark/system-design under a Creative Commons Attribution 4.0 International licence. The interactive recommendation tool is available at https://pwm.platformai.org/speclab.

## Code Availability

Source code for PWM-SyS is available in the Physics World Model repository at https://github.com/Shi-Labs/Physics-World-Model. Key components: catalog generation (`scripts/generate_system_catalog.py`), recommendation engine (`services/system_recommender.py`), SpecLab integration (`routers/spec_chat.py`) and full technical specification (`PROPOSAL.md`). The code is released under the MIT licence.

## Acknowledgements

[To be completed before submission.]

## Author Contributions

C.S. conceived the project, designed the TNA framework and evaluation protocol, curated the system descriptor catalog, implemented the platform and wrote the manuscript.

## Competing Interests

The author declares no competing interests.

---

## References

1. Barbastathis, G., Ozcan, A. & Situ, G. On the use of deep learning for computational imaging. *Optica* **6**, 921–943 (2019).
2. Bertero, M. & Boccacci, P. *Introduction to Inverse Problems in Imaging* (IOP Publishing, 1998).
3. Yuan, X. et al. Snapshot compressive imaging: theory, algorithms, and applications. *IEEE Trans. Pattern Anal. Mach. Intell.* **44**, 2191–2212 (2021).
4. Llull, P. et al. Coded aperture compressive temporal imaging. *Opt. Express* **21**, 10526–10545 (2013).
5. Adler, J. & Öktem, O. Learned primal-dual reconstruction. *IEEE Trans. Med. Imaging* **37**, 1322–1332 (2018).
6. Hammernik, K. et al. Learning a variational network for reconstruction of accelerated MRI data. *Magn. Reson. Med.* **79**, 3055–3071 (2018).
7. Zhong, E. D. et al. CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks. *Nat. Methods* **18**, 176–185 (2021).
8. Gustafsson, M. G. L. Surpassing the lateral resolution limit by a factor of two using structured illumination microscopy. *J. Microsc.* **198**, 82–87 (2000).
9. Betzig, E. et al. Imaging intracellular fluorescent proteins at nanometer resolution. *Science* **313**, 1642–1645 (2006).
10. Zbontar, J. et al. fastMRI: an open dataset and benchmarks for accelerated MRI. Preprint at https://arxiv.org/abs/1811.08839 (2018).
11. McCollough, C. H. et al. Low-dose CT for the detection and classification of metastatic liver lesions: results of the 2016 Low Dose CT Grand Challenge. *Med. Phys.* **44**, e339–e352 (2017).
12. Sage, D. et al. Super-resolution fight club: assessment of 2D and 3D single-molecule localization microscopy software. *Nat. Methods* **16**, 387–395 (2019).
13. Ongie, G. et al. Deep learning techniques for inverse problems in imaging. *IEEE J. Sel. Areas Inf. Theory* **1**, 39–56 (2020).
14. Liang, J. Punching holes in light: recent progress in single-shot coded-aperture optical imaging. *Rep. Prog. Phys.* **83**, 116101 (2020).
15. Goodman, J. W. *Introduction to Fourier Optics* (Roberts & Company, 2005).
16. Prince, J. L. & Links, J. M. *Medical Imaging Signals and Systems* (Pearson, 2006).
17. McCann, M. T., Jin, K. H. & Unser, M. Convolutional neural networks for inverse problems in imaging: a review. *IEEE Signal Process. Mag.* **34**, 85–95 (2017).
18. Marsh, K. et al. Multiple criteria decision analysis for health care decision making — emerging good practices: report 2 of the ISPOR MCDA Emerging Good Practices Task Force. *Value Health* **19**, 125–137 (2016).
19. Thokala, P. et al. Multiple criteria decision analysis for health care decision making — an introduction: report 1 of the ISPOR MCDA Emerging Good Practices Task Force. *Value Health* **19**, 1–13 (2016).
20. Goetghebeur, M. M. et al. Advancing the EVIDEM framework: facilitating development and review of health technology assessments. *Int. J. Technol. Assess. Health Care* **28**, 63–70 (2012).
21. Hansen, P. & Ombler, F. A new method for scoring additive multi-attribute value models using pairwise rankings of alternatives. *J. Multi-Criteria Decis. Anal.* **15**, 87–107 (2008).
22. Antun, V. et al. On instabilities of deep learning in image reconstruction and the potential costs of AI. *Proc. Natl Acad. Sci. USA* **117**, 30088–30098 (2020).
23. Shi, C. Eleven primitives and three gates: the universal structure of computational imaging. Preprint (2026).
24. Gao, L. et al. Single-shot compressed ultrafast photography at one hundred billion frames per second. *Nature* **516**, 74–77 (2014).
25. Wang, Z. et al. EfficientSCI: densely connected network with space-time factorization for large-scale video snapshot compressive imaging. *Proc. CVPR*, 18477–18486 (2023).
26. Meng, Z. et al. DiffusionSCI: generative diffusion model for snapshot compressive imaging. Preprint at https://arxiv.org/abs/2311.11725 (2023).
27. Maldague, X. P. V. *Theory and Practice of Infrared Technology for Nondestructive Testing* (Wiley, 2001).
28. Hand, D. J. Classifier technology and the illusion of progress. *Stat. Sci.* **21**, 1–14 (2006).
29. Deb, K. et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans. Evol. Comput.* **6**, 182–197 (2002).
30. Raji, I. D. et al. AI and the everything in the whole wide world benchmark. In *Proc. NeurIPS Datasets and Benchmarks Track* (2021).
31. Mangul, S. et al. Systematic benchmarking of omics computational tools. *Nat. Commun.* **10**, 1393 (2019).
32. Wang, A. et al. GLUE: a multi-task benchmark and analysis platform for natural language understanding. Preprint at https://arxiv.org/abs/1804.07461 (2018).
33. Wang, Z. et al. Image quality assessment: from error visibility to structural similarity. *IEEE Trans. Image Process.* **13**, 600–612 (2004).
34. Lustig, M., Donoho, D. & Pauly, J. M. Sparse MRI: the application of compressed sensing for rapid MR imaging. *Magn. Reson. Med.* **58**, 1182–1195 (2007).
35. Candès, E. J. & Wakin, M. B. An introduction to compressive sampling. *IEEE Signal Process. Mag.* **25**, 21–30 (2008).
36. Duarte, M. et al. Single-pixel imaging via compressive sampling. *IEEE Signal Process. Mag.* **25**, 83–91 (2008).
37. Wagadarikar, A. et al. Single disperser design for coded aperture snapshot spectral imaging. *Appl. Opt.* **47**, B44–B51 (2008).
38. Mait, J. N., Euliss, G. W. & Athale, R. A. Computational imaging. *Adv. Opt. Photon.* **10**, 409–483 (2018).
39. Bostan, E. et al. Deep phase decoder: self-calibrating phase microscopy with an untrained deep neural network. *Optica* **7**, 559–562 (2020).
40. Ulyanov, D., Vedaldi, A. & Lempitsky, V. Deep image prior. *Int. J. Comput. Vis.* **128**, 1867–1888 (2020).
41. Romano, Y., Elad, M. & Milanfar, P. The little engine that could: regularization by denoising (RED). *SIAM J. Imaging Sci.* **10**, 1804–1844 (2017).
42. Venkatakrishnan, S. V., Bouman, C. A. & Wohlberg, B. Plug-and-play priors for model based reconstruction. In *Proc. IEEE GlobalSIP*, 945–948 (2013).
43. Sun, J. et al. Block coordinate regularization by denoising. *IEEE Trans. Comput. Imaging* **6**, 908–921 (2020).
44. Agustsson, E. & Timofte, R. NTIRE 2017 challenge on single image super-resolution: dataset and study. In *Proc. CVPR Workshops* (2017).
45. Tian, C. et al. Deep learning on image denoising: an overview. *Neural Netw.* **131**, 251–275 (2020).

---

## Figure Legends

**Figure 1 | PWM-SyS overview.** **a**, Conventional imaging benchmarks fix the system and rank algorithms (left). PWM-SyS evaluates both system and solver jointly against task requirements (right). **b**, Three-layer architecture: Layer A (system descriptor catalog) provides neutral facts; Layer B (evaluation protocol) applies TNA scoring and Pareto ranking; Layer C (interactive platform) enables natural-language querying and simulation. **c**, Workflow: the user specifies a task query with requirements and constraints, PWM-SyS filters infeasible systems, scores feasible ones across eight TNA dimensions, identifies the Pareto frontier, and returns ranked recommendations.

**Figure 2 | System descriptor catalog.** **a**, Each of 168 system descriptors contains five property groups: physical chain, acquisition, solver, operations and sample compatibility. Example shown for CACTI (coded aperture compressive temporal imaging). **b**, Distribution of 168 systems across 5 carrier families and 19 application categories. Bar chart shows category counts; pie chart shows carrier distribution. **c**, Scatter plot of capital cost versus best PSNR for all 168 systems, coloured by carrier type. Horizontal dashed lines indicate typical budget ceilings for clinical (\$500 K) and research (\$50 K) deployments.

**Figure 3 | Task-Normalized Adequacy scoring.** **a**, TNA maps system capability to a [0, 10] scale relative to three task-defined reference points (*r*_min, *r*_target, *r*_comfort), shown for a "higher is better" dimension (frame rate). Systems below *r*_min score 0 (infeasible); those exceeding *r*_comfort score 10 (adequate with margin); interpolation is piecewise linear. **b**, The eight TNA dimensions, grouped by function: physical feasibility (D1–D4), reconstruction quality (D5), and practical constraints (D6–D8). **c**, Radar charts comparing TNA profiles of three representative systems (B-mode ultrasound, CT, cryo-EM) evaluated against a clinical screening task, illustrating how the same systems produce different profiles under different tasks.

**Figure 4 | Three-stage evaluation protocol.** **a**, Stage 1 (feasibility gate): 168 systems are filtered by hard constraints; example shows clinical screening query reducing candidates from 168 to 41. **b**, Stage 2 (Pareto ranking): feasible systems are ranked by Pareto dominance in 8-dimensional TNA space; example shows 8 Pareto-optimal systems (red) among 41 feasible (grey). **c**, Stage 3 (preference weighting): within the Pareto set, application-specific weights produce a scalar score for final ranking. Weights serve as tie-breakers and never override Pareto dominance.

**Figure 5 | Pilot study results.** **a**, Clinical screening (budget ≤ \$500 K, technician-operable, in-vivo): 41/168 feasible, 8 Pareto-optimal. Heatmap shows TNA scores (D1–D8) for the top 5 systems. B-mode ultrasound ranks first. **b**, High-speed research (single-shot, ≥ 10 Mfps, budget ≤ \$50 K): 3/168 candidates, only CACTI passes all constraints. Solver comparison shows EfficientSCI as the recommended solver (DiffusionSCI fails latency). **c**, Industrial NDT (non-destructive, non-contact, technician, ≤ \$50 K): 12/168 feasible, 4 Pareto-optimal. Active thermography ranks first. Bar charts show constraint-failure analysis for rejected systems.

**Figure 6 | Interactive platform.** **a**, SpecLab natural-language interface: user enters "Which imaging system for clinical screening under \$500K?" and receives ranked system recommendations with 8-dimension adequacy profiles. **b**, System design catalog page showing sortable tables of all 168 systems organized by category. **c**, Per-modality benchmark page linking system descriptor to reconstruction leaderboard and challenge datasets. Screenshots from pwm.platformai.org.

---

## Extended Data

### Extended Data Table 1 | Full system descriptor schema

| Field | Type | Description | Example (CACTI) |
|-------|------|-------------|-----------------|
| id | string | Unique modality identifier | cacti |
| display_name | string | Human-readable name | CACTI |
| category | string | Application category | compressive |
| carrier | string | Physical carrier | photon |
| shots_per_datacube | integer | Acquisitions per datacube | 1 |
| max_fps_hz | float | Maximum frame rate | 1.0 × 10⁸ |
| spatial_resolution_um | float | Spatial resolution (µm) | 10.0 |
| output_dims | string | Output dimensionality | 3D (x, y, t) |
| capital_cost_k_usd | float | Capital cost (k USD) | 15 |
| operator_skill | string | Required operator level | technician |
| best_method | string | Best reconstruction algorithm | EfficientSCI |
| best_psnr_db | float | Best PSNR on PWM benchmark | 38.6 |
| best_ssim | float | Best SSIM | 0.961 |
| solver_latency_s | float | Reconstruction latency | 2.1 |
| contact_required | boolean | Requires sample contact | false |
| destructive | boolean | Destroys sample | false |
| in_vivo | boolean | Compatible with living samples | false |
| observable | string | Measured physical quantity | intensity(x,y,t) |
| num_algorithms | integer | Algorithms in catalog | 8 |

### Extended Data Table 2 | Comparison with existing evaluation frameworks

| Framework | Scope | Cross-modality | Task-conditioned | System+solver | Constraint filtering | Interactive |
|-----------|-------|----------------|------------------|---------------|---------------------|-------------|
| fastMRI^10^ | MRI | No | No | No | No | No |
| AAPM CT^11^ | CT | No | No | No | No | No |
| SMLM challenge^12^ | SMLM | No | No | No | No | No |
| DIV2K^44^ | Natural images | No | No | No | No | No |
| EVIDEM^20^ | Health tech | Yes | Yes | No | No | Partial |
| PAPRIKA^21^ | General MCDA | Yes | Yes | No | No | Yes |
| **PWM-SyS** | **168 modalities** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### Extended Data Figure 1 | Pareto set sizes across task profiles

Distribution of Pareto-optimal set sizes for 50 randomly generated task profiles with varying constraint stringency. Median Pareto set size is 7 systems (interquartile range: 4–12), confirming that the Pareto frontier typically reduces the feasible set to a manageable number of candidates for human review.

### Extended Data Table 3 | Category summary statistics

| Category | Count | Avg PSNR (dB) | Avg cost (k USD) | Single-shot (%) |
|----------|-------|---------------|-------------------|-----------------|
| Astronomy | 4 | 30.4 | 126,762 | 25 |
| Coherent | 5 | 35.1 | 96 | 60 |
| Compressive | 4 | 38.6 | 34 | 100 |
| Computational Photography | 5 | 35.8 | 2 | 80 |
| Depth Imaging | 5 | 35.4 | 16 | 60 |
| Electron Microscopy | 11 | 33.2 | 1,609 | 27 |
| Experimental Science | 11 | 32.9 | 106,511 | 18 |
| Industrial Inspection | 10 | 35.6 | 75 | 40 |
| Medical | 37 | 37.5 | 1,054 | 30 |
| Microscopy | 24 | 36.8 | 216 | 42 |
| Multi-Modal Fusion | 6 | 34.2 | 2,183 | 17 |
| Neural Rendering | 2 | 35.9 | 1 | 100 |
| Quantum | 3 | 29.6 | 103 | 33 |
| Remote Sensing | 11 | 34.3 | 14,052 | 36 |
| Scanning Probe | 4 | 32.5 | 142 | 0 |
| Scientific Instrumentation | 12 | 32.8 | 18,208 | 17 |
| Spectroscopy | 8 | 33.0 | 312 | 38 |
| Ultrafast | 4 | 32.9 | 125,145 | 75 |

### Extended Data Table 4 | Carrier distribution

| Carrier | Count | Representative systems |
|---------|-------|----------------------|
| Photon | 72 | CACTI, CASSI, confocal, SIM, PALM/STORM, OCT, light-field |
| X-ray | 20 | CT, mammography, angiography, SAXS, XFEL, micro-CT |
| Electron | 14 | SEM, TEM, cryo-EM, EELS, electron holography, cryo-ET |
| Acoustic | 12 | B-mode ultrasound, SAM, acoustic emission, sonar, PAT |
| Spin/RF | 10 | MRI, fMRI, ASL-MRI, CEST-MRI, MR spectroscopy |
| RF | 9 | SAR, InSAR, GPR, radio astronomy, EHT |
| Ion | 4 | Atom probe, DESI, MALDI-MSI, SIMS |
| Gamma | 4 | PET, SPECT, PET/CT, PET/MR |
| Other | 23 | IR thermography, neutron, proton, magnetic, mechanical |

---

## Reporting Summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article.
