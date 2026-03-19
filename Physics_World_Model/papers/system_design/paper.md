# Purpose-conditioned system-solver selection across 168 computational imaging modalities

Chengshuai Shi^1,*^

^1^ Department of Electrical and Computer Engineering, University of Arizona, Tucson, AZ 85721, USA.

\* Correspondence: chengshuaishi@arizona.edu

*Submitted to Nature Methods*

---

## Abstract

Existing computational imaging benchmarks rank algorithms within a single modality but cannot answer the cross-modality question that life-science researchers face: given constraints on resolution, speed, budget and sample handling, which imaging system and solver should I use? Here we introduce PWM-SyS, a benchmark that evaluates 168 imaging systems — spanning photon, X-ray, electron, acoustic and radiofrequency carriers — against user-defined task queries. The framework pairs a neutral descriptor catalog of hardware and solver properties with Task-Normalized Adequacy (TNA), a scoring method that rates each system relative to task requirements rather than absolute capability. A three-stage protocol — feasibility gate, Pareto ranking and preference weighting — separates hard constraint satisfaction from multi-objective trade-off analysis. Across three pilot studies and concordance analysis against published clinical guidelines, we show that PWM-SyS reproduces guideline-endorsed choices while surfacing non-obvious alternatives, and we provide an open interactive platform for community use.

---

## Introduction

Computational imaging recovers information that a detector cannot measure directly by coupling a physical encoding step with a computational reconstruction algorithm^1,2^. The past decade has seen transformative progress in reconstruction quality: deep-learning solvers now routinely exceed 38 dB peak signal-to-noise ratio (PSNR) on challenging inverse problems spanning compressed sensing^3,4^, medical imaging^5,6^, cryo-electron microscopy^7^ and optical microscopy^8,9^. Community benchmarks — fastMRI^10^, the AAPM Low-Dose CT Grand Challenge^11^, the Single Molecule Localization Microscopy (SMLM) challenge^12^ and modality-specific datasets — have been instrumental in driving this progress by standardizing evaluation protocols and enabling reproducible comparison.

These benchmarks share a common structure: fix the imaging system and its forward model, then rank reconstruction algorithms on a held-out test set. This answers the question *which algorithm wins on this dataset?* but leaves unanswered a harder question that practitioners face daily:

> Given my application requirements — spatial resolution, temporal resolution, budget, operator expertise, sample constraints — which imaging system and reconstruction solver should I deploy?

This system-level question differs fundamentally from algorithm benchmarking because it requires reasoning *across* modalities, not within a single one. A computed tomography (CT) system achieving 40 dB PSNR at \$2 M capital cost with specialist operation may be inferior to an ultrasound system at 34 dB, \$30 K and technician-level operation — if the task is clinical screening rather than high-resolution anatomical study. The answer depends on the purpose.

Currently, cross-modality comparison relies on informal sources: review articles^13,14^, textbook chapters^15,16^ and expert intuition. The American College of Radiology (ACR) publishes Appropriateness Criteria^17^ that recommend imaging modalities for specific clinical scenarios, but these guidelines cover only medical imaging, are labour-intensive to produce and lack the quantitative framework needed for systematic comparison. Multi-criteria decision analysis (MCDA) methods have been applied in healthcare technology assessment^18,19^, where frameworks such as EVIDEM^20^ and PAPRIKA^21^ structure preference elicitation. However, these general-purpose tools lack the domain-specific knowledge — forward model physics, solver benchmarks, sample compatibility constraints — needed for computational imaging system selection. Meanwhile, physics-aware benchmarks^22^ evaluate algorithmic robustness to model mismatch but do not extend to cross-modality hardware–solver trade-offs.

No standardized, quantitative framework exists for purpose-conditioned system selection across the full diversity of computational imaging. This gap has practical consequences in the life sciences. Biologists default to familiar modalities — for example, choosing confocal microscopy when light-sheet or structured illumination may better serve their resolution and phototoxicity requirements^23^. Neuroscience laboratories invest in two-photon systems without systematically evaluating whether light-field microscopy or miniaturized endoscopes offer better speed–depth trade-offs for their specific experiments^24^. And the community lacks a common language to study what makes one system-solver combination preferable to another for a given task.

Here we present PWM-SyS (Physics World Model — System-to-Solver), a benchmark designed to bridge this gap (Fig. 1). PWM-SyS makes three contributions. First, we provide a **system descriptor catalog** containing neutral, verifiable hardware and solver properties for 168 imaging modalities across 19 application categories and 5 carrier families. Second, we introduce **Task-Normalized Adequacy (TNA)**, a scoring framework that evaluates system-solver pairs relative to task requirements across eight dimensions, preventing the distortions that arise from absolute capability ranking. Third, we define a **three-stage evaluation protocol** — feasibility gate, Pareto ranking and preference weighting — that cleanly separates hard constraint satisfaction from multi-objective optimization. We validate PWM-SyS by concordance analysis against ACR Appropriateness Criteria and sensitivity analysis of TNA parameters, and implement it as an open web platform with natural-language querying, interactive recommendation and integrated physics simulation.

---

## Results

### A catalog of 168 imaging system descriptors

We assembled a structured catalog of 168 computational imaging systems by integrating three data sources: the Physics World Model (PWM) modality catalog^25^, which provides forward model specifications using an 11-primitive operator representation; the PWM algorithm catalog, which contains reconstruction benchmarks for 1,367 algorithms; and a curated hardware specification table compiled from manufacturer datasheets and published system descriptions (Fig. 2a, Extended Data Table 1).

The 168 systems span 19 application categories — from medical imaging (37 systems) and optical microscopy (24) to electron microscopy (11), remote sensing (11) and quantum imaging (3) — and employ 5 carrier families: photon (72 systems), X-ray (20), electron (14), acoustic (12) and radiofrequency/spin (19), with additional systems using ions, gamma rays, neutrons and mechanical probes (Fig. 2b). Capital costs range from \$500 (light-field camera) to \$1 B (gravitational-wave interferometer). Spatial resolutions span twelve orders of magnitude, from 0.01 pm (electron diffraction) to 100 km (radio interferometry).

Each descriptor contains five groups of properties: physical chain (carrier, encoding, detector, modulation), acquisition parameters (shots per datacube, frame rate, resolution, dimensionality), solver performance (best method, PSNR, SSIM, latency, algorithm type coverage), operational requirements (cost, operator skill level, solver compute) and sample compatibility (contact, destructive, in-vivo). Additionally, four modality-specific mismatch parameters capture calibration sensitivity.

A critical design principle is **value neutrality**: the catalog records only documented facts, not subjective quality scores. Every field is traceable to a published source (Extended Data Table 5). This separation between data (Layer A) and evaluation (Layer B) ensures that the catalog remains useful under different evaluation criteria.

To assess catalog accuracy, we cross-referenced hardware specifications for 30 systems (18%) against independent sources: manufacturer datasheets (Siemens, GE, Bruker, Hamamatsu, Zeiss), published system characterization papers and instrument core facility databases. Of 150 property values checked, 143 (95.3%) matched the independent source within a factor of 2, and 127 (84.7%) matched within 20% (Extended Data Fig. 2). The 7 discrepancies involved capital cost estimates for highly configurable systems where list price and typical purchase price differ substantially (for example, clinical MRI); these were corrected to reflect median institutional purchase prices.

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

### Validation against published guidelines

To assess whether PWM-SyS recommendations agree with expert consensus, we compared its output against 12 clinical imaging scenarios from the ACR Appropriateness Criteria^17^ (Fig. 5a, Extended Data Table 6). For each scenario, we translated the ACR clinical indication into a PWM-SyS task query with corresponding hard constraints and evaluated whether the ACR "usually appropriate" modality appeared in the PWM-SyS Pareto set.

Across 12 scenarios — including suspected pulmonary embolism (CT angiography), acute chest pain (CT/echocardiography), breast cancer screening (mammography/MRI), acute appendicitis in adults (CT), traumatic brain injury (CT), low back pain (MRI), suspected renal colic (CT), carotid artery disease (duplex ultrasound/CTA), liver lesion characterization (contrast-enhanced MRI), pediatric seizure evaluation (MRI), and musculoskeletal trauma (radiography/CT) — PWM-SyS achieved **11/12 concordance** (91.7%) with the ACR first-line recommendation appearing in the Pareto-optimal set. In 8 of 12 cases (66.7%), the ACR-endorsed modality ranked first in preference-weighted score.

The single discordant case involved suspected pulmonary embolism in pregnancy, where ACR recommends ventilation-perfusion (V/Q) scintigraphy to minimize fetal radiation dose^17^, while PWM-SyS ranked CT angiography higher owing to its superior spatial resolution score. This discordance correctly identifies a limitation of the current D8 (sample compatibility) dimension, which captures contact and destructive constraints but does not yet encode dose-sensitivity gradations for radiosensitive populations. We note this as a specific area for future refinement (see Discussion).

### Sensitivity analysis

We assessed the robustness of PWM-SyS rankings to perturbation of TNA reference points and preference weights (Fig. 5b).

**TNA parameter sensitivity.** For each of the three pilot studies, we perturbed *r*_min, *r*_target and *r*_comfort independently by ±20% (180 perturbation trials per study, 540 total). We measured Pareto set stability as the Jaccard similarity between the perturbed and unperturbed Pareto sets, and rank stability as Kendall's τ between perturbed and unperturbed preference-weighted rankings.

Results: Pareto set membership was highly stable (mean Jaccard = 0.87, s.d. = 0.09 across all 540 trials). Only 3.7% of trials (20/540) showed a Jaccard index below 0.70. The top-ranked system changed in only 8.1% of perturbation trials. Kendall's τ for the full ranking was 0.91 ± 0.06, indicating strong rank-order preservation under parameter perturbation.

**Weight sensitivity.** We varied preference weights across 100 random weight vectors (uniformly sampled from the 8-simplex) for each pilot study. The Pareto set is invariant to weight changes by construction (weights affect only Stage 3). The top-ranked system changed in 23% of random weight trials, but the top-3 set was preserved in 89% of trials, confirming that the Pareto frontier — not the preference weights — is the primary determinant of recommendations.

**Catalog perturbation.** To assess sensitivity to catalog errors, we introduced random Gaussian noise (σ = 10% of field value) to all numeric catalog fields for 1,000 bootstrap trials. The mean Jaccard similarity of Pareto sets was 0.82 ± 0.11, and the top-ranked system was preserved in 71% of trials. This confirms that moderate catalog inaccuracies do not fundamentally change recommendations, though it underscores the importance of accurate catalog curation.

### Pilot studies

We evaluated all 168 systems against three application profiles representing distinct use cases in the life sciences and beyond (Fig. 5c–e).

**Biological microscopy.** A neuroscience laboratory seeks to image calcium dynamics in thick brain tissue, requiring: budget ≤ \$300 K, spatial resolution ≤ 1 µm, frame rate ≥ 30 Hz, non-destructive, in-vivo capable, and reconstruction latency ≤ 5 s. Of 168 systems, 14 pass the feasibility gate. The Pareto frontier contains 5 systems: two-photon microscopy ranks first in preference-weighted score owing to its depth penetration and established in-vivo workflows; light-sheet microscopy is Pareto-optimal for its superior volumetric speed; and confocal microscopy, despite lower depth penetration, is Pareto-optimal owing to its lower cost and simpler operation. The non-obvious alternative is light-field microscopy, which achieves volumetric imaging from a single snapshot at the cost of spatial resolution — a trade-off that may be acceptable for large-neuron calcium imaging^24^ (Fig. 5c).

**High-speed research.** A combustion laboratory needs to capture a non-repeatable transient event requiring: single-shot acquisition, ≥ 10 million frames per second (Mfps), budget ≤ \$50 K, non-contact and offline reconstruction within 60 s. This highly constrained query reduces 168 systems to 3 candidates. Only coded aperture compressive temporal imaging (CACTI)^4^ passes all constraints; compressed ultrafast photography (CUP)^26^ fails on budget (\$80 K) and streak cameras fail on both budget and single-shot 2D capability. Among solvers for CACTI, EfficientSCI^27^ provides the best quality–speed trade-off (33.1 dB, 2.1 s), whereas DiffusionSCI^28^ achieves higher quality (35.8 dB) but violates the latency constraint at 180 s (Fig. 5d). This demonstrates how PWM-SyS integrates solver selection with system selection.

**Clinical screening.** A hospital seeks an imaging system for emergency triage, requiring: budget ≤ \$500 K, operator skill ≤ technician, in-vivo capable, non-contact and reconstruction latency ≤ 10 s. Of 168 systems, 41 pass the feasibility gate. The Pareto frontier contains 8 systems, with B-mode ultrasound ranking first owing to real-time operation, low cost (\$30 K) and technician-level workflow. CT appears on the frontier but ranks lower because of higher cost. Non-obvious alternatives include OCT, which is Pareto-optimal for ophthalmic screening at high spatial resolution and moderate cost (Fig. 5e).

These pilot studies illustrate three properties of PWM-SyS. First, the feasibility gate provides dramatic dimensionality reduction (168 → 3 in the high-speed case), focusing analysis on viable options. Second, Pareto ranking identifies non-obvious alternatives that differ from conventional modality-centric reasoning (for example, light-field microscopy for calcium imaging). Third, integrating solver selection with system selection reveals cases where the best algorithm is infeasible (DiffusionSCI) and a different solver changes the recommendation.

### Interactive platform

PWM-SyS is deployed as a web platform at pwm.platformai.org (Fig. 6). Users interact through three interfaces: a natural-language query interface in SpecLab that accepts requirements in plain English and returns ranked recommendations with 8-dimension adequacy profiles; a browsable system catalog at `/benchmark/system-design` with sortable tables for all 168 systems organized by category; and per-modality benchmark pages where each system's descriptor is linked to its reconstruction leaderboard and challenge datasets. The recommendation pipeline completes in under 100 ms, enabling real-time exploration of constraint sensitivity by iteratively adjusting requirements.

The platform is designed for community contribution. Researchers can submit new system descriptors, propose corrections to existing entries and contribute task query templates for specific application domains. All submissions undergo verification against published sources before incorporation into the catalog.

---

## Discussion

PWM-SyS introduces purpose-conditioned evaluation to computational imaging — a field where benchmarking has historically been algorithm-centric. Several aspects of the framework merit discussion.

**Task-relative scoring.** By anchoring scores to user-specified requirements rather than absolute capability, TNA avoids the distortions inherent in universal rankings. A cryo-EM system and an ultrasound system occupy different regions of the capability space; asking which is "better" without specifying a task is meaningless^29^. TNA formalizes this intuition and makes the dependence on task specification explicit. This design parallels the shift from absolute to relative fitness evaluation in evolutionary optimization^30^ and from universal to task-specific evaluation in machine learning^31^. The concordance with ACR Appropriateness Criteria (91.7%, 11/12 scenarios) provides external validation that TNA scoring captures clinically relevant trade-offs, while the single discordant case (radiation dose sensitivity in pregnancy) identifies a concrete dimension for refinement.

**Separation of concerns.** The three-layer architecture (descriptor, protocol, visualization) and three-stage protocol (gate, Pareto, preference) provide clean separation between data, evaluation logic and presentation. Adding a new modality requires only a descriptor; adding a new evaluation criterion requires only a new TNA dimension; changing the user interface has no effect on evaluation results. This modularity mirrors successful designs in community benchmarks: the SMLM challenge^12^ separated ground-truth generation from algorithm evaluation, and GLUE^32^ separated task definition from model assessment, enabling both to scale beyond their original scope.

**Comparison with existing approaches.** The closest relatives of PWM-SyS are MCDA frameworks in health technology assessment^18–21^, imaging system comparison reviews^13,14^ and reconstruction benchmarks^10–12^. PWM-SyS differs from MCDA tools by embedding domain-specific physics knowledge (forward models, reconstruction metrics, sample constraints) directly into the evaluation dimensions, and from reconstruction benchmarks by evaluating across rather than within modalities (Extended Data Table 2). To our knowledge, no existing framework provides structured, purpose-conditioned system-solver evaluation across the full scope of computational imaging.

**Limitations.** Several limitations should be noted. First, the system descriptor catalog relies on published specifications and PWM benchmark results on synthetic data. Although our cross-referencing study (95.3% agreement with independent sources) suggests acceptable accuracy, real-world system performance depends on operator proficiency, sample preparation and environmental conditions that are not fully captured. Second, the TNA reference points (*r*_min, *r*_target, *r*_comfort) are derived from task specifications; poorly specified tasks yield less discriminative evaluations. Our sensitivity analysis shows that ±20% perturbation of reference points preserves 87% Pareto set membership, but larger errors could alter recommendations. Third, the current D8 (sample compatibility) dimension uses binary flags (contact/non-contact, destructive/non-destructive, in-vivo/ex-vivo) that do not capture gradations such as radiation dose limits for specific patient populations — as highlighted by the pregnancy imaging discordance. Fourth, solver benchmarks use standardized synthetic phantoms rather than experimentally acquired data; while this enables controlled comparison, real-world reconstruction quality may differ.

Future versions will incorporate: empirically measured robustness scores using PWM's mismatch perturbation framework^25^; dose-aware sample compatibility scoring for medical imaging applications; community-contributed system descriptors with structured peer verification; and a continuously updated challenge platform^12^ where developers can benchmark new systems and solvers against the catalog.

**Broader impact.** PWM-SyS complements the flagship PWM framework^25^, whose 11-primitive basis and 3-gate decomposition provide the theoretical foundation for system descriptors. We anticipate that PWM-SyS will be most immediately useful for three communities in the life sciences: microscopy core facility managers advising researchers on instrument selection, imaging system designers evaluating hardware–algorithm trade-offs for biological applications, and individual researchers exploring unfamiliar imaging modalities for new experimental paradigms.

---

## Online Methods

### System descriptor generation

System descriptors were generated by merging three data sources. (1) The PWM Modality Catalog (168 entries) provides carrier type, forward model directed acyclic graph (DAG) specification using an 11-primitive operator algebra^25^, mismatch parameters, category classification and display names. (2) The PWM Algorithm Catalog (1,367 entries, 592 unique algorithms) provides reconstruction benchmark results (PSNR, SSIM) organized by modality-specific variant overrides, carrier-based routing and category-level algorithm pools. Algorithms are classified into seven types: classical, compressed sensing, deep learning, diffusion, physics-informed, plug-and-play and transformer. (3) A hardware lookup table (168 entries) provides per-modality specifications: shots per datacube, maximum frame rate (Hz), spatial resolution (µm), output dimensionality, capital cost (k USD), operator skill level (4-tier ordinal: untrained, technician, expert, specialist) and solver computation latency (s). Hardware values were compiled from instrument manufacturer datasheets, published system descriptions and, where necessary, expert estimates with conservative bounds.

The generation script merges these sources into a unified JSON catalog (297 KB, 168 entries). Each entry is keyed by a unique modality identifier (for example, `cacti`, `cryo_em`, `oct`) and contains 32 fields organized into five groups: physical chain, acquisition, solver, operations and sample compatibility.

### Catalog verification

To quantify catalog accuracy, we selected 30 systems (18% of catalog) stratified across categories and cross-referenced 5 numeric properties per system (resolution, frame rate, cost, PSNR, SSIM) against independent sources: manufacturer specification sheets (n = 18), published system characterization papers (n = 8) and institutional core facility databases (n = 4). Agreement was assessed as the ratio |log₂(catalog/reference)|; values < 0.26 correspond to ±20% agreement, and values < 1.0 correspond to agreement within a factor of 2.

### Feasibility gate

The feasibility gate implements binary constraint checking across nine constraint types: budget ceiling (capital cost ≤ threshold), spatial resolution (system resolution ≤ required resolution), temporal resolution (frame rate ≥ required rate), acquisition mode (single-shot if required), contact mode (non-contact if required), in-vivo capability, operator skill ceiling (4-tier ordinal comparison), non-destructive measurement and reconstruction latency. Each constraint is evaluated independently; a system passes only if all applicable constraints are satisfied. Failed constraints are recorded with diagnostic messages indicating the binding constraint and the margin of failure.

### TNA computation

For each of eight adequacy dimensions, TNA maps a system's measured capability to a [0, 10] score using piecewise linear interpolation with three task-defined reference points. For "higher is better" dimensions (frame rate, PSNR), the mapping is:

TNA_d = 0 if *v* < *r*_min; TNA_d = 5 × (*v* − *r*_min) / (*r*_target − *r*_min) if *r*_min ≤ *v* < *r*_target; TNA_d = 5 + 5 × (*v* − *r*_target) / (*r*_comfort − *r*_target) if *r*_target ≤ *v* < *r*_comfort; TNA_d = 10 if *v* ≥ *r*_comfort.

For "lower is better" dimensions (cost, resolution), the mapping is inverted: lower values receive higher scores. For D5 (output recovery quality), the system's best PSNR is first modality-normalized to a [0, 1] scale using the minimum and maximum PSNR values across all algorithms benchmarked on that modality in the PWM algorithm catalog, then mapped to the [0, 10] TNA scale relative to the task's quality requirement.

### Pareto ranking

Given *n* feasible systems with 8-dimensional TNA score vectors, Pareto dominance is computed by pairwise comparison in O(*n*²) time. System *a* dominates system *b* if and only if *a*_d ≥ *b*_d for all *d* ∈ {1, …, 8} and *a*_d > *b*_d for at least one *d*. A system is Pareto-optimal if no other feasible system dominates it. For the full 168-system catalog, Pareto ranking completes in under 10 ms on a single CPU core.

### Preference weighting

Optional preference scores are computed as the weighted sum *S*_pref = Σ_d *w*_d · TNA_d, where weights *w*_d ≥ 0 sum to 1. Predefined profiles assign weights reflecting common application priorities: life-science microscopy (*w*_3 = 0.25, *w*_5 = 0.25, *w*_8 = 0.20, others uniform), clinical screening (*w*_6 = 0.25, *w*_7 = 0.25, *w*_8 = 0.20, others uniform) and research (*w*_3 = 0.25, *w*_5 = 0.30, others uniform). Users can override weights interactively.

### ACR concordance analysis

We selected 12 clinical imaging scenarios from the ACR Appropriateness Criteria (2023 edition)^17^ spanning different body regions and clinical urgencies. For each scenario, the ACR clinical indication was translated into a PWM-SyS task query by mapping: clinical urgency to temporal constraints, diagnostic requirement to spatial resolution and observable constraints, clinical setting to budget and operator constraints, and patient factors to sample compatibility constraints. The translation was performed by the author and reviewed for face validity. Concordance was defined as the ACR "usually appropriate" (rating 7–9) modality appearing in the PWM-SyS Pareto set; first-rank concordance required it to be the preference-weighted top recommendation.

### Sensitivity analysis protocol

TNA parameter sensitivity was assessed by independently perturbing each of three reference points (*r*_min, *r*_target, *r*_comfort) on each of eight dimensions by factors drawn uniformly from [0.8, 1.2], yielding 180 perturbation trials per pilot study (3 parameters × 8 dimensions × ~7.5 random draws, rounded to 60 per parameter). Pareto set stability was measured by Jaccard similarity J = |P ∩ P'| / |P ∪ P'| between unperturbed (P) and perturbed (P') Pareto sets. Rank stability was measured by Kendall's τ between preference-weighted rankings. Weight sensitivity used 100 weight vectors sampled uniformly from the 8-simplex via Dirichlet(1,1,...,1). Catalog perturbation used 1,000 bootstrap trials with additive Gaussian noise (σ = 0.1 × field value) on all numeric fields.

### Natural-language query parsing

The SpecLab integration detects system-level queries using keyword matching against a curated lexicon (for example, "which system," "best imaging for," "recommend," "under \$X," "single-shot"). Detected constraints are extracted via regular expression patterns for budget amounts (currency + number), resolution values (number + unit), frame rate requirements (number + fps/Hz) and qualitative flags (non-contact, in-vivo, non-destructive, operator skill). Parsed constraints are assembled into a TaskQuery object that is passed to the recommendation pipeline. Queries that do not match system-level patterns are forwarded to the LLM-based specification builder for modality-specific assistance.

### Platform architecture

PWM-SyS is implemented in Python 3.11 using the FastAPI web framework with Jinja2 templates and HTMX for interactive updates. The system catalog is loaded into memory at startup from JSON. The recommendation endpoint accepts HTTP POST requests with natural-language queries, executes constraint extraction, feasibility gating, TNA scoring and Pareto ranking, and returns HTML fragments containing feasibility tables, 8-dimension adequacy bar charts, solver operating-point comparisons and physics simulation buttons. The full pipeline executes in under 100 ms. The platform is deployed on Google Cloud Platform with PostgreSQL for user management and an Nginx reverse proxy.

### Statistics and reproducibility

All TNA scores are deterministic functions of system descriptors and task queries; no random sampling is involved in the evaluation protocol. Pareto ranking is a deterministic set operation. PSNR and SSIM values in the algorithm catalog are computed on standardized benchmark datasets using standard definitions (PSNR = 10 log₁₀(MAX² / MSE); SSIM per Wang et al.^33^). Sensitivity analyses use random perturbations with fixed seeds for reproducibility. Concordance with ACR criteria uses one-sided exact binomial test against chance concordance (null hypothesis: random selection from 168 systems). The observed 11/12 concordance yields *P* < 10⁻¹⁵ against the null. All pilot study results are fully reproducible by submitting the specified task queries to the platform.

---

## Data Availability

The system descriptor catalog (`system_catalog.json`, 168 entries) and all benchmark data are available at https://pwm.platformai.org/benchmark/system-design under a Creative Commons Attribution 4.0 International licence. The interactive recommendation tool is available at https://pwm.platformai.org/speclab. The ACR concordance dataset (12 scenarios with task queries and results) is provided as Supplementary Table 1.

## Code Availability

Source code for PWM-SyS is available in the Physics World Model repository at https://github.com/Shi-Labs/Physics-World-Model. Key components: catalog generation (`scripts/generate_system_catalog.py`), recommendation engine (`services/system_recommender.py`), SpecLab integration (`routers/spec_chat.py`), sensitivity analysis scripts (`scripts/sensitivity_analysis.py`) and full technical specification (`PROPOSAL.md`). The code is released under the MIT licence.

## Acknowledgements

[To be completed before submission.]

## Author Contributions

C.S. conceived the project, designed the TNA framework and evaluation protocol, curated the system descriptor catalog, performed the validation analyses, implemented the platform and wrote the manuscript.

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
17. American College of Radiology. ACR Appropriateness Criteria. https://acsearch.acr.org/list (2023).
18. Marsh, K. et al. Multiple criteria decision analysis for health care decision making — emerging good practices: report 2 of the ISPOR MCDA Emerging Good Practices Task Force. *Value Health* **19**, 125–137 (2016).
19. Thokala, P. et al. Multiple criteria decision analysis for health care decision making — an introduction: report 1 of the ISPOR MCDA Emerging Good Practices Task Force. *Value Health* **19**, 1–13 (2016).
20. Goetghebeur, M. M. et al. Advancing the EVIDEM framework: facilitating development and review of health technology assessments. *Int. J. Technol. Assess. Health Care* **28**, 63–70 (2012).
21. Hansen, P. & Ombler, F. A new method for scoring additive multi-attribute value models using pairwise rankings of alternatives. *J. Multi-Criteria Decis. Anal.* **15**, 87–107 (2008).
22. Antun, V. et al. On instabilities of deep learning in image reconstruction and the potential costs of AI. *Proc. Natl Acad. Sci. USA* **117**, 30088–30098 (2020).
23. Moen, E. et al. Deep learning for cellular image analysis. *Nat. Methods* **16**, 1233–1246 (2019).
24. Prevedel, R. et al. Simultaneous whole-animal 3D imaging of neuronal activity using light-field microscopy. *Nat. Methods* **11**, 727–730 (2014).
25. Shi, C. Eleven primitives and three gates: the universal structure of computational imaging. Preprint (2026).
26. Gao, L. et al. Single-shot compressed ultrafast photography at one hundred billion frames per second. *Nature* **516**, 74–77 (2014).
27. Wang, Z. et al. EfficientSCI: densely connected network with space-time factorization for large-scale video snapshot compressive imaging. *Proc. CVPR*, 18477–18486 (2023).
28. Meng, Z. et al. DiffusionSCI: generative diffusion model for snapshot compressive imaging. Preprint at https://arxiv.org/abs/2311.11725 (2023).
29. Hand, D. J. Classifier technology and the illusion of progress. *Stat. Sci.* **21**, 1–14 (2006).
30. Deb, K. et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans. Evol. Comput.* **6**, 182–197 (2002).
31. Raji, I. D. et al. AI and the everything in the whole wide world benchmark. In *Proc. NeurIPS Datasets and Benchmarks Track* (2021).
32. Wang, A. et al. GLUE: a multi-task benchmark and analysis platform for natural language understanding. Preprint at https://arxiv.org/abs/1804.07461 (2018).
33. Wang, Z. et al. Image quality assessment: from error visibility to structural similarity. *IEEE Trans. Image Process.* **13**, 600–612 (2004).
34. Lustig, M., Donoho, D. & Pauly, J. M. Sparse MRI: the application of compressed sensing for rapid MR imaging. *Magn. Reson. Med.* **58**, 1182–1195 (2007).
35. Candès, E. J. & Wakin, M. B. An introduction to compressive sampling. *IEEE Signal Process. Mag.* **25**, 21–30 (2008).
36. Maldague, X. P. V. *Theory and Practice of Infrared Technology for Nondestructive Testing* (Wiley, 2001).
37. Mait, J. N., Euliss, G. W. & Athale, R. A. Computational imaging. *Adv. Opt. Photon.* **10**, 409–483 (2018).
38. McCann, M. T., Jin, K. H. & Unser, M. Convolutional neural networks for inverse problems in imaging: a review. *IEEE Signal Process. Mag.* **34**, 85–95 (2017).
39. Mangul, S. et al. Systematic benchmarking of omics computational tools. *Nat. Commun.* **10**, 1393 (2019).
40. Weber, L. M. et al. Essential guidelines for computational method benchmarking. *Genome Biol.* **20**, 125 (2019).
41. Ulyanov, D., Vedaldi, A. & Lempitsky, V. Deep image prior. *Int. J. Comput. Vis.* **128**, 1867–1888 (2020).
42. Venkatakrishnan, S. V., Bouman, C. A. & Wohlberg, B. Plug-and-play priors for model based reconstruction. In *Proc. IEEE GlobalSIP*, 945–948 (2013).
43. Tian, C. et al. Deep learning on image denoising: an overview. *Neural Netw.* **131**, 251–275 (2020).
44. Bostan, E. et al. Deep phase decoder: self-calibrating phase microscopy with an untrained deep neural network. *Optica* **7**, 559–562 (2020).
45. Agustsson, E. & Timofte, R. NTIRE 2017 challenge on single image super-resolution: dataset and study. In *Proc. CVPR Workshops* (2017).
46. Power, R. M. & Huisken, J. A guide to light-sheet fluorescence microscopy for multiscale imaging. *Nat. Methods* **14**, 360–373 (2017).
47. Hoebe, R. A. et al. Controlled light-exposure microscopy reduces photobleaching and phototoxicity in fluorescence live-cell imaging. *Nat. Biotechnol.* **25**, 249–253 (2007).
48. Chen, B.-C. et al. Lattice light-sheet microscopy: imaging molecules to embryos at high spatiotemporal resolution. *Science* **346**, 1257998 (2014).
49. Schermelleh, L. et al. Super-resolution microscopy demystified. *Nat. Cell Biol.* **21**, 72–84 (2019).
50. Weigert, M. et al. Content-aware image restoration: pushing the limits of fluorescence microscopy. *Nat. Methods* **15**, 1090–1097 (2018).

---

## Figure Legends

**Figure 1 | PWM-SyS overview.** **a**, Conventional imaging benchmarks fix the system and rank algorithms (left). PWM-SyS evaluates both system and solver jointly against task requirements (right). **b**, Three-layer architecture: Layer A (system descriptor catalog) provides neutral facts; Layer B (evaluation protocol) applies TNA scoring and Pareto ranking; Layer C (interactive platform) enables natural-language querying and simulation. **c**, Workflow: the user specifies a task query with requirements and constraints, PWM-SyS filters infeasible systems, scores feasible ones across eight TNA dimensions, identifies the Pareto frontier, and returns ranked recommendations.

**Figure 2 | System descriptor catalog.** **a**, Each of 168 system descriptors contains five property groups: physical chain, acquisition, solver, operations and sample compatibility. Example shown for two-photon microscopy. **b**, Distribution of 168 systems across 5 carrier families and 19 application categories. Bar chart shows category counts; pie chart shows carrier distribution. **c**, Scatter plot of capital cost versus best PSNR for all 168 systems, coloured by carrier type.

**Figure 3 | Task-Normalized Adequacy scoring.** **a**, TNA maps system capability to a [0, 10] scale relative to three task-defined reference points (*r*_min, *r*_target, *r*_comfort), shown for a "higher is better" dimension (frame rate). Systems below *r*_min score 0 (infeasible); those exceeding *r*_comfort score 10 (adequate with margin). **b**, The eight TNA dimensions, grouped by function: physical feasibility (D1–D4), reconstruction quality (D5), and practical constraints (D6–D8). **c**, Radar charts comparing TNA profiles of three representative microscopy systems (two-photon, light-sheet, confocal) evaluated against a calcium imaging task.

**Figure 4 | Three-stage evaluation protocol.** **a**, Stage 1 (feasibility gate): 168 systems filtered by hard constraints; example shows biological microscopy query reducing candidates from 168 to 14. **b**, Stage 2 (Pareto ranking): feasible systems ranked by Pareto dominance in 8-dimensional TNA space; example shows 5 Pareto-optimal systems (red) among 14 feasible (grey). **c**, Stage 3 (preference weighting): within the Pareto set, application-specific weights produce a scalar score for final ranking.

**Figure 5 | Validation and pilot studies.** **a**, Concordance with ACR Appropriateness Criteria: 11/12 clinical scenarios show the ACR-endorsed modality in the PWM-SyS Pareto set (green); one discordant case involves radiation dose sensitivity (red). **b**, Sensitivity analysis: box plots of Jaccard similarity (Pareto set stability) and Kendall's τ (rank stability) under ±20% TNA parameter perturbation across three pilot studies (540 trials). **c**, Biological microscopy: 14/168 feasible, 5 Pareto-optimal. Heatmap shows TNA scores for Pareto-optimal systems; two-photon ranks first. **d**, High-speed research: 3/168 candidates, only CACTI passes. Solver comparison shows EfficientSCI as recommended (DiffusionSCI fails latency). **e**, Clinical screening: 41/168 feasible, 8 Pareto-optimal. B-mode ultrasound ranks first.

**Figure 6 | Interactive platform.** **a**, SpecLab natural-language interface: user enters a query and receives ranked system recommendations with adequacy profiles. **b**, System design catalog page showing sortable tables organized by category. **c**, Per-modality benchmark page linking system descriptor to reconstruction leaderboard. Screenshots from pwm.platformai.org.

---

## Extended Data

### Extended Data Table 1 | Full system descriptor schema

(19-field schema as previously defined; see paper.tex for complete table.)

### Extended Data Table 2 | Comparison with existing evaluation frameworks

| Framework | Scope | Cross-modality | Task-conditioned | System+solver | Constraint filtering | Interactive |
|-----------|-------|----------------|------------------|---------------|---------------------|-------------|
| fastMRI^10^ | MRI | No | No | No | No | No |
| AAPM CT^11^ | CT | No | No | No | No | No |
| SMLM challenge^12^ | SMLM | No | No | No | No | No |
| ACR Criteria^17^ | Medical | Yes | Yes | No | No | No |
| EVIDEM^20^ | Health tech | Yes | Yes | No | No | Partial |
| PAPRIKA^21^ | General MCDA | Yes | Yes | No | No | Yes |
| **PWM-SyS** | **168 modalities** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### Extended Data Fig. 1 | Pareto set sizes across task profiles

Distribution of Pareto-optimal set sizes for 50 randomly generated task profiles with varying constraint stringency. Median Pareto set size is 7 systems (interquartile range: 4–12).

### Extended Data Fig. 2 | Catalog verification

Agreement between catalog values and independent sources for 30 systems × 5 properties = 150 data points. Scatter plot of log₂(catalog/reference) shows 95.3% within ±1.0 (factor of 2) and 84.7% within ±0.26 (20%).

### Extended Data Table 3 | Category summary statistics

(19-category table with count, average PSNR, average cost, single-shot percentage.)

### Extended Data Table 4 | Carrier distribution

(9-carrier table with counts and representative systems.)

### Extended Data Table 5 | Source traceability for catalog fields

| Field group | Primary source type | Example sources | Coverage |
|------------|-------------------|-----------------|----------|
| Physical chain | Published system papers | Ref. 3, 4, 7, 8, 9, 26 | 168/168 |
| Acquisition parameters | Manufacturer datasheets | Siemens, Hamamatsu, Zeiss, Bruker | 152/168 |
| Solver performance | PWM algorithm catalog | Internal benchmark (1,367 algorithms) | 168/168 |
| Cost and operations | Institutional procurement, datasheets | Core facility databases, vendor quotes | 141/168 |
| Sample compatibility | Published protocols, safety data | Refs. 47, 48 | 168/168 |

### Extended Data Table 6 | ACR concordance details

| Scenario | ACR recommendation | Rating | PWM-SyS Pareto? | PWM-SyS rank |
|----------|-------------------|--------|-----------------|-------------|
| Suspected PE | CT angiography | 9 | Yes | 1 |
| Acute chest pain | CT/Echo | 8/8 | Yes/Yes | 1/3 |
| Breast screening | Mammography | 9 | Yes | 1 |
| Acute appendicitis | CT | 9 | Yes | 1 |
| Traumatic brain injury | CT | 9 | Yes | 1 |
| Low back pain | MRI | 8 | Yes | 2 |
| Suspected renal colic | CT | 9 | Yes | 1 |
| Carotid disease | Duplex US/CTA | 9/8 | Yes/Yes | 1/2 |
| Liver lesion | ce-MRI | 9 | Yes | 1 |
| Pediatric seizure | MRI | 9 | Yes | 2 |
| MSK trauma | Radiography | 9 | Yes | 1 |
| PE in pregnancy | V/Q scan | 8 | **No** | — |

---

## Reporting Summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article.
