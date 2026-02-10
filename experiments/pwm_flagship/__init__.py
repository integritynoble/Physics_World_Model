"""PWM Flagship experiments (Paper 1).

Full-pipeline demonstrations across depth (SPC, CACTI, CASSI) and breadth
(CT, Widefield, Holography) modalities, plus universality (26/26 templates)
and ablation studies.

Modules
-------
spc_loop        Full PWM pipeline on SPC (design -> preflight -> calibration -> reconstruction)
cacti_loop      Full PWM pipeline on CACTI
cassi_loop      Full PWM pipeline on CASSI (references PWMI-CASSI results)
breadth_ct      CT breadth anchor (compile + adjoint + mismatch/cal)
breadth_wf      Widefield breadth anchor
breadth_holo    Holography breadth anchor
universality    26/26 template compilation + validation
ablations       4 ablations x 3 modalities degradation study
"""

__version__ = "0.1.0"
