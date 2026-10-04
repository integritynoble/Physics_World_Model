"""Source elements for the 54 registry modalities that have none (LIB-40 IMG-02).

HONEST VALIDATION LEVEL: every entry here is `drafted`. The radiation or particle that the instrument really uses
and its typical operating values were written from general instrument knowledge and the source primitives of the
Finite Primitive Basis / System Design papers. None of the numbers has been checked against a cited paper, a
datasheet or a measurement, so none is a validated value; `verified_against` is null for all of them. Treat them
as starting points to be replaced by measured values (IMG-08), not as facts.

Entry fields
  carrier       what actually leaves the source (may differ from the catalog's carrier; see `carrier_note`)
  primitive     one of the registered source primitives (`photon_source`, `xray_source`, `acoustic_source`,
                `spin_source`, `electron_beam_source`, `generic_source`); its only parameter is `strength`
                (`beam_current_na` for the electron beam), so the values below are catalog metadata and are NOT
                passed to the compiled operator
  parameters    numeric only, the unit is the key suffix (checked by tests)
  constraints   {parameter, min, max, reason}; bounds that a design must stay inside, `null` means unbounded
  descriptors   non-numeric facts (type, mode, pattern)
  controllable  False when the source is the object or the environment (sun, star, ambient light)
"""
from __future__ import annotations

from typing import Any, Dict, List

VALIDATION = "drafted"
SOURCE_PRIMITIVES = ("photon_source", "xray_source", "acoustic_source", "spin_source", "electron_beam_source",
                     "generic_source")

# Unit suffixes accepted on parameter keys (tests refuse a numeric parameter without one).
UNIT_SUFFIXES = ("_nm", "_um", "_mm", "_m", "_cm1", "_hz", "_khz", "_mhz", "_ghz", "_s", "_ms", "_us", "_ns", "_ps",
                 "_fs", "_mw", "_w", "_mj", "_uj", "_pj", "_ev", "_kev", "_gev", "_kv", "_v", "_ma", "_na", "_pa",
                 "_ut", "_mt", "_t", "_k", "_mk", "_deg", "_mrad", "_pa_s", "_mpa", "_ul_min", "_lux", "_db",
                 "_jy", "_t_m", "_w_m2", "_kw_cm2", "_mw_mm2", "_n_cm2_s", "_per_pulse", "_per_s", "_fraction",
                 "_ratio", "_mag", "_per_mode", "_mgy", "_days", "_count", "_ppm", "_kvp")


def _el(name: str, carrier: str, primitive: str, parameters: Dict[str, float], basis: str, *,
        constraints: List[Dict[str, Any]] = (), descriptors: Dict[str, str] = None, controllable: bool = True,
        carrier_note: str = None) -> Dict[str, Any]:
    assert primitive in SOURCE_PRIMITIVES, primitive
    return {"name": name, "transfer": "identity", "throughput": 1.0, "parameters": dict(parameters), "noise": [],
            "carrier": carrier, "primitive": primitive, "constraints": list(constraints),
            "descriptors": dict(descriptors or {}), "controllable": controllable,
            "validation": VALIDATION, "verified_against": None, "basis": basis,
            **({"carrier_note": carrier_note} if carrier_note else {})}


def _c(parameter: str, lo, hi, reason: str) -> Dict[str, Any]:
    return {"parameter": parameter, "min": lo, "max": hi, "reason": reason}


_MAGNET = ("Main magnet", "spin", "spin_source")
_SUN = "Sun (top-of-atmosphere solar irradiance)"
_BASIS_SUN = "Solar constant ~1361 W/m2 at top of atmosphere; the sun is an environmental, uncontrollable source."

SOURCES: Dict[str, List[Dict[str, Any]]] = {
    "afm": [_el("Cantilever drive (tapping mode)", "force (tip-sample), mechanical", "generic_source",
                {"drive_frequency_khz": 300, "drive_amplitude_nm": 20, "tip_radius_nm": 10, "lift_height_nm": 0},
                "Tapping-mode silicon cantilever near its resonance; the image contrast comes from the tip-sample force, "
                "light only reads the deflection.",
                constraints=[_c("drive_frequency_khz", 10, 2000, "cantilever resonance range, typical"),
                             _c("tip_radius_nm", 1, 100, "limits lateral resolution")],
                carrier_note="catalog says photon; the interaction is the tip-sample force, the optical lever is a detector")],
    "asl_mri": [_el("Main magnet (3 T)", "spin", "spin_source",
                    {"field_strength_T": 3.0, "larmor_frequency_mhz": 127.7},
                    "Clinical 3 T magnet, proton Larmor frequency 42.58 MHz/T.", constraints=[
                        _c("field_strength_T", 1.5, 7.0, "clinical to ultra-high field")]),
                  _el("Arterial labeling RF", "spin", "spin_source",
                      {"label_duration_ms": 1800, "post_label_delay_ms": 1800, "labeling_efficiency_fraction": 0.85},
                      "Pseudo-continuous labeling, typical consensus timings for adults.",
                      constraints=[_c("labeling_efficiency_fraction", 0.0, 1.0, "fraction of inverted spins"),
                                   _c("specific_absorption_rate_w_kg", None, 3.2, "head SAR limit, normal mode")])],
    "atom_probe": [_el("Pulsed UV laser plus standing voltage", "ions (field evaporated)", "generic_source",
                       {"wavelength_nm": 355, "pulse_duration_ps": 12, "pulse_energy_pj": 50,
                        "repetition_rate_khz": 250, "standing_voltage_kv": 5, "specimen_temperature_k": 50},
                       "Laser-assisted local-electrode atom probe; each pulse triggers field evaporation of one or more ions.",
                       constraints=[_c("specimen_temperature_k", 20, 100, "cryogenic specimen stage"),
                                    _c("pulse_energy_pj", 1, 200, "too high heats the tip")],
                       carrier_note="catalog says acoustic; the evaporated species are ions")],
    "brillouin": [_el("Single-mode CW laser", "photon", "photon_source",
                      {"wavelength_nm": 532, "power_mw": 20, "linewidth_mhz": 1},
                      "Single-longitudinal-mode 532 nm laser; the Brillouin shift of water is about 5 GHz, so the "
                      "linewidth must be far below that.",
                      constraints=[_c("linewidth_mhz", None, 100, "must resolve a ~5 GHz shift with MHz precision"),
                                   _c("power_mw", None, 60, "photodamage and heating in live samples, typical")])],
    "cars": [_el("Pump and Stokes picosecond lasers", "photon (nonlinear)", "photon_source",
                 {"pump_wavelength_nm": 817, "stokes_wavelength_nm": 1064, "pulse_duration_ps": 6,
                  "repetition_rate_mhz": 80, "average_power_mw": 10},
                 "Two synchronised picosecond beams; the difference in wavenumber 1e7/817 - 1e7/1064 = 2841 cm-1 "
                 "addresses the CH2 stretch.",
                 constraints=[_c("raman_shift_cm1", 500, 3500, "vibrational window, set by the two wavelengths"),
                              _c("average_power_mw", None, 50, "sample damage, typical")])],
    "cathodoluminescence": [_el("Focused electron beam", "electron", "electron_beam_source",
                                {"beam_energy_kev": 10, "beam_current_na": 1.0, "spot_size_nm": 10},
                                "SEM/STEM electron gun; the beam excites luminescence, the photons are the signal.",
                                constraints=[_c("beam_energy_kev", 1, 300, "SEM to STEM range"),
                                             _c("beam_current_na", 0.001, 100, "beam damage and carrier saturation")],
                                carrier_note="catalog says electron to photon: the source is the electron, the photon is the signal")],
    "cest_mri": [_el("Main magnet (3 T)", "spin", "spin_source", {"field_strength_T": 3.0},
                     "Clinical 3 T magnet."),
                 _el("Saturation RF train", "spin", "spin_source",
                     {"saturation_b1_ut": 1.0, "saturation_duration_ms": 2000, "offset_min_ppm": -6.0,
                      "offset_max_ppm": 6.0},
                     "Continuous or pulsed-train saturation at a set of offsets from water.",
                     constraints=[_c("saturation_b1_ut", 0.2, 6.0, "typical range, limited by SAR"),
                                  _c("specific_absorption_rate_w_kg", None, 3.2, "head SAR limit, normal mode")])],
    "coronagraphy": [_el("Target star and its planet", "photon", "photon_source",
                         {"wavelength_nm": 600, "bandwidth_fraction": 0.1, "star_planet_contrast_ratio": 1e-10,
                          "stellar_magnitude_mag": 5},
                         "The star is the dominant source; an Earth-like planet in reflected light is about 1e-10 of the star.",
                         constraints=[_c("star_planet_contrast_ratio", 1e-12, 1e-4, "planet / star flux")],
                         controllable=False)],
    "cup": [_el("Ultrafast pulsed laser", "photon", "photon_source",
                {"wavelength_nm": 532, "pulse_duration_ps": 7},
                "Picosecond pulse illuminating a transient scene; CUP images the scene with a streak camera.",
                constraints=[_c("pulse_duration_ps", 0.01, 100, "shorter than the streak sweep window")])],
    "desi": [_el("Charged solvent spray", "charged droplets", "generic_source",
                 {"spray_voltage_kv": 4.5, "solvent_flow_ul_min": 2.5, "nebulizing_gas_pressure_mpa": 0.7,
                  "spot_size_um": 200},
                 "Electrospray of solvent onto the surface; secondary droplets carry the analyte into the mass spectrometer.",
                 constraints=[_c("spot_size_um", 50, 500, "sets the spatial resolution")],
                 carrier_note="catalog says infrared photon; the probe is charged solvent droplets")],
    "dna_paint": [_el("Excitation laser (TIRF)", "photon", "photon_source",
                      {"wavelength_nm": 561, "intensity_kw_cm2": 0.5, "imager_concentration_nm": 1.0,
                       "frame_time_ms": 100},
                      "Total-internal-reflection excitation of transiently binding dye-labeled imager strands.",
                      constraints=[_c("imager_concentration_nm", 0.05, 10, "too high gives background, too low gives slow acquisition")])],
    "eddy_current": [_el("AC excitation coil", "electromagnetic (induction)", "generic_source",
                         {"frequency_khz": 100, "drive_current_ma": 20},
                         "Sinusoidal coil current induces eddy currents in a conductor; the skin depth sets which depth is probed.",
                         constraints=[_c("frequency_khz", 0.1, 5000, "skin depth to sensor bandwidth")],
                         carrier_note="catalog says xray; the carrier is a low-frequency magnetic field")],
    "edx_mapping": [_el("Focused STEM electron probe", "electron", "electron_beam_source",
                        {"beam_energy_kev": 200, "beam_current_na": 0.5, "probe_size_nm": 0.2, "dwell_time_ms": 1},
                        "Scanning probe generating characteristic X-rays at each pixel; the X-rays are the signal.",
                        constraints=[_c("beam_energy_kev", 30, 300, "STEM range"),
                                     _c("beam_current_na", 0.01, 5, "counts vs beam damage")],
                        carrier_note="catalog says xray: the source is the electron beam, the X-ray is the signal")],
    "eht_imaging": [_el("Black-hole accretion flow", "photon (mm-wave radio)", "photon_source",
                        {"frequency_ghz": 230, "bandwidth_ghz": 4, "wavelength_mm": 1.3},
                        "The astrophysical object emits at 1.3 mm; it cannot be changed.",
                        constraints=[_c("frequency_ghz", 86, 345, "VLBI observing bands")], controllable=False)],
    "entangled_photon": [_el("Pump laser and nonlinear crystal (SPDC)", "photon (entangled pair)", "photon_source",
                             {"pump_wavelength_nm": 405, "pump_power_mw": 50, "pair_wavelength_nm": 810},
                             "Type-I or type-II spontaneous parametric down-conversion; energy conservation gives "
                             "2 x 405 nm pump photon energy per 810 nm pair photon.",
                             constraints=[_c("pump_power_mw", None, 200, "multi-pair emission raises accidentals")])],
    "event_camera": [_el("Ambient scene illumination", "photon", "photon_source",
                         {"illuminance_lux": 100, "scene_dynamic_range_db": 120},
                         "No controlled source: events come from brightness change under ambient light.",
                         constraints=[_c("illuminance_lux", 0.1, 100000, "scene range of an event sensor, typical")],
                         controllable=False)],
    "ftir_imaging": [_el("Globar thermal infrared source", "infrared photon", "photon_source",
                         {"source_temperature_k": 1200, "wavenumber_min_cm1": 400, "wavenumber_max_cm1": 4000,
                          "spectral_resolution_cm1": 4},
                         "Silicon carbide globar, broadband mid-infrared; modulated by the interferometer.",
                         constraints=[_c("spectral_resolution_cm1", 0.1, 16, "mirror travel")])],
    "ghost_imaging": [_el("Entangled or pseudothermal light", "photon (entangled pair)", "photon_source",
                          {"pump_wavelength_nm": 405, "pump_power_mw": 50, "pair_wavelength_nm": 810},
                          "SPDC pairs, as in the first ghost imaging experiment.",
                          descriptors={"alternatives": "pseudothermal laser through rotating ground glass; "
                                       "computational: laser plus spatial light modulator"})],
    "hdr_imaging": [_el("Ambient scene illumination", "photon", "photon_source",
                        {"illuminance_lux": 1000, "scene_dynamic_range_db": 120},
                        "A natural scene spans more than 100 dB; no controlled source.",
                        constraints=[_c("scene_dynamic_range_db", 60, 160, "typical scenes")], controllable=False)],
    "hyperspectral_remote": [_el(_SUN, "photon", "photon_source",
                                 {"solar_constant_w_m2": 1361, "solar_zenith_angle_deg": 30,
                                  "wavelength_min_nm": 400, "wavelength_max_nm": 2500},
                                 _BASIS_SUN, constraints=[_c("solar_zenith_angle_deg", 0, 75, "typical acquisition limit")],
                                 controllable=False)],
    "impedance_tomo": [_el("Injected alternating current", "electrical current", "generic_source",
                           {"frequency_khz": 50, "current_ma": 1.0, "n_electrodes_count": 16},
                           "Current injected through surface electrodes, voltages measured at the others.",
                           constraints=[_c("current_ma", None, 5, "patient safety; the limit rises with frequency, to be verified against IEC 60601-1")],
                           carrier_note="catalog says photon; the carrier is electrical current")],
    "insar": [_el("C-band SAR transmitter", "microwave", "generic_source",
                  {"center_frequency_ghz": 5.405, "bandwidth_mhz": 56.5, "prf_hz": 1700, "incidence_angle_deg": 39,
                   "revisit_days": 12},
                  "Spaceborne C-band radar; two passes at a fixed repeat interval give the interferometric phase.",
                  constraints=[_c("incidence_angle_deg", 20, 50, "typical SAR incidence")])],
    "libs": [_el("Pulsed Nd:YAG laser", "photon", "photon_source",
                 {"wavelength_nm": 1064, "pulse_energy_mj": 20, "pulse_duration_ns": 8, "repetition_rate_hz": 10,
                  "spot_diameter_um": 100},
                 "Nanosecond pulse ablates the surface to a plasma whose emission is the signal.",
                 constraints=[_c("pulse_energy_mj", 1, 200, "plasma ignition threshold to crater size")])],
    "magnetic_particle": [_el("Drive field and selection gradient", "magnetic field", "generic_source",
                              {"drive_frequency_khz": 25, "drive_amplitude_mt": 20, "selection_gradient_t_m": 2.0},
                              "Oscillating drive field moves the field-free point through the tracer; the tracer's "
                              "nonlinear magnetisation is the signal.",
                              constraints=[_c("drive_amplitude_mt", None, 30, "peripheral nerve stimulation and heating limit, to be verified")],
                              carrier_note="catalog says xray; the carrier is a magnetic field")],
    "maldi_msi": [_el("Pulsed UV laser", "photon", "photon_source",
                      {"wavelength_nm": 355, "pulse_duration_ns": 3, "repetition_rate_khz": 1, "spot_diameter_um": 20},
                      "Matrix-assisted laser desorption/ionization; the ions are the signal.",
                      constraints=[_c("spot_diameter_um", 5, 200, "spatial resolution")],
                      carrier_note="catalog says xray; the probe is a UV laser (or 337 nm nitrogen laser)")],
    "mfm": [_el("Cantilever with magnetized tip", "magnetic stray field (force)", "generic_source",
                {"drive_frequency_khz": 75, "drive_amplitude_nm": 20, "lift_height_nm": 50},
                "Two-pass scan: topography first, then the tip is lifted and senses the magnetic force gradient.",
                constraints=[_c("lift_height_nm", 10, 200, "to separate magnetic from van der Waals force")],
                carrier_note="catalog says photon; the interaction is magnetic force")],
    "mr_elastography": [_el("Main magnet (3 T)", "spin", "spin_source", {"field_strength_T": 3.0},
                            "Clinical 3 T magnet."),
                        _el("Mechanical vibration driver", "acoustic (shear wave)", "acoustic_source",
                            {"vibration_frequency_hz": 60, "displacement_amplitude_um": 50},
                            "External actuator generating shear waves; liver MRE uses 60 Hz, brain about 50 Hz.",
                            constraints=[_c("vibration_frequency_hz", 20, 200, "tissue wavelength vs resolution")])],
    "mr_fingerprinting": [_el("Main magnet (3 T)", "spin", "spin_source", {"field_strength_T": 3.0}, "Clinical 3 T magnet."),
                          _el("Pseudorandom excitation", "spin", "spin_source",
                              {"flip_angle_max_deg": 70, "repetition_time_ms": 12, "n_timepoints_count": 1000},
                              "Varying flip angle and TR train; each tissue yields a distinct signal evolution.",
                              constraints=[_c("flip_angle_max_deg", 1, 90, "SAR"),
                                           _c("specific_absorption_rate_w_kg", None, 3.2, "head SAR limit, normal mode")])],
    "mra": [_el("Main magnet (3 T)", "spin", "spin_source", {"field_strength_T": 3.0}, "Clinical 3 T magnet."),
            _el("Time-of-flight excitation", "spin", "spin_source",
                {"flip_angle_deg": 20, "repetition_time_ms": 25, "echo_time_ms": 3.5},
                "Gradient-echo with short TR so that inflowing blood is bright.",
                constraints=[_c("repetition_time_ms", 15, 40, "saturation of static tissue")])],
    "multispectral_sat": [_el(_SUN, "photon", "photon_source",
                              {"solar_constant_w_m2": 1361, "solar_zenith_angle_deg": 30,
                               "wavelength_min_nm": 400, "wavelength_max_nm": 2300}, _BASIS_SUN,
                              constraints=[_c("solar_zenith_angle_deg", 0, 75, "typical acquisition limit")],
                              controllable=False)],
    "neutron_diffraction": [_el("Thermal neutron beam", "neutron", "generic_source",
                                {"wavelength_nm": 0.15, "wavelength_spread_fraction": 0.01,
                                 "flux_n_cm2_s": 1e7},
                                "Reactor or spallation beam line; 1.5 Angstrom neutrons have thermal energy.",
                                constraints=[_c("wavelength_nm", 0.05, 1.0, "crystal d-spacing range")],
                                carrier_note="catalog says photon; the probe is neutrons")],
    "nirs_brain": [_el("Near-infrared LEDs or laser diodes", "photon", "photon_source",
                       {"wavelength_1_nm": 760, "wavelength_2_nm": 850, "power_mw": 5, "irradiance_mw_mm2": 1.0},
                       "Two wavelengths either side of the haemoglobin isosbestic point (about 800 nm) allow oxy- and "
                       "deoxy-haemoglobin to be separated.",
                       constraints=[_c("irradiance_mw_mm2", None, 4.0,
                                       "skin exposure limit at 850 nm, from ANSI Z136.1 (0.2 x 10^(0.002(lambda-700)) W/cm2); to be verified")],
                       carrier_note="catalog says electron; the source is near-infrared light")],
    "ocean_color": [_el(_SUN, "photon", "photon_source",
                        {"solar_constant_w_m2": 1361, "solar_zenith_angle_deg": 30,
                         "wavelength_min_nm": 400, "wavelength_max_nm": 900}, _BASIS_SUN,
                        constraints=[_c("solar_zenith_angle_deg", 0, 70, "glint and path length, typical")],
                        controllable=False)],
    "particle_calorimetry": [_el("Incident particle beam", "particle (electron or hadron)", "generic_source",
                                 {"particle_energy_gev": 50, "beam_spot_mm": 10},
                                 "Test-beam electrons or hadrons depositing their energy in the calorimeter.",
                                 constraints=[_c("particle_energy_gev", 0.1, 1000, "test beam to collider")])],
    "passive_microwave": [_el("Thermal emission of the scene", "microwave (thermal)", "generic_source",
                              {"brightness_temperature_k": 250, "frequency_ghz": 36.5},
                              "Passive: the scene itself emits, no transmitter. Typical channels 6.9 to 89 GHz.",
                              constraints=[_c("frequency_ghz", 1, 200, "radiometer channels")], controllable=False)],
    "polsar": [_el("L-band polarimetric SAR transmitter", "microwave", "generic_source",
                   {"center_frequency_ghz": 1.27, "bandwidth_mhz": 84, "incidence_angle_deg": 35},
                   "Alternating H and V transmit pulses with H and V receive give the four channels.",
                   descriptors={"polarization": "quad (HH, HV, VH, VV)"},
                   constraints=[_c("incidence_angle_deg", 20, 50, "typical SAR incidence")])],
    "pump_probe": [_el("Femtosecond pump and probe pulses", "photon (ultrafast)", "photon_source",
                       {"pump_wavelength_nm": 720, "probe_wavelength_nm": 810, "pulse_duration_fs": 150,
                        "repetition_rate_mhz": 80, "average_power_mw": 5, "delay_range_ps": 100},
                       "Titanium-sapphire laser with an optical parametric oscillator; the delay line sets the "
                       "time axis.",
                       constraints=[_c("average_power_mw", None, 20, "sample damage, typical")])],
    "quantum_illumination": [_el("Pump and nonlinear crystal (SPDC)", "photon (entangled pair)", "photon_source",
                                 {"pump_wavelength_nm": 405, "pair_wavelength_nm": 810,
                                  "mean_photons_per_mode": 0.01},
                                 "Signal-idler pairs; the advantage over classical light exists at low brightness.",
                                 constraints=[_c("mean_photons_per_mode", None, 0.1,
                                                 "quantum advantage requires a source much weaker than one photon per mode")])],
    "radio_astronomy": [_el("Astronomical radio source", "radio", "generic_source",
                            {"observing_frequency_ghz": 1.4, "bandwidth_mhz": 100, "source_flux_density_jy": 1.0},
                            "Natural emission (for example the 21 cm line); the array cannot change it.",
                            constraints=[_c("observing_frequency_ghz", 0.05, 100, "aperture synthesis range")],
                            controllable=False)],
    "radio_interferometry": [_el("Astronomical radio source (VLBI)", "radio", "generic_source",
                                 {"observing_frequency_ghz": 43, "bandwidth_mhz": 256, "source_flux_density_jy": 1.0},
                                 "Compact source observed by widely separated stations; the source is not controllable.",
                                 constraints=[_c("observing_frequency_ghz", 0.3, 345, "VLBI bands")],
                                 controllable=False)],
    "raman_imaging": [_el("CW excitation laser", "photon", "photon_source",
                          {"wavelength_nm": 785, "power_mw": 10, "spot_diameter_um": 1.0},
                          "785 nm limits fluorescence background; 532 nm gives a stronger Raman cross-section.",
                          constraints=[_c("power_mw", None, 50, "sample heating, material dependent")])],
    "shearography": [_el("Long-coherence laser", "coherent photon", "photon_source",
                         {"wavelength_nm": 532, "power_mw": 100, "coherence_length_m": 10},
                         "Diode-pumped solid-state laser; the coherence length must exceed the path difference of the shearing optics.",
                         constraints=[_c("coherence_length_m", 0.1, None, "interference of the two sheared images")])],
    "shg": [_el("Femtosecond Ti:sapphire laser", "photon (nonlinear)", "photon_source",
                {"wavelength_nm": 890, "pulse_duration_fs": 140, "repetition_rate_mhz": 80, "average_power_mw": 20},
                "Second harmonic is produced at 445 nm in non-centrosymmetric structures such as collagen.",
                constraints=[_c("average_power_mw", None, 50, "tissue damage, typical")])],
    "sims": [_el("Primary ion beam", "ion", "generic_source",
                 {"primary_ion_energy_kev": 16, "primary_current_pa": 1, "spot_size_nm": 100},
                 "Cs+ or O2- beam sputtering the surface; secondary ions are the signal.",
                 constraints=[_c("spot_size_nm", 50, 5000, "resolution vs signal")],
                 carrier_note="catalog says xray; the probe is an ion beam")],
    "solar_imaging": [_el("Solar corona", "photon (EUV)", "photon_source",
                          {"coronal_temperature_mk": 1.0, "line_wavelength_nm": 17.1},
                          "Hot plasma emission lines, for example Fe IX at 17.1 nm; not controllable.",
                          constraints=[_c("line_wavelength_nm", 5, 40, "EUV imaging bands")], controllable=False)],
    "srs": [_el("Pump and Stokes picosecond lasers", "photon (nonlinear)", "photon_source",
                {"pump_wavelength_nm": 797, "stokes_wavelength_nm": 1031, "pulse_duration_ps": 2,
                 "repetition_rate_mhz": 80, "modulation_frequency_mhz": 20, "pump_power_mw": 20,
                 "stokes_power_mw": 100},
                "Stokes beam is intensity-modulated; the loss in the pump is detected by lock-in. "
                "1e7/797 - 1e7/1031 = 2845 cm-1 (CH2).",
                constraints=[_c("raman_shift_cm1", 500, 3500, "vibrational window")])],
    "stm": [_el("Tunnelling bias between tip and sample", "electron (tunnelling current)", "generic_source",
                {"bias_voltage_v": 0.1, "setpoint_current_na": 1.0},
                "A bias of tenths of a volt drives tunnelling electrons across the vacuum gap; the current falls "
                "exponentially with distance.",
                constraints=[_c("bias_voltage_v", -5, 5, "tip and sample stability"),
                             _c("setpoint_current_na", 0.001, 10, "gap distance")])],
    "streak_camera": [_el("Pulsed laser", "photon", "photon_source",
                          {"wavelength_nm": 800, "pulse_duration_fs": 100},
                          "Ultrashort pulse whose time profile is mapped onto a spatial axis by the streak tube.",
                          constraints=[_c("pulse_duration_fs", 10, 1000000, "streak time resolution")])],
    "swi": [_el("Main magnet (3 T)", "spin", "spin_source", {"field_strength_T": 3.0}, "Clinical 3 T magnet."),
            _el("Long-echo gradient echo", "spin", "spin_source",
                {"echo_time_ms": 20, "repetition_time_ms": 30, "flip_angle_deg": 15},
                "Long echo time makes phase sensitive to susceptibility.",
                constraints=[_c("echo_time_ms", 5, 40, "susceptibility contrast vs signal loss")])],
    "talbot_lau": [_el("X-ray tube with source grating", "xray", "xray_source",
                       {"tube_voltage_kvp": 40, "tube_current_ma": 25, "focal_spot_mm": 1.0,
                        "design_energy_kev": 28},
                       "Laboratory tube; the source grating makes the large focal spot spatially coherent.",
                       constraints=[_c("design_energy_kev", 10, 100, "grating design energy sets visibility")])],
    "us_mri": [_el("Ultrasound transducer", "acoustic", "acoustic_source",
                   {"center_frequency_mhz": 5.0, "fractional_bandwidth_fraction": 0.6}, "Typical diagnostic array."),
               _el("Main magnet (3 T)", "spin", "spin_source", {"field_strength_T": 3.0}, "Clinical 3 T magnet.")],
    "xfel_sfx": [_el("X-ray free-electron laser", "xray", "xray_source",
                     {"photon_energy_kev": 9.5, "pulse_duration_fs": 40, "photons_per_pulse": 1e12,
                      "repetition_rate_hz": 120, "focus_diameter_um": 1.5},
                     "Femtosecond pulses outrun radiation damage ('diffract before destroy').",
                     constraints=[_c("pulse_duration_fs", 1, 100, "must be shorter than damage processes")])],
    "xray_crystallography": [_el("Synchrotron beam line", "xray", "xray_source",
                                 {"photon_energy_kev": 12.4, "flux_photons_per_s": 1e12, "beam_size_um": 100},
                                 "12.4 keV is 1.0 Angstrom.",
                                 constraints=[_c("absorbed_dose_mgy", None, 30,
                                                 "Garman limit at 100 K, to be verified")])],
    "xrf_imaging": [_el("X-ray tube or micro-beam", "xray", "xray_source",
                        {"tube_voltage_kvp": 50, "tube_current_ma": 0.5, "spot_size_um": 50},
                        "Excitation above the absorption edge of the target elements.",
                        constraints=[_c("tube_voltage_kvp", 10, 100, "must exceed the edges of the elements of interest")])],
}


def check(modality: str) -> List[str]:
    """Problems with a modality's drafted source elements (empty list when none)."""
    bad: List[str] = []
    for e in SOURCES.get(modality, []):
        if e["primitive"] not in SOURCE_PRIMITIVES:
            bad.append(f"unknown primitive {e['primitive']}")
        if e["validation"] != VALIDATION or e["verified_against"] is not None:
            bad.append("validation level not 'drafted'")
        keys = set(e["parameters"])
        for k, v in e["parameters"].items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                bad.append(f"{k}: not numeric")
            if not k.lower().endswith(UNIT_SUFFIXES):
                bad.append(f"{k}: no unit suffix")
        for c in e["constraints"]:
            if c["min"] is not None and c["max"] is not None and c["min"] > c["max"]:
                bad.append(f"constraint {c['parameter']}: min > max")
            if c["parameter"] in keys:
                v = e["parameters"][c["parameter"]]
                if (c["min"] is not None and v < c["min"]) or (c["max"] is not None and v > c["max"]):
                    bad.append(f"{c['parameter']}={v} is outside its own constraint")
    return bad
