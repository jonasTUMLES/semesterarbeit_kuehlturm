"""
cooling_tower -- open wet cooling tower (forced draft, counterflow, film fill), Poppe method.

Semester thesis extension of the dry-cooler model: the same cooling task (water 37 °C -> 25 °C) solved with an open wet
cooling tower, so that dry cooler, hybrid cooler (dry + adiabatic pad) and cooling tower can be compared over the Munich
climate year with the existing comparison machinery (comparison/operation.py, annual_summary, add_costs).

Design decisions (see docs/cooling-tower.md):
  - open circuit: the process water itself runs over the fill (no intermediate heat exchanger, no pinch)
  - counterflow, forced draft
  - Poppe method for design and operation, fill data evaluated with Poppe (consistency rule, Kloppers & Kröger 2005);
    Merkel is not implemented -- it is ruled out in the methodology (inaccurate water consumption)

Notation: inside the package the letters of the VDI-Wärmeatlas N4 (h enthalpy, Delta_h_V0, c_p; media L Luft,
D Dampf, W Wasser: T_W, T_L, h_L, c_pW, ...), with the repository's conventions (T in °C instead of theta, Y for the
humidity ratio per kg dry air as in VDI M8 and src/thermo/precooling.py, state indices i/o, ev evaporation; see
docs/notation.md). The table VDI / Kröger / code is in moist_air.py. evaluate.py maps to the comparison's columns
(T_c = water, T_a = air). Tower-specific symbols: Me (Merkel number), L_fi (fill height), A_fr (fill cross-section),
Le_f (Lewis factor), sat_W / sat_L = saturated at the water / air temperature.

  config.py          the tower settings (one frozen Settings object per run, like comparison/config.py)
  geometry.py        fill cross-section, fill height, fill type
  fill.py            fill performance correlations (Merkel number per metre, loss coefficient), with validity ranges
  moist_air.py       properties: specific heats, humidity ratios, enthalpies, Lewis factor, water-side table  [done]
  poppe.py           the Poppe equations, Runge-Kutta integration, iteration on the outlet humidity, supersaturation
  pressure_drop.py   air-side pressure drop of the tower
  water_balance.py   evaporation, drift, blowdown, make-up water
  design.py          design: required Merkel number over L/G, fill height and cross-section, cost optimum
  evaluate.py        one operating point -> one result row with the SAME columns as comparison/evaluate.run_point

Status: moist_air.py implemented and tested (tests/test_cooling_tower.py); the other modules hold the description of
what they will do, no implementation yet.
"""
