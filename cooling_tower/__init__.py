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

Notation: as docs/notation.md (T in °C, Y humidity ratio per kg dry air, indices c coolant = water, a air, i in, o out,
ev evaporation). Tower-specific symbols: Me (Merkel number), L_fi (fill height), A_fr (fill cross-section),
Le_f (Lewis factor), primes '' = saturated at the local water temperature.

  config.py          the tower settings (one frozen Settings object per run, like comparison/config.py)
  geometry.py        fill cross-section, fill height, fill type
  fill.py            fill performance correlations (Merkel number per metre, loss coefficient), with validity ranges
  moist_air.py       saturation state at the water temperature (Y'', h''), Lewis factor after Bosnjakovic
  poppe.py           the Poppe equations, Runge-Kutta integration, iteration on the outlet humidity, supersaturation
  pressure_drop.py   air-side pressure drop of the tower
  water_balance.py   evaporation, drift, blowdown, make-up water
  design.py          design: required Merkel number over L/G, fill height and cross-section, cost optimum
  evaluate.py        one operating point -> one result row with the SAME columns as comparison/evaluate.run_point

Status: skeleton only -- every module holds the description of what it will do, no implementation yet.
"""
