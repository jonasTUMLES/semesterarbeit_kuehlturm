# Cooling tower (`cooling_tower/`)

Semester thesis extension: an open wet cooling tower for the same cooling task as the dry cooler (water 37 °C -> 25 °C),
compared with the dry and the hybrid cooler over the Munich climate year. Status: skeleton, nothing implemented yet.

## Decisions

- **Open circuit**: the process water runs over the fill itself (no intermediate heat exchanger, no pinch).
- **Counterflow, forced draft**, film fill.
- **Poppe method** for design and operation. Merkel is not sufficient because it neglects the reduction of the water
  flow by evaporation in the energy balance and gives only the outlet enthalpy of the air; the evaporated water follows
  only from the extra assumption of saturated outlet air and is not accurate (Kloppers & Kröger 2005, J. Eng. Gas
  Turbines Power 127, pp. 1, 5; IJHMT 48, pp. 765 f.; VDI-Wärmeatlas N4, p. 1947). Merkel is not implemented; Poppe's limiting
  case (Le_f = 1, no evaporation loss) is used only to validate the implementation (Perry's Example 12-10).
- **Consistency rule**: fill data evaluated with Poppe (Kloppers dissertation, App. T).
- **Site**: Munich (DWD station 03379, `climate_data/`, shared with the dry-cooler comparison).

## Reuse of the predecessor's code

| what | where |
|---|---|
| climate year, wet-bulb temperature, station pressure | `src/inputs/weather.py` |
| fan power | `src/auxiliaries/economics.calc_fan_power` |
| cooling limit (wet bulb) | `src/thermo/precooling.calc_cooling_limit` |
| fluid and humid-air properties | `src/thermo/fluid_properties.py` |
| fan control pattern (smallest sufficient air flow, brentq) | `comparison/operation.control_point` |
| year loops, annual summary, costs | `comparison/operation.py` (needs rows with the columns of `comparison/evaluate.run_point`) |

## Open points

- [ ] Design ambient: wet-bulb percentile for Munich (ASHRAE 0.4 % or 1 %), cross-check with the climate year
- [ ] Equations for supersaturated air (Kloppers & Kröger 2005, IJHMT)
- [ ] Poppe loss-coefficient correlation of the fill (Kloppers dissertation, App. R-T)
- [ ] Common capital-cost basis for all three systems
- [ ] Wastewater fee for the blowdown; bleed-off of the pad (ASHRAE Ch. 41)
- [ ] Drift fraction and cycles of concentration (justify the values)

## Sources

Kloppers & Kröger 2005 (IJHMT 48, 765-777; J. Eng. Gas Turbines Power 127, 1-7); Kloppers, dissertation, Stellenbosch
2003; VDI-Wärmeatlas 2019, N4 (Erens, Reuter); Perry's Chemical Engineers' Handbook 2019, Sec. 12; Rubio-Castro et al.
2011 (Appl. Therm. Eng. 31, 3615-3628); Picardo & Variyar 2012 (Energy Convers. Manag. 57, 167-172); Kröger 2004.
