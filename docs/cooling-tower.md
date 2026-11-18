# Cooling tower (`cooling_tower/`)

Semester thesis extension: an open wet cooling tower for the same cooling task as the dry cooler (water 37 °C -> 25 °C),
compared with the dry and the hybrid cooler over the Munich climate year. Status: `moist_air.py`, `poppe.py` (design
mode, unsaturated and supersaturated air) and `fill_evaluation.py` implemented and tested
(`tests/test_cooling_tower.py`); fill correlation, pressure drop, water balance, design and operation still open.

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

## poppe.py -- implementation notes

Governing equations with the water temperature as independent variable, integrated with classic RK4 from the bottom
(air inlet) to the top; outer brentq iteration on the outlet humidity Y_o (needed in the water mass balance, VDI Gl. 24).

- **Unsaturated air** (Y <= Y_sat(T_L)): VDI-Wärmeatlas N4, Gl. (20), (21), (24), (25); Merkel number
  dMe/dT_W = c_pW / N with the local water flow inside the integral (Kröger Eq. 4.2.29), the definition used by
  Kloppers' fill data.
- **Supersaturated air** (mist, Y > Y_sat(T_L)): not in the VDI; Kröger 2004, Eqs. 4.2.18, 4.2.33-4.2.36 (pp. 242,
  246 f.) = Kloppers, dissertation, Eqs. B.33, B.38, B.42, B.43, B.47 (pp. B.7 f.). Y is the total water content
  (vapour + mist); the evaporation is driven by Y_sat(T_W) - Y_sat(T_L); the integrated enthalpy is h_L_ss (mist as
  liquid). For Y = Y_sat(T_L) the equations coincide with the unsaturated ones, so the switch is continuous
  (test `test_switch_is_continuous`).
- **Switching in every Runge-Kutta stage**: `air_state(h, Y, p, ...)` decides from the current state whether the air
  is supersaturated (T_L from the unsaturated formula, test Y > Y_sat(T_L) -- equivalent to Kloppers' T_wb > T_a) and
  returns T_L and Y_sat(T_L); in the supersaturated case T_L follows from h_L_ss (`moist_air.T_L_from_h_L_ss`).
- **Property model**: `saturation='coolprop'` (operation) or `'kroeger'` (validation), used for the saturation at the
  water AND at the air temperature.

### Validation against Kloppers' worked example (dissertation, App. G)

2 intervals, interval 1 unsaturated, interval 2 supersaturated. Kloppers decides the regime at the START of each
interval and keeps the saturation humidity w_sa of the start for all four Runge-Kutta stages (p. G.9: w_sa(2,1) =
w_sa(1); p. G.11: "by proceeding along the same lines"). Repeating exactly this procedure with `poppe.derivatives`
(test helper `kloppers_integrate`) reproduces every stage (j, k, l) and the end state:

| | Kloppers | code, his procedure |
|---|---|---|
| T_a level 1, w_sa | 290.8448 K, 0.012689 | 290.8448 K, 0.012689 |
| j, k, l stage (2,1) | 0.007599, 24957, 0.3731 | 0.007599, 24957, 0.3731 |
| w_o, i_ss, Me_P | 0.02226, 75378 J/kg, 0.7341 | 0.022265, 75378 J/kg, 0.73412 |
| T_ao | 297.85 K | 297.84 K |

Misprints in App. G: p. G.10 Lewis factor printed with 0.662 instead of 0.622 (value 0.9229 is with 0.622);
p. G.11 m_w/m_a printed 0.9548 once instead of 0.9584; l(2,3) printed 0.03364 instead of 0.3364; in step (1,2)
Kloppers reuses i_v(1,1) = 2553650 instead of 2559265 (0.1 J/kg in i_ma(1)).

**Finding**: the frozen w_sa makes Kloppers' procedure only first order in the interval size. With switching and
Y_sat(T_L) in every stage (`poppe.integrate`) the solution is converged already with 2 intervals; Kloppers' procedure
approaches it only slowly (test `test_stage_switching_vs_kloppers_procedure`):

| intervals | Kloppers' procedure Y_o / Me | poppe.integrate Y_o / Me |
|---|---|---|
| 2 | 0.02226 / 0.7341 | 0.02133 / 0.7365 |
| 20 | 0.02150 / 0.7360 | 0.02133 / 0.7365 |
| 160 | 0.02135 / 0.7364 | 0.02133 / 0.7365 |

So the App. G outlet humidity 0.02226 carries a discretisation error of about 4 % (Me about 0.3 %).

## Fill data (`fill_evaluation.py`, `cooling_tower/data/kloppers2003_film_fill.csv`)

Kloppers, dissertation, App. T: cross-corrugated film fill, fill heights 0.6 / 0.9 / 1.2 m, 18 tests each, frontal
area 2.25 m^2. Measurements (Tables T.1/T.5/T.9) and his evaluation (Tables T.2/T.6/T.10) transcribed from the scanned
PDF and checked number by number against the page images; m_w/G_w = m_a/G_a = 2.25 m^2 in all 54 rows. G_a is formed
with the dry air flow. Range: G_w 2.7-6.8 kg/(m^2 s), G_a 1.2-4.2 kg/(m^2 s), T_wi 37.3-47.6 °C.

Each test is re-evaluated with `poppe.design` and CoolProp, i.e. with exactly the code and properties of the design
(consistency rule). Inlet humidity from dry and wet bulb (`moist_air.Y_inlet_wb`, CoolProp input 'B').
Result (`python -m cooling_tower.fill_evaluation`, test `test_fill_tests_kloppers_coolprop`):

| | CoolProp | Kröger properties (cross-check) |
|---|---|---|
| Me/L_fi vs. Kloppers' Me_P, mean | -0.13 % | +0.09 % |
| Me/L_fi vs. Kloppers' Me_P, max | 0.33 % | 0.9 % |
| air outlet temperature vs. Kloppers' T_aoP, max | 0.04 K | 0.03 K |
| air outlet temperature vs. measurement, mean / max | -0.31 K / 1.7 K | same |
| supersaturated at the outlet | 54 of 54 | 54 of 54 |

The remaining -0.13 % is the property model (systematic). The deviation from the measured outlet temperature is the
same for Kloppers' own Poppe values, i.e. a property of the model / measurement, not of the implementation.
The column `Me_P_calc` is the consistent data base for the fill correlation (`fill.py`).

## Pressure drop (`pressure_drop.py`)

Forced draft: fan at the bottom, blowing into a plenum below the fill; the fan has to provide the total pressure drop
(VDI N4, 6.4). Framework VDI N4 6.3/6.4: K = Delta_p/(rho v^2/2) (Gl. 36), all coefficients referred to the mean
state in the fill and added (Gl. 40), Delta_p = K_tot rho_fill v_fill^2/2 (Gl. 41).

Reference state: m_av = m_L (1 + Y) below (i) / above (o) the fill, G_av,m = (G_av,i + G_av,o)/2,
1/rho_av,m = 0.5 (1/rho_av,i + 1/rho_av,o). Conversion of a coefficient to this state: Kloppers (D.5)
K_fi = K (rho_m/rho)(m_av/m_av,m)^2 (A_fr/A)^2 (exact form of the VDI conversion in 6.3).

| VDI 6.3 | component | coefficient (own reference state) | source |
|---|---|---|---|
| 1 | fan inlet | zeta_E = 0.05 rounded (0.5 sharp-edged, sensitivity), at the fan | VDI L1.3, Sec. 1.3, Fig. 4 |
| 1 | fan -> plenum | (1 - A_F/A_plenum)^2, at the fan (Borda-Carnot) | VDI L1.3 Eq. (7) = Kloppers (D.16) |
| 2 | rain zone | neglected | VDI N4 6.2; range of Kloppers (D.9) not met |
| 3 | fill supports | 0.5 (incl. contraction), below the fill | Kloppers App. J |
| 4 | fill | K_fdm = K_fdm1 L_fi, (T.4); + momentum (D.19); mean fill state | Kloppers App. T, App. D |
| – | expansion after the fill | (D.16) with sigma = 1 -> 0 | Kloppers (D.16) |
| 5 | spray zone | L_sp [0.4 G_W/G_L + 1], above the fill; L_sp = 0.15 m (as in the fill tests) | Kloppers (D.6) |
| 5 | water distribution | 0.5, above the fill | Kloppers D.2.3 |
| 6+7 | drift eliminator | 27.4892 Ry^-0.14247, Ry = m_av/(mu_av A_fr), above the fill | Kloppers (D.7), type c App. I |
| 8 | outlet shell | none (open outlet assumed -- to be confirmed) | – |
| 9 | exit | K = 1 at the outlet (Borda-Carnot, A_out/A_ambient -> 0) | VDI N4 6.3, VDI L1.3 Eq. (7) |

Fill: K_fdm1 = (5.154914 G_W^0.877646 G_L^-1.462034 + 10.806728 G_W^0.226578 G_L^0.293222) L_fi^-0.236292
(Kloppers T.4, r^2 = 0.904; against his 54 values K_fdm,P: rms 5.3 %, max 14.7 %; K_P is on average 0.3 % below
K_M, so the heat-transfer model hardly matters for the pressure drop). Momentum (D.19):
K_fi = K_fdm + (G_av,o^2/rho_o - G_av,i^2/rho_i)/(G_av,m^2/rho_m); without factor 2 because the coefficients refer
to the total pressure (D.2/D.3). Buoyancy neglected (forced draft).

Not used: Kloppers (D.10)-(D.12) (round towers with an open inlet around the circumference, d_i = 80-120 m),
(D.13)/(D.14) (induced-draft rectangular towers), (D.15) (supports of a tower shell), (D.8)/(D.9) (rain zone).

Properties: humid air after Kröger App. A (density A.3.1, viscosity A.1.3/A.2.3/A.3.3, `moist_air.rho_av`,
`mu_av`), as in Kloppers' examples. Supersaturated air: the gas holds Y_sat(T_L), the mist adds mass but no volume.
Fan power P_F = V_F Delta_p / eta_F, V_F at the fan (inlet air), eta_F = 0.7 (as in the predecessor's code).

Validation against Kloppers App. J (= Kröger Vol. 2, pp. 207-223): rho_av, mu_av, the conversions K_fsfi, K_wdfi,
K_spfi, K_defi, K_upfi and the momentum correction of the fill are reproduced (test
`test_conversion_to_fill_kroeger_example`). Misprint: the fan upstream coefficient on p. 220 is printed as 0.5, the
value used is the specified 0.52 (p. 208).

Example (37 -> 25 °C, L/G = 1, G_W = G_L = 3 kg/(m^2 s), L_fi = 1.2 m, A_F = A_fr/3, Kröger properties):
Delta_p = 160 Pa -- fill 110 Pa, drift eliminator 22 Pa, fan -> plenum 17 Pa, exit 4.5 Pa, the rest 1-2 Pa each.

## Change log

**2026-11-18**
- New `pressure_drop.py`: loss coefficients of all components (table above), conversion (D.5), total pressure drop
  (VDI Gl. 40/41), fan power.
- `moist_air.py`: density and viscosity of humid air after Kröger App. A (`rho_av`, `mu_a`, `mu_v`, `mu_av`).
- Tests against Kloppers App. J and against the 54 measured loss coefficients.

**2026-11-17**
- `poppe.py`: supersaturated branch (Kröger Eqs. 4.2.33-4.2.36); new `saturation_function` and `air_state`;
  `derivatives` and `integrate` take `p` and `Y_sat_L_fn`; switching in every RK stage; air temperature and
  supersaturation flag per level from `air_state`; warning "supersaturated branch not implemented" removed.
- `moist_air.py`: new `T_L_from_h_L_ss` (temperature of supersaturated air, brentq, lower bound = temperature from
  the unsaturated formula) and `Y_inlet_wb` (humidity from the wet bulb, CoolProp).
- New `fill_evaluation.py` and `data/kloppers2003_film_fill.csv` (Kloppers App. T).
- Tests: level-1 test replaced by the full App. G procedure (helper `kloppers_integrate`); new tests for the
  supersaturated state and stage (2,1), convergence against Kloppers' procedure, continuity of the switch, inversion
  T_L(h_L_ss), Munich winter case with CoolProp, the 54 fill tests.

## Open points

- [ ] Design ambient: wet-bulb percentile for Munich (ASHRAE 0.4 % or 1 %), cross-check with the climate year
- [x] Equations for supersaturated air (Kröger 2004, Eqs. 4.2.18, 4.2.33-4.2.36; Kloppers App. B) -- 2026-11-17
- [ ] Outlet of the tower really open (no outlet shell)? Fan diameter / plenum geometry (`geometry.py`)
- [ ] Fit of the fill correlation Me/L_fi = c1 G_W^c2 G_L^c3 L_fi^c4 (T_W,in^c5) to the 54 re-evaluated tests (`fill.py`)
- [ ] Poppe loss-coefficient correlation of the fill (Kloppers dissertation, App. R-T)
- [ ] Common capital-cost basis for all three systems
- [ ] Wastewater fee for the blowdown; bleed-off of the pad (ASHRAE Ch. 41)
- [ ] Drift fraction and cycles of concentration (justify the values)

## Sources

Kloppers & Kröger 2005 (IJHMT 48, 765-777; J. Eng. Gas Turbines Power 127, 1-7); Kloppers, dissertation, Stellenbosch
2003; VDI-Wärmeatlas 2019, N4 (Erens, Reuter); Perry's Chemical Engineers' Handbook 2019, Sec. 12; Rubio-Castro et al.
2011 (Appl. Therm. Eng. 31, 3615-3628); Picardo & Variyar 2012 (Energy Convers. Manag. 57, 167-172); Kröger 2004.
