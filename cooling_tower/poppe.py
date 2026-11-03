"""
poppe.py -- the Poppe method for a counterflow wet cooling tower.

Planned:
  - the governing equations with the water temperature as the independent variable: dY/dT_c, dh_a/dT_c, dMe/dT_c
    (VDI-Wärmeatlas N4, Eqs. 20, 21; Rubio-Castro et al. 2011, Eqs. 1-3), the local water flow m_dot_c(Y)
    (VDI N4 Eq. 24; Rubio-Castro Eq. 5) and the Lewis factor from moist_air.py
  - fourth-order Runge-Kutta from the bottom (cold water, air inlet) to the top (hot water, air outlet)
    (Kloppers dissertation App. B.4; Rubio-Castro App. B)
  - outer iteration on the outlet humidity ratio Y_o (unknown at the start, needed in the water-flow equation)
  - separate equations for supersaturated air (cold weather, plume) -- to be taken from Kloppers & Kröger 2005, IJHMT
  - results: required Merkel number, outlet air state (T_a_o, Y_o, h_a_o), evaporated water m_dot_ev
  - rating mode (geometry fixed, T_c_o unknown): iterate T_c_o until the required Merkel number equals the one the fill
    provides (fill.py at the actual G_w, G_a)

Validation switches: Lewis factor fixed to 1 and evaporation loss neglected in the energy balance reduce the
equations to Merkel's (VDI N4) -- used only to check the implementation against Perry's Example 12-10
(tests/test_cooling_tower.py), not as a method of the comparison.

Root finding: scipy brentq, as the comparison's fan control already does (comparison/operation._min_flow).
"""
