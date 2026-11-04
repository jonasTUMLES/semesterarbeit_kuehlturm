"""
test_cooling_tower.py -- validation of the cooling-tower package.

moist_air.py (this file, now):
  - Kröger's property equations against the worked example of Kloppers' dissertation, App. G (Poppe analysis of an
    expanded metal fill; inputs p = 101712.27 Pa, T_ai = 9.7 °C, T_wo = 27.77 °C, T_wi = 39.67 °C):
    specific heats, vapour pressure, Y_sat_W, h_D, h_L_sat_W, Lewis factor at the first two Runge-Kutta steps (pp. G.2-G.5),
    and the inlet enthalpy i_mai = 25291.87496 J/kg for the given w_i = 0.00616336 (p. G.1)
  - internal consistency (inversion T_L(h_L, Y), limit of the Lewis factor, the water-side table)
  - CoolProp against Kröger (Y_sat_W, h_L): shows the size of the difference between the two
    property sets and that both use the same enthalpy reference (0 °C)

Planned (poppe.py):
  - Poppe: Kloppers dissertation, App. G, Tab. H.2: Me_P = 0.741 (Merkel 0.685)
  - limiting case: Poppe with Le_f = 1 and no evaporation loss (poppe.py validation switches) reproduces
    Perry's Handbook, Example 12-10 (41 -> 29.4 °C, wet bulb 25 °C, L/G = 1.2): KaV/L = 2.32 (p. 12-16)
  - the predecessor's tests (tests/test_solvers_golden.py etc.) must keep passing unchanged
"""

import numpy as np
import pytest

from cooling_tower import moist_air as ma

# --- Kloppers, dissertation (Stellenbosch 2003), App. G ---------------------------------------------------------
P_G = 101712.3              # atmospheric pressure [Pa] (p. G.1: 101712.27; p. G.3 uses 101712.3)
T_AI = 9.7                  # air inlet temperature [°C]
T_WO = 27.77                # water outlet = first Runge-Kutta point (level 0) [°C]
T_W12 = 30.745              # second intermediate step, 303.895 K [°C]
W_I = 0.00616336            # inlet humidity ratio [kg/kg dry air] (p. G.1, after Kröger)
I_MAI = 25291.87496         # inlet enthalpy [J/kg dry air] (p. G.1)


# =============================================================================
# Kröger's equations against Kloppers, App. G
# =============================================================================

def test_latent_heat_at_0C():
    # Kröger p. 448: i_fgwo = Delta_h_V0 = 2.5016e6 J/kg (Eq. A.4.5 at 273.15 K)
    assert ma.DELTA_H_V0 == pytest.approx(2.5016e6, rel=1e-4)


@pytest.mark.parametrize("T_W, c_pL, c_pD, c_pW", [
    (T_WO, 1006.567, 1874.385, 4191.744),      # p. G.2/G.3, evaluated at (300.92 + 273.15)/2 = 287.035 K
    (T_W12, 1006.603, 1875.654, 4189.941),     # p. G.5, evaluated at 288.5225 K
])
def test_specific_heats_kloppers(T_W, c_pL, c_pD, c_pW):
    assert ma.specific_heats(T_W) == pytest.approx((c_pL, c_pD, c_pW), abs=0.01)


@pytest.mark.parametrize("T_W, p_D, Y_sat_W", [
    (T_WO, 3729.0, 0.02379),                   # p. G.3
    (T_W12, 4427.4, 0.028454),                 # p. G.5
])
def test_saturation_kloppers(T_W, p_D, Y_sat_W):
    assert ma.p_D_sat_kroeger(T_W) == pytest.approx(p_D, abs=1.0)
    assert ma.Y_sat_W_kroeger(T_W, P_G) == pytest.approx(Y_sat_W, abs=2e-5)


def test_enthalpies_kloppers():
    # p. G.3: i_v(1,1) = 2553650 J/kg, i_masw(1,1) = 88711 J/kg (Kröger notation) -- Kloppers prints w_sw rounded to 0.02379 but
    # computes i_masw with the unrounded value (0.0237927...); with 0.02379 the enthalpy would be 7 J/kg lower
    assert ma.h_D(T_WO) == pytest.approx(2553650.0, abs=5.0)
    assert ma.h_L_sat_W(T_WO, ma.Y_sat_W_kroeger(T_WO, P_G)) == pytest.approx(88711.0, abs=1.0)


def test_lewis_factor_kloppers():
    # p. G.3: Le_f(1,1) = 0.9205 (Kloppers writes 0.865^0.667, the VDI 0.865^(2/3): 0.92048 vs. 0.92052)
    assert ma.Le_f(0.02379, W_I) == pytest.approx(0.9205, abs=1e-4)


def test_inlet_enthalpy_kloppers():
    # p. G.1: inlet state given (after Kröger) as w_i = 0.00616336 and i_mai = 25291.87496 J/kg at T_ai = 9.7 °C
    assert ma.h_L(T_AI, W_I) == pytest.approx(I_MAI, abs=0.1)


# =============================================================================
# Internal consistency
# =============================================================================

@pytest.mark.parametrize("T_L, Y", [(-5.0, 0.002), (9.7, W_I), (25.0, 0.015), (37.0, 0.04)])
def test_inversion_T_L(T_L, Y):
    assert ma.T_L_from_h_L(ma.h_L(T_L, Y), Y) == pytest.approx(T_L, abs=1e-5)


def test_lewis_factor_limit():
    # (a - 1)/ln(a) -> 1 for Y -> Y_sat_W: no division by zero, value 0.865^(2/3)
    assert ma.Le_f(0.02, 0.02) == pytest.approx(0.865**(2 / 3))
    assert ma.Le_f(0.02, 0.02 - 1e-10) == pytest.approx(0.865**(2 / 3), rel=1e-6)


def test_supersaturated_enthalpy_reduces_to_saturated():
    # Kröger Eq. (4.2.18) with Y = Y_sat_L (no mist) equals h_L (Eq. 4.2.4) at saturation
    assert ma.h_L_ss(20.0, 0.015, 0.015) == pytest.approx(ma.h_L(20.0, 0.015), abs=1e-6)


def test_water_side_table_matches_scalar_functions():
    T_grid = np.linspace(T_WO, 39.67, 5)
    tab = ma.water_side_table(T_grid, P_G, saturation='kroeger')
    for i, T in enumerate(T_grid):
        Y_sat_W = ma.Y_sat_W_kroeger(T, P_G)
        assert tab.Y_sat_W[i] == pytest.approx(Y_sat_W)
        assert tab.h_D[i] == pytest.approx(ma.h_D(T))
        assert tab.h_L_sat_W[i] == pytest.approx(ma.h_L_sat_W(T, Y_sat_W))
        assert (tab.c_pL[i], tab.c_pD[i], tab.c_pW[i]) == pytest.approx(ma.specific_heats(T))
    with pytest.raises(ValueError):
        ma.water_side_table(T_grid, P_G, saturation='other')


def test_range_warning_below_0C():
    with pytest.warns(UserWarning, match="outside the range"):
        ma.p_D_sat_kroeger(-5.0)


# =============================================================================
# CoolProp against Kröger -- size of the difference, common enthalpy reference
# =============================================================================

P_MUC = 95500.0             # Munich station pressure, about 955 hPa


@pytest.mark.parametrize("T_W", [10.0, 25.0, 40.0])
def test_Y_sat_W_coolprop_vs_kroeger(T_W):
    # both include the non-ideal (enhancement) effect; expected agreement within a few per mille
    assert ma.Y_sat_W_coolprop(T_W, P_MUC) == pytest.approx(ma.Y_sat_W_kroeger(T_W, P_MUC), rel=5e-3)


@pytest.mark.parametrize("T, Y", [(10.0, 0.005), (25.0, 0.012), (37.0, 0.03)])
def test_h_L_coolprop_vs_kroeger(T, Y):
    # CoolProp 'Hda' = enthalpy per kg dry air (real humid air, ASHRAE reference). Agreement here confirms that
    # Kröger's formulas and CoolProp share the reference 0 °C (dry air, liquid water); an offset would show up
    # as an absolute difference of the order of kJ/kg.
    import CoolProp.CoolProp as CP
    h_cp = CP.HAPropsSI('Hda', 'T', T + 273.15, 'P', P_MUC, 'W', Y)
    assert ma.h_L(T, Y) == pytest.approx(h_cp, abs=200.0)


def test_Y_sat_L_below_0C_runs():
    # CoolProp also covers air below 0 °C (inlet of cold hours); the value against a table is still to be checked
    Y = ma.Y_sat_L(-10.0, P_MUC)
    assert 0.0 < Y < 0.003
