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

poppe.py:
  - Kloppers' dissertation, App. G (same inputs, 2 intervals, w_o = 0.02226 as given there), pp. G.2-G.12:
    interval 1 unsaturated, interval 2 supersaturated. Kloppers decides the regime at the interval START and keeps the
    saturation humidity w_sa of the start for all four Runge-Kutta stages of the interval. The helper
    kloppers_integrate() repeats exactly this procedure with poppe.derivatives -> his increments j, k, l of every
    stage and the end state (w_o = 0.02226, i_ss = 75378 J/kg, T_ao = 297.85 K, Me_P = 0.7341) are reproduced, which
    checks the unsaturated AND the supersaturated equations.
  - poppe.integrate itself decides the regime and Y_sat_L in EVERY stage. Kloppers' procedure is only of first order
    in the interval size (frozen w_sa) and converges to the same solution for many intervals
  - switching is continuous at saturation; inversion T_L(h_L_ss)
  - numerics: convergence of RK4 with the number of intervals, interval size option, mass and energy balance,
    error for too little air

Planned (poppe.py):
  - limiting case: Poppe with Le_f = 1 and no evaporation loss (poppe.py validation switches) reproduces
    Perry's Handbook, Example 12-10 (41 -> 29.4 °C, wet bulb 25 °C, L/G = 1.2): KaV/L = 2.32 (p. 12-16)
  - the predecessor's tests (tests/test_solvers_golden.py etc.) must keep passing unchanged
"""

import warnings

import numpy as np
import pytest

from cooling_tower import moist_air as ma
from cooling_tower import poppe as pp

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


# =============================================================================
# poppe.py -- Kloppers, dissertation, App. G (unsaturated interval 1, supersaturated interval 2)
# =============================================================================

P_G1 = 101712.27            # p. G.1
T_WI = 39.67                # water inlet [°C] (312.82 K)
M_W_I = 3.999               # water inlet flow [kg/s]
M_L = 4.134                 # dry air flow [kg/s]
W_O = 0.02226               # outlet humidity ratio used in Kloppers' mass balance (converged value, p. G.2)
SAT_K = ma.Y_sat_W_kroeger  # Kröger's saturation formula, also at the air temperature (Kloppers uses A.2.1/A.3.5)


@pytest.fixture
def kloppers_table():
    return ma.water_side_table(pp.water_grid(T_WI, T_WO, 2), P_G1, saturation='kroeger')


def kloppers_integrate(tab, monkeypatch):
    """Kloppers' procedure (App. G): the regime (unsaturated / supersaturated) and the saturation humidity w_sa are
    determined at the START of each interval and kept for its four Runge-Kutta stages (p. G.9: w_sa(2,1) = w_sa(1);
    p. G.11: 'by proceeding along the same lines'). Returns the levels (Y, h, Me) and the stage increments."""
    n = (len(tab.T_W) - 1) // 2
    out = [np.array((W_I, I_MAI, 0.0))]
    stages = []
    air_state = pp.air_state
    for k in range(n):
        i0, im, i1 = 2 * k, 2 * k + 1, 2 * k + 2
        dT = tab.T_W[i1] - tab.T_W[i0]
        Y, h = out[-1][0], out[-1][1]
        state = air_state(h, Y, P_G1, SAT_K)
        monkeypatch.setattr(pp, 'air_state', lambda *_, s=state: s)      # freeze regime and w_sa
        args = (tab, M_W_I, M_L, W_O, P_G1, SAT_K)
        s1 = np.array(pp.derivatives(Y, h, i0, *args))
        s2 = np.array(pp.derivatives(Y + 0.5 * dT * s1[0], h + 0.5 * dT * s1[1], im, *args))
        s3 = np.array(pp.derivatives(Y + 0.5 * dT * s2[0], h + 0.5 * dT * s2[1], im, *args))
        s4 = np.array(pp.derivatives(Y + dT * s3[0], h + dT * s3[1], i1, *args))
        monkeypatch.setattr(pp, 'air_state', air_state)
        stages.append([dT * s for s in (s1, s2, s3, s4)])
        out.append(out[-1] + dT / 6.0 * (s1 + 2.0 * s2 + 2.0 * s3 + s4))
    return np.array(out), stages


def test_rk4_first_step_kloppers(kloppers_table):
    # p. G.3/G.4: increments of step (1,1) = Delta T_W (5.95 K) times the derivative: j = 0.006982, k = 24537.63,
    # l = 0.41635
    d = pp.derivatives(W_I, I_MAI, 0, kloppers_table, M_W_I, M_L, W_O, P_G1, SAT_K)
    dT = kloppers_table.T_W[2] - kloppers_table.T_W[0]
    assert dT == pytest.approx(5.95)
    assert dT * d[0] == pytest.approx(0.006982, abs=1e-6)
    assert dT * d[1] == pytest.approx(24537.63, abs=0.2)
    assert dT * d[2] == pytest.approx(0.41635, abs=1e-5)


def test_supersaturated_state_level_1_kloppers():
    # p. G.8: at level 1 (w = 0.0133338, i = 50019.67 J/kg) the air is supersaturated, T_a = 290.8448 K,
    # w_sa = 0.012689; p. G.10: Le_f = 0.9229 (printed with 0.662 instead of 0.622 -- misprint, value is with 0.622)
    T_L, Y_sat_L, supersat = pp.air_state(50019.67, 0.0133338, P_G1, SAT_K)
    assert supersat
    assert T_L == pytest.approx(290.8448 - 273.15, abs=1e-3)
    assert Y_sat_L == pytest.approx(0.012689, abs=1e-6)
    assert ma.Le_f_supersat(SAT_K(33.72, P_G1), Y_sat_L) == pytest.approx(0.9229, abs=1e-4)


def test_supersaturated_step_2_1_kloppers(kloppers_table):
    # pp. G.10/G.11: first stage of interval 2 (T_W = 306.87 K, supersaturated): j = 0.007599, k = 24957, l = 0.3731
    # (in k Kloppers prints m_w/m_a = 0.9548 once, the value used is 0.9584)
    d = pp.derivatives(0.0133338, 50019.67, 2, kloppers_table, M_W_I, M_L, W_O, P_G1, SAT_K)
    assert 5.95 * d[0] == pytest.approx(0.007599, abs=1e-6)
    assert 5.95 * d[1] == pytest.approx(24957.0, abs=1.0)
    assert 5.95 * d[2] == pytest.approx(0.3731, abs=1e-4)


def test_kloppers_procedure_app_g(kloppers_table, monkeypatch):
    levels, stages = kloppers_integrate(kloppers_table, monkeypatch)
    # level 1 (p. G.6): w = 0.0133338, i_ma = 50019.67 (Kloppers reuses i_v(1,1) in step (1,2): 0.1 J/kg), Me = 0.3984
    assert levels[1][0] == pytest.approx(0.0133338, abs=2e-7)
    assert levels[1][1] == pytest.approx(50019.67, abs=0.3)
    assert levels[1][2] == pytest.approx(0.3984, abs=1e-4)
    # interval 2, stages (2,2)-(2,4) (p. G.11; l(2,3) printed 0.03364, used 0.3364)
    j = [s[0] for s in stages[1]]
    k = [s[1] for s in stages[1]]
    l = [s[2] for s in stages[1]]
    assert j[1:] == pytest.approx([0.008957, 0.008973, 0.010124], abs=2e-6)
    assert k[1:] == pytest.approx([25346.0, 25365.0, 25771.0], abs=1.0)
    assert l[1:] == pytest.approx([0.33598, 0.3364, 0.2965], abs=1e-4)
    # level 2 = fill outlet (p. G.11/G.12): w_o = 0.02226, i_ss = 75378 J/kg, Me_P = 0.7341, T_ao = 297.8508 K
    assert levels[2][0] == pytest.approx(0.02226, abs=1e-5)
    assert levels[2][1] == pytest.approx(75378.0, abs=1.0)
    assert levels[2][2] == pytest.approx(0.7341, abs=1e-4)
    T_L, _, supersat = pp.air_state(levels[2][1], levels[2][0], P_G1, SAT_K)
    assert supersat
    assert T_L == pytest.approx(297.8508 - 273.15, abs=5e-3)


def test_stage_switching_vs_kloppers_procedure(monkeypatch):
    # Kloppers' frozen w_sa is first order in the interval size; for many intervals it approaches the solution of
    # poppe.integrate (regime and Y_sat_L in every stage), which is already converged with 2 intervals
    tab_fine = ma.water_side_table(pp.water_grid(T_WI, T_WO, 160), P_G1, saturation='kroeger')
    levels, _ = kloppers_integrate(tab_fine, monkeypatch)
    tab = ma.water_side_table(pp.water_grid(T_WI, T_WO, 5), P_G1, saturation='kroeger')
    Y, h, Me = pp.integrate(tab, W_I, I_MAI, M_W_I, M_L, W_O, P_G1, SAT_K)
    assert levels[-1][0] == pytest.approx(Y[-1], rel=2e-3)
    assert levels[-1][2] == pytest.approx(Me[-1], rel=2e-4)


def test_switch_is_continuous(kloppers_table):
    # at exactly saturated air (no mist) both sets of equations coincide: 1 J/kg above and below the saturated
    # enthalpy give practically the same derivatives
    Y = SAT_K(15.0, P_G1)
    h_sat = ma.h_L(15.0, Y)
    args = (kloppers_table, M_W_I, M_L, W_O, P_G1, SAT_K)
    assert not pp.air_state(h_sat + 1.0, Y, P_G1, SAT_K)[2]
    assert pp.air_state(h_sat - 1.0, Y, P_G1, SAT_K)[2]
    d_unsat = pp.derivatives(Y, h_sat + 1.0, 2, *args)
    d_ss = pp.derivatives(Y, h_sat - 1.0, 2, *args)
    assert d_ss == pytest.approx(d_unsat, rel=1e-3)


@pytest.mark.parametrize("T_L, Y", [(17.69, 0.0133338), (24.70, 0.02226), (12.0, 0.0100)])
def test_inversion_T_L_supersaturated(T_L, Y):
    h = ma.h_L_ss(T_L, Y, SAT_K(T_L, P_G1))
    assert ma.T_L_from_h_L_ss(h, Y, P_G1, SAT_K) == pytest.approx(T_L, abs=1e-5)


def test_design_kloppers():
    # full design with outer iteration: supersaturated from level 1, no warning; Me close to Kloppers' 0.7341
    # (his value carries the error of the frozen w_sa, see above)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = pp.design(T_WI, T_WO, M_W_I, M_L, T_AI, W_I, P_G1, n_intervals=2, saturation='kroeger')
        r20 = pp.design(T_WI, T_WO, M_W_I, M_L, T_AI, W_I, P_G1, n_intervals=20, saturation='kroeger')
    assert r.supersaturated
    assert list(r.supersat_profile) == [False, True, True]
    assert r.Me == pytest.approx(0.7341, rel=0.005)
    assert r.Me == pytest.approx(r20.Me, rel=1e-4)
    assert r.Y_o == pytest.approx(r20.Y_o, rel=1e-3)


# =============================================================================
# poppe.py -- numerics and balances
# =============================================================================

def _design(**kw):
    args = dict(T_W_i=37.0, T_W_o=25.0, m_dot_W_i=10.0, m_dot_L=10.0, T_L_i=30.0, Y_i=0.012, p=P_MUC,
                saturation='kroeger')
    args.update(kw)
    return pp.design(**args)


def test_design_unsaturated_summer_case():
    # 37 -> 25 °C, warm humid air, L/G = 1: unsaturated throughout, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _design()
    assert not r.supersaturated
    assert r.Y[0] == pytest.approx(0.012)
    assert r.Y_o == pytest.approx(r.Y[-1], abs=1e-9)               # outer iteration converged
    assert np.all(np.diff(r.Y) > 0) and np.all(np.diff(r.Me_profile) > 0)


def test_rk4_convergence():
    # RK4: the result hardly changes with the number of intervals
    assert _design(n_intervals=5).Me == pytest.approx(_design(n_intervals=80).Me, rel=1e-5)


def test_interval_size_option():
    assert pp.n_intervals_for(T_WI, T_WO, dT_interval=1.0) == 12      # 11.9 K -> 12 intervals of 0.99 K
    assert pp.n_intervals_for(37.0, 25.0, dT_interval=0.5) == 24      # exactly divisible, no extra interval
    assert _design(dT_interval=0.5).n_intervals == 24


def test_mass_and_energy_balance():
    r = _design()
    assert r.m_dot_W_o == pytest.approx(10.0 - r.m_dot_ev)            # Gl. (24) at the bottom
    # energy: the equations balance m_L dh_L = m_W c_pW dT_W + c_pW T_W dm_W (Kröger Eqs. 4.2.25/4.2.26) with c_pW at
    # each point. Since c_pW T is not the exact water enthalpy (c_pW depends on T), the integral differs from
    # m_W_i c_pW T_W_i - m_W_o c_pW T_W_o by integral m_W T_W dc_pW (about 0.4 % here). With this term subtracted the
    # balance must close up to the RK4 and trapezoid error.
    cp = np.array([ma.specific_heats(float(T))[2] for T in r.T_W])
    Q_air = 10.0 * (r.h_L_o - r.h_L[0])
    Q_water = r.m_dot_W[-1] * cp[-1] * r.T_W[-1] - r.m_dot_W[0] * cp[0] * r.T_W[0]
    Q_dcp = np.trapezoid(r.m_dot_W * r.T_W, cp)
    assert Q_air == pytest.approx(Q_water - Q_dcp, rel=1e-6)


def test_too_little_air_raises():
    with pytest.raises(pp.PoppeError, match="too little air"):
        _design(m_dot_L=3.0)


def test_design_winter_coolprop():
    # Munich winter with CoolProp (operation): air below 0 °C, supersaturated almost through the whole fill (mist,
    # plume); the run must work without warnings and give a plausible outlet state
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = pp.design(37.0, 25.0, 10.0, 10.0, -5.0, 0.0022, P_MUC)
    assert r.supersaturated
    assert r.Me > 0.0
    assert -5.0 < r.T_L_o < 37.0
    assert 0.0022 < r.Y_o < ma.Y_sat_W_coolprop(37.0, P_MUC)


# =============================================================================
# fill_evaluation.py -- Kloppers' 54 film fill tests (App. T) evaluated with poppe.design and CoolProp
# =============================================================================

def test_fill_tests_kloppers_coolprop():
    # Me per metre against Kloppers' Poppe evaluation (Tables T.2/T.6/T.10). With CoolProp: on average -0.13 %
    # (systematic, property model), at most 0.33 %; air outlet temperature at most 0.04 K from Kloppers' T_ao_P.
    # (With Kröger's property equations instead of CoolProp: +0.09 % on average, at most 0.9 %, checked separately.)
    # All tests are supersaturated at the outlet.
    from cooling_tower import fill_evaluation as fe
    out = fe.evaluate_tests(n_intervals=10)
    assert len(out) == 54
    assert out['supersat_calc'].all()
    assert out['dMe_rel'].abs().mean() < 0.003
    assert out['dMe_rel'].abs().max() < 0.005
    assert out['dT_ao_P'].abs().max() < 0.06


# =============================================================================
# fill.py -- Poppe correlation of the film fill
# =============================================================================

def _kloppers_poppe_fit():
    # fit to Kloppers' own Poppe values (Tables T.2/T.6/T.10) -- needs no CoolProp
    from cooling_tower import fill, fill_evaluation as fe
    return fill.fit_correlation(fe.load_tests(), y='Me_P', name='Kloppers Me_P')


def test_fill_fit_kloppers_values():
    corr, stats = _kloppers_poppe_fit()
    assert stats.n == 54
    assert stats.R2 > 0.98
    assert stats.rms_rel < 0.05
    assert corr.G_W_range == pytest.approx((2.687, 6.831))
    assert corr.L_fi_range == pytest.approx((0.6, 1.2))


def test_fill_height_is_inverse_of_Me():
    corr, _ = _kloppers_poppe_fit()
    for G_W, G_L, L_fi in [(3.0, 2.5, 0.8), (5.5, 3.5, 1.1)]:
        assert corr.fill_height(corr.Me(G_W, G_L, L_fi), G_W, G_L) == pytest.approx(L_fi, rel=1e-12)


def test_fill_range_warning():
    corr, _ = _kloppers_poppe_fit()
    with pytest.warns(UserWarning, match="extrapolation"):
        corr.Me_per_m(8.0, 2.5, 0.9)


def test_fill_correlation_own_evaluation():
    # the correlation used in the design: fitted to the own re-evaluation (CoolProp). Since the own Me differ from
    # Kloppers' Poppe values by only -0.13 % on average, the coefficients must be practically the same
    from cooling_tower import fill
    if not fill.DATA_POPPE.exists():
        pytest.skip("run  python -m cooling_tower.fill_evaluation  to create the data base of the correlation")
    own = fill.film_fill_poppe()
    ref, _ = _kloppers_poppe_fit()
    assert own.c1 == pytest.approx(ref.c1, rel=0.01)
    assert (own.c2, own.c3, own.c4) == pytest.approx((ref.c2, ref.c3, ref.c4), abs=0.01)


# =============================================================================
# pressure_drop.py -- against Kloppers' worked example (App. J = Kröger 2004, Vol. 2, Sec. 8.3, pp. 207-223)
# =============================================================================

# states of the example: 1 below the fill, 5 above the fill, 15 mean (p. 211 and pp. 218-220)
KJ = dict(T1=33.5, w1=0.016569, p1=101325.0, rho1=1.1397, m1=438.4,
          T5=306.7645 - 273.15, w5=0.033922, p5=101170.6, rho5=1.12634, m5=445.8834, mu5=1.84273e-5,
          rho15=1.132982, m15=442.1426, A_fr=144.0, G_w=2.86111, G_a=2.994823)


def test_humid_air_properties_kroeger_example():
    # p. 211 / p. 218: rho_av1 = 1.1397, rho_av5 = 1.12634 kg/m^3 (Kröger A.3.1), mu_av5 = 1.84273e-5 kg/(m s) (A.3.3)
    assert ma.rho_av(KJ['T1'], KJ['w1'], KJ['p1']) == pytest.approx(KJ['rho1'], abs=1e-4)
    assert ma.rho_av(KJ['T5'], KJ['w5'], KJ['p5']) == pytest.approx(KJ['rho5'], abs=1e-4)
    assert ma.mu_a(KJ['T5']) == pytest.approx(1.8779e-5, rel=1e-4)
    assert ma.mu_v(KJ['T5']) == pytest.approx(1.0254e-5, rel=1e-4)
    assert ma.mu_av(KJ['T5'], KJ['w5']) == pytest.approx(KJ['mu5'], rel=1e-4)


def test_conversion_to_fill_kroeger_example():
    # (inputs as printed, rounded -> agreement to about 1e-5)
    # pp. 218-220: K_fsfi = 0.4886764, K_wdfi = 0.5114949, K_upfi = 4.36576 (fan d_F = 8 m), K_spfi = 0.706958
    # (L_sp = 0.5 m), K_defi = 5.0649 (type c), fill with momentum K_fi = 4.487634 from K_fdm = 4.441997
    from cooling_tower import pressure_drop as pd_
    import math
    m, A = (KJ['rho15'], KJ['m15'], KJ['A_fr']), KJ['A_fr']
    assert pd_.refer_to_fill(0.5, KJ['rho1'], KJ['m1'], A, *m) == pytest.approx(0.4886764, abs=1e-5)
    assert pd_.refer_to_fill(0.5, KJ['rho5'], KJ['m5'], A, *m) == pytest.approx(0.5114949, abs=1e-5)
    # fan upstream loss: specified K_up = 0.52 (p. 208); the formula on p. 220 shows 0.5, but 4.36576 is with 0.52
    assert pd_.refer_to_fill(0.52, KJ['rho5'], KJ['m5'], math.pi * 8.0**2 / 4.0, *m) == pytest.approx(4.36576, abs=1e-4)
    K_sp = pd_.K_spray_zone(0.5, KJ['G_w'], KJ['G_a'])
    assert pd_.refer_to_fill(K_sp, KJ['rho5'], KJ['m5'], A, *m) == pytest.approx(0.706958, abs=1e-5)
    K_de = pd_.K_drift_eliminator(KJ['m5'], KJ['mu5'], A)
    assert pd_.refer_to_fill(K_de, KJ['rho5'], KJ['m5'], A, *m) == pytest.approx(5.0649, abs=1e-3)
    assert pd_.K_fill(4.441997, KJ['m1'] / A, KJ['rho1'], KJ['m5'] / A, KJ['rho5']) == pytest.approx(4.487634, abs=1e-5)


def test_fill_loss_coefficient_T4_against_measurements():
    # Kloppers (T.4) against his 54 values K_fdm,P (Tables T.2/T.6/T.10): rms 5.3 %, max 14.7 %
    from cooling_tower import pressure_drop as pd_, fill_evaluation as fe
    d = fe.load_tests()
    K = [pd_.K_fdm1_kloppers(r.G_w, r.G_a, r.L_fi) for r in d.itertuples()]
    rel = np.array(K) / d['K_fdm_P'].to_numpy() - 1.0
    assert np.sqrt(np.mean(rel**2)) < 0.06
    assert np.abs(rel).max() < 0.16
    assert abs(rel.mean()) < 0.01


def test_borda_carnot():
    from cooling_tower import pressure_drop as pd_
    assert pd_.K_expansion(1.0, 3.0) == pytest.approx((2.0 / 3.0)**2)
    assert pd_.K_expansion(2.0, 2.0) == 0.0                     # no expansion (e.g. above the fill)
    with pytest.raises(ValueError):
        pd_.K_expansion(3.0, 1.0)


def _tower_case(**kw):
    # 37 -> 25 °C, L/G = 1, G_W = G_L = 3 kg/(m^2 s), fill 1.2 m, fan area A_fr/3
    from cooling_tower import pressure_drop as pd_
    r = pp.design(37.0, 25.0, 30.0, 30.0, 30.0, 0.012, P_MUC, n_intervals=10, saturation='kroeger')
    path = pd_.FlowPath(A_fr=10.0, A_F=10.0 / 3.0, **kw)
    return pd_, r, pd_.tower_pressure_drop(path, 30.0, 30.0, 1.2, 30.0, 0.012, r.T_L_o, r.Y_o, P_MUC,
                                            saturation='kroeger')


def test_tower_pressure_drop_consistency():
    pd_, r, res = _tower_case()
    assert res.dp == pytest.approx(sum(res.dp_parts.values()))
    assert res.K_total == pytest.approx(sum(res.K.values()))
    # the fan -> plenum loss computed directly at the fan, VDI L1.3 Eq. (7), equals the converted value
    s_i = res.states['i']
    u_F = s_i.m_av / (s_i.rho * 10.0 / 3.0)
    assert res.dp_parts['fan -> plenum'] == pytest.approx((1 - 1 / 3)**2 * s_i.rho * u_F**2 / 2, rel=1e-12)
    # open outlet with the fill cross-section: exit loss close to one dynamic pressure of the fill
    assert res.K['exit'] == pytest.approx(1.0, abs=0.05)
    # plausible magnitudes: fill about 100 Pa at 1.2 m and G = 3 (Kloppers, Table T.9), total some 100-300 Pa
    assert 70.0 < res.dp_parts['fill'] < 140.0
    assert 100.0 < res.dp < 300.0
    assert pd_.fan_power(res) == pytest.approx(res.V_F * res.dp / 0.7)


def test_tower_pressure_drop_fan_inlet_sensitivity():
    pd_, _, res = _tower_case()
    _, _, res_sharp = _tower_case(zeta_inlet=pd_.ZETA_INLET_SHARP)
    assert res_sharp.K['fan inlet'] == pytest.approx(10.0 * res.K['fan inlet'])
    assert res_sharp.dp > res.dp
