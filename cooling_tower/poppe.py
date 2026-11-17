"""
poppe.py -- the Poppe method for a counterflow wet cooling tower (design mode).

Task (design): water cooled from T_W_i (top) to T_W_o (bottom) with the dry air flow m_dot_L, inlet air T_L_i, Y_i
at pressure p -> required Merkel number Me, outlet air state, evaporated water. The fill (fill.py) has to provide Me.

Equations, water temperature T_W as the independent variable.

Unsaturated air (Y <= Y_sat_L(T_L)) -- VDI-Wärmeatlas N4, Sec. 4.2 (identical in Kröger 2004, Eqs. 4.2.25, 4.2.26,
4.2.29, and Kloppers' dissertation, App. B, Eqs. B.24, B.25, B.30):

  Gl. (20)   dY/dT_W   = c_pW (m_dot_W/m_dot_L) (Y_sat_W - Y) / N
  Gl. (21)   dh_L/dT_W = c_pW (m_dot_W/m_dot_L) [1 + (Y_sat_W - Y) c_pW T_W / N]
  with the common denominator of (20) and (21)
             N = (h_L_sat_W - h_L) + (Le_f - 1) [(h_L_sat_W - h_L) - (Y_sat_W - Y)(Delta_h_V0 + c_pD T_W)]
                 - (Y_sat_W - Y) c_pW T_W
  Gl. (22)   NTU = integral beta_X dA / m_dot_L = integral (dY/dT_W) / (Y_sat_W - Y) dT_W,
             NTU = (m_dot_W/m_dot_L) Me
             The code integrates the Merkel number Me = integral beta_X dA / m_dot_W instead, with the LOCAL water
             flow inside the integral, so that the mass flows cancel (Kröger Eq. 4.2.29):
             dMe/dT_W = c_pW / N
             This Me is the definition of Kloppers' fill correlations (and of his App. G), so it is the quantity the
             fill has to provide; NTU is not needed (if ever: NTU ~ (m_dot_W_i/m_dot_L) Me).
  Gl. (24)   m_dot_W / m_dot_L = (m_dot_W2 - m_dot_L (Y_2 - Y)) / m_dot_L    (index 2 = top: water inlet m_dot_W_i,
             air outlet Y_o -- algebraically the same as the VDI form [1 - (M_L/M_W2)(X_2 - X)]^-1 M_W2/M_L)
  Gl. (25)   Le_f after Bosnjakovic (moist_air.Le_f)

Supersaturated air (Y > Y_sat_L(T_L), the excess water is mist) -- not in the VDI; Kröger Eqs. 4.2.18, 4.2.33-4.2.36
(pp. 242, 246 f.) = Kloppers Eqs. B.33, B.38, B.42, B.43, B.47 (pp. B.7 f.). Y is the total water content (vapour +
mist), Y_sat_L = Y_sat(T_L) the vapour the air can hold, h_L_ss the enthalpy with the mist as liquid:

  (Ü1)  h_L_ss = c_pL T_L + Y_sat_L (Delta_h_V0 + c_pD T_L) + (Y - Y_sat_L) c_pW T_L         (moist_air.h_L_ss)
  (Ü2)  Le_f with Y_sat_L instead of Y                                                      (moist_air.Le_f_supersat)
  (Ü3)  N_ss = (h_L_sat_W - h_L_ss) + (Le_f - 1) [(h_L_sat_W - h_L_ss) - (Y_sat_W - Y_sat_L)(Delta_h_V0 + c_pD T_W)
                                                  + (Y - Y_sat_L) c_pW T_W]
               + (Y - Y_sat_W) c_pW T_W
  (Ü4)  dY/dT_W      = c_pW (m_dot_W/m_dot_L) (Y_sat_W - Y_sat_L) / N_ss
  (Ü5)  dh_L_ss/dT_W = c_pW (m_dot_W/m_dot_L) [1 + (Y_sat_W - Y_sat_L) c_pW T_W / N_ss]
  (Ü6)  dMe/dT_W     = c_pW / N_ss
  Gl. (24) unchanged. For Y = Y_sat_L (saturated, no mist) (Ü1)-(Ü6) are identical with the unsaturated equations,
  so switching between the two sets is continuous.

Numerical procedure:
  1. Grid: N_int intervals between T_W_o (bottom, air inlet) and T_W_i (top, air outlet), 2 N_int + 1 points
     (interval ends and midpoints). Everything that depends only on T_W (c_p, Y_sat_W, h_D, h_L_sat_W) comes from
     moist_air.water_side_table, computed once per design call.
  2. Inner loop, integrate(Y_o): classic 4th-order Runge-Kutta from the bottom (Y_i, h_L_i, Me = 0) to the top.
     RK4 evaluates the derivatives at the start, twice at the midpoint and at the end of each interval -- hence the
     midpoints in the grid (Kloppers' dissertation, App. G, steps (n,1)-(n,4); Kröger App. B).
     The integrated enthalpy is h_L while the air is unsaturated and h_L_ss once it is supersaturated (Kloppers
     p. B.8: "i_ma must be replaced by i_ss"); both are the total enthalpy of the air stream.
  3. Switching: in EVERY Runge-Kutta stage air_state() decides from the current (h, Y) whether the air is
     unsaturated or supersaturated, and derivatives() uses the matching set of equations. (Kloppers checks only at
     the interval ends, App. G p. G.8; since the switch is continuous both give practically the same result.)
  4. Outer loop: the outlet humidity Y_o is needed in Gl. (24) but is a result. scipy brentq solves
     Y_top(Y_o) - Y_o = 0 in [Y_i, Y_sat_W(T_W_i)] (brentq as in comparison/operation._min_flow).
  5. After convergence: air temperature and state per level (air_state).

Not (yet) included:
  - validation switches for Merkel's limiting case (Le_f = 1, no evaporation loss) -- later.
  - rating mode (fill given, T_W_o unknown) -- not needed, design only.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from cooling_tower import moist_air as ma


# =============================================================================
# 1. Result
# =============================================================================

@dataclass
class PoppeResult:
    """Result of one Poppe design calculation. Profiles at the interval ends, from the bottom (index 0, air inlet,
    water outlet) to the top (index -1, air outlet, water inlet)."""
    Me: float                  # required Merkel number [-] (Kröger Eq. 4.2.29 = definition of the fill data)
    Y_o: float                 # outlet water content of the air (vapour + mist) [kg/kg dry air]
    h_L_o: float               # outlet enthalpy [J/kg dry air] (h_L or h_L_ss)
    T_L_o: float               # outlet air temperature [°C]
    m_dot_ev: float            # evaporated water m_dot_L (Y_o - Y_i) [kg/s]
    m_dot_W_o: float           # water leaving the fill at the bottom [kg/s], Gl. (24) at Y = Y_i
    supersaturated: bool       # True if the air is supersaturated (mist) anywhere in the fill
    n_intervals: int           # number of Runge-Kutta intervals used
    T_W: np.ndarray            # water temperature [°C]
    Y: np.ndarray              # water content of the air (vapour + mist) [kg/kg dry air]
    h_L: np.ndarray            # enthalpy of the air [J/kg dry air] (h_L or h_L_ss)
    T_L: np.ndarray            # air temperature [°C]
    Me_profile: np.ndarray     # Merkel number counted from the bottom [-]
    m_dot_W: np.ndarray        # water flow [kg/s]
    supersat_profile: np.ndarray  # True where Y > Y_sat_L(T_L)


class PoppeError(ValueError):
    """The design point cannot be computed with the Poppe equations (driving force <= 0 -- too little air -- or no
    root of the outer iteration)."""


# =============================================================================
# 2. Air state and derivatives
# =============================================================================

def saturation_function(saturation: str):
    """Saturation humidity ratio at the AIR temperature, same property model as Y_sat_W in the water-side table:
    'coolprop' -> moist_air.Y_sat_L, 'kroeger' -> moist_air.Y_sat_W_kroeger (the formula only depends on T and p)."""
    if saturation == 'coolprop':
        return ma.Y_sat_L
    if saturation == 'kroeger':
        return ma.Y_sat_W_kroeger
    raise ValueError(f"saturation must be 'coolprop' or 'kroeger', got {saturation!r}")


def air_state(h: float, Y: float, p: float, Y_sat_L_fn) -> tuple:
    """(T_L, Y_sat_L, supersaturated) of air with enthalpy h [J/kg dry air] and water content Y [kg/kg dry air].
    First T_L from the unsaturated formula; if the air could not hold Y as vapour at that temperature
    (Y > Y_sat_L(T_L)), it is supersaturated and T_L follows from the supersaturated enthalpy h_L_ss
    (Kloppers' criterion T_wb > T_a, App. G p. G.8, is the same test)."""
    T_L = ma.T_L_from_h_L(h, Y)
    Y_sat_L = Y_sat_L_fn(T_L, p)
    if Y <= Y_sat_L:
        return T_L, Y_sat_L, False
    T_L = ma.T_L_from_h_L_ss(h, Y, p, Y_sat_L_fn, T_lo=T_L)
    return T_L, Y_sat_L_fn(T_L, p), True


def derivatives(Y: float, h_L: float, i: int, tab: ma.WaterSideTable, m_dot_W_i: float, m_dot_L: float,
                Y_o: float, p: float, Y_sat_L_fn) -> tuple:
    """(dY/dT_W, dh_L/dT_W, dMe/dT_W) at grid point i of the water-side table, for the air state (Y, h_L) and the
    guessed outlet humidity Y_o. Unsaturated or supersaturated equations depending on the air state.
    Raises PoppeError if the driving force vanishes."""
    T_W = tab.T_W[i]
    c_pW = tab.c_pW[i]
    Y_sat_W = tab.Y_sat_W[i]
    dh_sat = tab.h_L_sat_W[i] - h_L            # h''_L(theta_W) - h_L  (or - h_L_ss)

    _, Y_sat_L, supersat = air_state(h_L, Y, p, Y_sat_L_fn)

    if not supersat:
        # unsaturated: VDI Gl. (20), (21), (25)
        dY_sat = Y_sat_W - Y                   # X''(theta_W) - X
        if dY_sat <= 0.0:
            raise PoppeError(f"Y_sat_W - Y = {dY_sat:.3e} <= 0 at T_W = {T_W:.3f} °C: the air is already as humid "
                             f"as air saturated at the water temperature -- too little air (increase m_dot_L)")
        Le = ma.Le_f(Y_sat_W, Y)                                                          # Gl. (25)
        N = dh_sat + (Le - 1.0) * (dh_sat - dY_sat * tab.h_D[i]) - dY_sat * c_pW * T_W     # denominator of (20), (21)
    else:
        # supersaturated: Kröger Eqs. 4.2.33-4.2.36 / Kloppers B.38, B.42, B.43, B.47
        dY_sat = Y_sat_W - Y_sat_L             # X''(theta_W) - X''(theta_L)
        if dY_sat <= 0.0:
            raise PoppeError(f"Y_sat_W - Y_sat_L = {dY_sat:.3e} <= 0 at T_W = {T_W:.3f} °C: the supersaturated air "
                             f"is as warm as the water -- too little air (increase m_dot_L)")
        Le = ma.Le_f_supersat(Y_sat_W, Y_sat_L)                                           # (Ü2)
        N = (dh_sat + (Le - 1.0) * (dh_sat - dY_sat * tab.h_D[i] + (Y - Y_sat_L) * c_pW * T_W)
             + (Y - Y_sat_W) * c_pW * T_W)                                                 # (Ü3)
    if N <= 0.0:
        raise PoppeError(f"denominator N = {N:.3e} <= 0 at T_W = {T_W:.3f} °C: no driving force left -- too little "
                         f"air (increase m_dot_L)")

    ratio = m_dot_W_i / m_dot_L - (Y_o - Y)                  # Gl. (24): m_dot_W / m_dot_L at this height
    dY = c_pW * ratio * dY_sat / N                           # Gl. (20) / (Ü4)
    dh = c_pW * ratio * (1.0 + dY_sat * c_pW * T_W / N)      # Gl. (21) / (Ü5)
    dMe = c_pW / N                                           # Kröger Eq. (4.2.29) / (Ü6)
    return dY, dh, dMe


# =============================================================================
# 3. Inner loop: Runge-Kutta 4th order for a given outlet humidity Y_o
# =============================================================================

def integrate(tab: ma.WaterSideTable, Y_i: float, h_L_i: float, m_dot_W_i: float, m_dot_L: float,
              Y_o: float, p: float, Y_sat_L_fn) -> tuple:
    """Integrates (Y, h_L, Me) from the bottom (grid point 0) to the top (grid point 2N) with classic RK4.
    tab must hold 2N + 1 equidistant points (ends and midpoints of N intervals).
    Returns the arrays Y, h_L, Me at the N + 1 interval ends."""
    n = (len(tab.T_W) - 1) // 2
    out = np.empty((n + 1, 3))                 # columns Y, h_L, Me
    out[0] = (Y_i, h_L_i, 0.0)

    args = (tab, m_dot_W_i, m_dot_L, Y_o, p, Y_sat_L_fn)
    for k in range(n):
        i0, im, i1 = 2 * k, 2 * k + 1, 2 * k + 2            # start, midpoint, end of interval k in the grid
        dT = tab.T_W[i1] - tab.T_W[i0]                       # interval size Delta T_W
        Y, h = out[k, 0], out[k, 1]
        # four RK4 stages; dT * stage = Kloppers' increments j (Y), k (h_L), l (Me)
        s1 = np.array(derivatives(Y, h, i0, *args))
        s2 = np.array(derivatives(Y + 0.5 * dT * s1[0], h + 0.5 * dT * s1[1], im, *args))
        s3 = np.array(derivatives(Y + 0.5 * dT * s2[0], h + 0.5 * dT * s2[1], im, *args))
        s4 = np.array(derivatives(Y + dT * s3[0], h + dT * s3[1], i1, *args))
        out[k + 1] = out[k] + dT / 6.0 * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
    return out[:, 0], out[:, 1], out[:, 2]


# =============================================================================
# 4. Grid
# =============================================================================

def n_intervals_for(T_W_i: float, T_W_o: float, n_intervals: int = 20, dT_interval: float | None = None) -> int:
    """Number of Runge-Kutta intervals: given directly (n_intervals), or from a maximum interval size dT_interval [K]
    (the smallest number of equal intervals that are not larger than dT_interval)."""
    if dT_interval is not None:
        if dT_interval <= 0.0:
            raise ValueError(f"dT_interval must be > 0, got {dT_interval}")
        return max(1, math.ceil((T_W_i - T_W_o) / dT_interval - 1e-9))
    if n_intervals < 1:
        raise ValueError(f"n_intervals must be >= 1, got {n_intervals}")
    return int(n_intervals)


def water_grid(T_W_i: float, T_W_o: float, n: int) -> np.ndarray:
    """2n + 1 equidistant water temperatures from T_W_o (bottom) to T_W_i (top)."""
    return np.linspace(T_W_o, T_W_i, 2 * n + 1)


# =============================================================================
# 5. Design: outer iteration on Y_o (brentq)
# =============================================================================

def design(T_W_i: float, T_W_o: float, m_dot_W_i: float, m_dot_L: float, T_L_i: float, Y_i: float, p: float,
           n_intervals: int = 20, dT_interval: float | None = None, saturation: str = 'coolprop',
           xtol: float = 1e-10) -> PoppeResult:
    """Poppe design calculation of a counterflow wet cooling tower (VDI N4, Sec. 4.2; supersaturated air after
    Kröger, Sec. 4.2).

    T_W_i, T_W_o   water inlet (top) and outlet (bottom) temperature [°C], T_W_i > T_W_o
    m_dot_W_i      water flow at the inlet [kg/s]
    m_dot_L        DRY air flow [kg/s]
    T_L_i, Y_i     inlet air temperature [°C] and humidity ratio [kg/kg dry air] (moist_air.Y_inlet)
    p              air pressure [Pa]
    n_intervals    number of Runge-Kutta intervals, or
    dT_interval    maximum interval size [K] (replaces n_intervals if given)
    saturation     'coolprop' (operation) or 'kroeger' (validation against Kloppers App. G); used for the saturation
                   at the water AND at the air temperature
    xtol           absolute tolerance of the outer iteration on Y_o [kg/kg]
    """
    if not T_W_i > T_W_o:
        raise ValueError(f"T_W_i = {T_W_i} must be larger than T_W_o = {T_W_o}")
    if m_dot_W_i <= 0.0 or m_dot_L <= 0.0:
        raise ValueError("mass flows must be positive")

    Y_sat_L_fn = saturation_function(saturation)
    n = n_intervals_for(T_W_i, T_W_o, n_intervals, dT_interval)
    tab = ma.water_side_table(water_grid(T_W_i, T_W_o, n), p, saturation=saturation)
    h_L_i = ma.h_L(T_L_i, Y_i)

    def residual(Y_o: float) -> float:
        return integrate(tab, Y_i, h_L_i, m_dot_W_i, m_dot_L, Y_o, p, Y_sat_L_fn)[0][-1] - Y_o

    # Bracket: Y_o = Y_i (no evaporation assumed) -> the integration still humidifies the air, residual > 0.
    # The air cannot leave more humid than air saturated at the water inlet temperature -> residual < 0 at
    # Y_o = Y_sat_W(T_W_i). The second bound also keeps m_dot_W > 0 in Gl. (24) (matters only for absurd L/G).
    Y_lo = Y_i
    Y_hi = min(float(tab.Y_sat_W[-1]), Y_i + 0.999 * m_dot_W_i / m_dot_L)
    r_lo, r_hi = residual(Y_lo), residual(Y_hi)
    if not (r_lo > 0.0 > r_hi):
        raise PoppeError(f"outer iteration not bracketed: residual(Y_o = {Y_lo:.5f}) = {r_lo:.3e}, "
                         f"residual(Y_o = {Y_hi:.5f}) = {r_hi:.3e}")
    Y_o = brentq(residual, Y_lo, Y_hi, xtol=xtol)

    # converged profiles at the interval ends
    Y, h, Me = integrate(tab, Y_i, h_L_i, m_dot_W_i, m_dot_L, Y_o, p, Y_sat_L_fn)
    T_W = tab.T_W[::2]
    m_dot_W = m_dot_W_i - m_dot_L * (Y_o - Y)                                     # Gl. (24)
    states = [air_state(float(hk), float(yk), p, Y_sat_L_fn) for hk, yk in zip(h, Y)]
    T_L = np.array([s[0] for s in states])
    supersat = np.array([s[2] for s in states])

    return PoppeResult(Me=float(Me[-1]), Y_o=float(Y_o), h_L_o=float(h[-1]), T_L_o=float(T_L[-1]),
                       m_dot_ev=float(m_dot_L * (Y_o - Y_i)), m_dot_W_o=float(m_dot_W[0]),
                       supersaturated=bool(supersat.any()), n_intervals=n, T_W=T_W, Y=Y, h_L=h, T_L=T_L,
                       Me_profile=Me, m_dot_W=m_dot_W, supersat_profile=supersat)
