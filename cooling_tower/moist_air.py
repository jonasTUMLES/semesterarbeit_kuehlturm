"""
moist_air.py -- Stoffwerte feuchter Luft und Wasser für die Poppe-Methode (moist-air and water properties).

Model: dry air and water vapour as ideal gases, enthalpies per kg DRY air with the reference state 0 °C for dry air
and liquid water (VDI-Wärmeatlas N4, Sec. 4.1, Eqs. 7-9 and text; Kröger 2004, Vol. 1, Sec. 4.2 and App. A). The
Poppe equations (poppe.py) are derived with exactly this model (Kröger Eqs. 4.2.4, 4.2.10, 4.2.11), so all
enthalpies come from these formulas -- one model, one reference state, the same numbers as the worked example of
Kloppers' dissertation (App. G), which is the validation case. The specific heats are Kröger's temperature-dependent
polynomials (App. A) instead of the VDI's constant mean values.

CoolProp is used only where it has a real advantage:
  - Y_sat_W (saturation at the water temperature) in operation -- with the station pressure of the hour
  - Y_i (inlet air from T, phi, p of the weather data) -- also below 0 °C, no detour via the wet bulb
  - Y_sat_L (saturation at the AIR temperature) for the supersaturation check
Kröger's equation for Y_sat_W is kept for the validation against Kloppers (App. G gives the inlet state w_i
directly, so no inlet-humidity formula of Kröger is needed).

Units: temperatures in °C in the interface (as the rest of the repository); Kröger's polynomials and CoolProp take
Kelvin and are converted inside. Pressures in Pa, enthalpies in J/kg dry air, humidity ratios in kg water/kg dry air.

Notation -- VDI-Wärmeatlas N4 letters (L Luft, D Dampf, W Wasser) with the repository's conventions (T instead of
theta, Y for the humidity ratio as in VDI M8 and src/thermo/precooling.py, state indices i/o):

  code          VDI N4                 Kröger      meaning
  T_W           theta_W                T_w         water temperature
  T_L           theta_L                T_a         air temperature
  Y             X                      w           humidity ratio of the air
  Y_sat_W       X''(theta_W)           w_sw        saturation humidity ratio at T_W
  Y_sat_L       X''(theta_L)           w_sa        saturation humidity ratio at T_L
  p_D_sat       p''_D                  p_v         saturation vapour pressure
  h_L           h_L                    i_ma        enthalpy of humid air
  h_L_sat_W     h''_L(theta_W)         i_masw      enthalpy of air saturated at T_W
  h_D           h_D                    i_v         enthalpy of water vapour at T_W
  h_L_ss        --                     i_ss        enthalpy of supersaturated air (vapour + mist)
  Delta_h_V0    Delta h_V0             i_fgwo      latent heat at 0 °C
  c_pL, c_pD, c_pW   c_pL, c_pD, c_pW  c_pa, c_pv, c_pw   specific heats of dry air, vapour, liquid water
  Le_f          alpha/(beta_X c_pLD)   Le_f        Lewis factor
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import brentq

from src.thermo.fluid_properties import relative_to_absolute_humidity

T0_K = 273.15                  # reference temperature 0 °C [K]


# =============================================================================
# 1. Specific heats and latent heat (Kröger 2004, Vol. 1, App. A; T in K)
# =============================================================================

def c_pL_kroeger(T_K: float) -> float:
    """Specific heat of dry air [J/(kg K)], Kröger Eq. (A.1.2), p. 441 (valid from 220-380 K)."""
    return 1.045356e3 - 3.161783e-1 * T_K + 7.083814e-4 * T_K**2 - 2.705209e-7 * T_K**3


def c_pD_kroeger(T_K: float) -> float:
    """Specific heat of water vapour [J/(kg K)], Kröger Eq. (A.2.2), p. 444 (valid from 273.15-380 K)."""
    return 1.3605e3 + 2.31334 * T_K - 2.46784e-10 * T_K**5 + 5.91332e-13 * T_K**6


def c_pW_kroeger(T_K: float) -> float:
    """Specific heat of liquid water [J/(kg K)], Kröger Eq. (A.4.2), p. 449 (valid from 273.15-380 K)."""
    return 8.15599e3 - 2.80627e1 * T_K + 5.11283e-2 * T_K**2 - 2.17582e-13 * T_K**6


def Delta_h_V_kroeger(T_K: float) -> float:
    """Latent heat of vaporisation of water [J/kg], Kröger Eq. (A.4.5), p. 449."""
    return 3.4831814e6 - 5.8627703e3 * T_K + 12.139568 * T_K**2 - 1.40290431e-2 * T_K**3


# Latent heat at 0 °C (VDI Delta h_V0 = 2500 kJ/kg, rounded) -- a constant: Kröger p. 448 under Eq. (A.3.6b),
# "i_fgwo is evaluated at 273.15 K according to equation (A.4.5), i.e., i_fgwo = 2.5016 x 10^6 J/kg" (= 2 501 598.5)
DELTA_H_V0 = Delta_h_V_kroeger(T0_K)


def T_eval_K(T: float) -> float:
    """Evaluation temperature of the specific heats [K] for a state at T [°C]: the middle between 0 °C and T,
    (T_K + 273.15) / 2 (Kröger p. 238 under Eq. 4.2.4 and p. 448 under Eq. A.3.6b for c_pL, c_pD; Kloppers
    dissertation App. G, pp. G.2 ff., also for c_pW). For a c_p that is nearly linear in T this is the integral mean
    between 0 °C and T, so that c_p T is the enthalpy with the reference 0 °C."""
    return T / 2.0 + T0_K


def specific_heats(T: float) -> tuple:
    """(c_pL, c_pD, c_pW) [J/(kg K)] of a state at T [°C], all at T_eval_K(T)."""
    T_K = T_eval_K(T)
    return c_pL_kroeger(T_K), c_pD_kroeger(T_K), c_pW_kroeger(T_K)


# =============================================================================
# 2. Humidity ratios
# =============================================================================

def p_D_sat_kroeger(T: float) -> float:
    """Saturation vapour pressure over liquid water p''_D [Pa] at T [°C], Goff, Kröger Eq. (A.2.1), p. 444
    (VDI N4 Eq. 5 is an alternative, 0-70 °C). Valid 273.15-380 K -- for the WATER temperature always fine; not for
    air below 0 °C."""
    _check_liquid_range(T, "p_D_sat_kroeger")
    T_K = T + T0_K
    z = (10.79586 * (1.0 - 273.16 / T_K) + 5.02808 * math.log10(273.16 / T_K)
         + 1.50474e-4 * (1.0 - 10.0**(-8.29692 * (T_K / 273.16 - 1.0)))
         + 4.2873e-4 * (10.0**(4.76955 * (1.0 - 273.16 / T_K)) - 1.0) + 2.786118312)
    return 10.0**z


def Y_sat_W_kroeger(T_W: float, p: float) -> float:
    """Saturation humidity ratio X''(theta_W) [kg/kg dry air] at the water temperature T_W [°C]: Kröger Eq. (A.3.5)
    with T = T_wb = T_W, i.e. 0.62509 p''_D / (p - 1.005 p''_D) with p''_D from (A.2.1) -- the form used in Kloppers'
    dissertation, App. G, p. G.3. For the validation and as a cross-check of the CoolProp value."""
    p_D = p_D_sat_kroeger(T_W)
    return 0.62509 * p_D / (p - 1.005 * p_D)


def Y_sat_W_coolprop(T_W: float, p: float) -> float:
    """Saturation humidity ratio X''(theta_W) [kg/kg dry air] at the water temperature T_W [°C] and pressure p [Pa],
    CoolProp HAPropsSI (real humid air, Bell et al. 2014). The value used in operation."""
    return CP.HAPropsSI('W', 'T', T_W + T0_K, 'P', p, 'R', 1.0)


def Y_inlet(T_L: float, phi: float, p: float) -> float:
    """Humidity ratio of the inlet air [kg/kg dry air] from the weather data (T_L [°C], phi [0-1], p [Pa]), via the
    predecessor's CoolProp helper (src/thermo/fluid_properties.relative_to_absolute_humidity) -- the same value the
    dry and hybrid coolers get, valid also below 0 °C."""
    return relative_to_absolute_humidity(T_L, p, phi)


def Y_sat_L(T_L: float, p: float) -> float:
    """Saturation humidity ratio X''(theta_L) [kg/kg dry air] at the AIR temperature T_L [°C], CoolProp. Below 0 °C
    CoolProp should refer the saturation to ice -- NOT YET CHECKED against a table (open point,
    docs/cooling-tower.md)."""
    return CP.HAPropsSI('W', 'T', T_L + T0_K, 'P', p, 'R', 1.0)


def is_supersaturated(T_L: float, Y: float, p: float) -> bool:
    """True if the air holds more water than saturated air at its own temperature (mist; Kröger p. 241 f.)."""
    return Y > Y_sat_L(T_L, p)


# =============================================================================
# 3. Enthalpies (per kg dry air, temperatures in °C, reference 0 °C)
# =============================================================================

def h_L(T_L: float, Y: float) -> float:
    """Enthalpy of humid air [J/kg dry air], VDI N4 Eq. (7) = Kröger Eq. (4.2.4), p. 238 / (A.3.6b), p. 448:
    c_pL T_L + Y (Delta_h_V0 + c_pD T_L), c_pL and c_pD at T_eval_K(T_L). Needed as a formula only at the inlet
    (start value) and for the inversion T_L_from_h_L; during the integration h_L follows from Kröger Eq. (4.2.26)."""
    c_pL, c_pD, _ = specific_heats(T_L)
    return c_pL * T_L + Y * (DELTA_H_V0 + c_pD * T_L)


def h_D(T_W: float) -> float:
    """Enthalpy of water vapour at the water temperature [J/kg], VDI N4 text to Eq. (8) (h_D = Delta_h_V0 + c_pD
    theta) = Kröger p. 239 under Eq. (4.2.8); c_pD at T_eval_K(T_W)."""
    _, c_pD, _ = specific_heats(T_W)
    return DELTA_H_V0 + c_pD * T_W


def h_L_sat_W(T_W: float, Y_sat_W: float) -> float:
    """Enthalpy of air saturated at the water temperature h''_L(theta_W) [J/kg dry air], VDI N4 Eq. (8) =
    Kröger Eq. (4.2.10), p. 240: c_pL T_W + Y_sat_W (Delta_h_V0 + c_pD T_W), specific heats at T_eval_K(T_W)."""
    c_pL, c_pD, _ = specific_heats(T_W)
    return c_pL * T_W + Y_sat_W * (DELTA_H_V0 + c_pD * T_W)


def h_L_ss(T_L: float, Y: float, Y_sat_L_: float) -> float:
    """Enthalpy of supersaturated air (saturated vapour + liquid mist) [J/kg dry air], Kröger Eq. (4.2.18), p. 242:
    c_pL T_L + Y_sat_L (Delta_h_V0 + c_pD T_L) + (Y - Y_sat_L) c_pW T_L, specific heats at T_eval_K(T_L).
    For the later supersaturated branch of poppe.py (Kröger Eqs. 4.2.34-4.2.36)."""
    c_pL, c_pD, c_pW = specific_heats(T_L)
    return c_pL * T_L + Y_sat_L_ * (DELTA_H_V0 + c_pD * T_L) + (Y - Y_sat_L_) * c_pW * T_L


def T_L_from_h_L(h: float, Y: float, T_lo: float = -40.0, T_hi: float = 60.0) -> float:
    """Air temperature [°C] of unsaturated air with enthalpy h [J/kg dry air] and humidity ratio Y: h_L solved for
    T_L (iteratively, c_pL and c_pD depend on T; Kloppers dissertation App. G, p. G.7, does the same by trial).
    Needed for the outlet air temperature and the supersaturation check.
    Search interval [T_lo, T_hi]: the air in the tower lies between the coldest ambient air (Munich climate year
    -10.7 °C, climate_data/munich_past_year_design_points.csv) and the water inlet temperature (at most 50 °C,
    Rubio-Castro et al. 2011, Eq. 33); -40 ... 60 °C encloses this range with margin. h_L rises monotonically with T,
    so brentq finds the single root if h lies inside the interval."""
    return brentq(lambda T: h_L(T, Y) - h, T_lo, T_hi, xtol=1e-6)


# =============================================================================
# 4. Lewis factor (Bosnjakovic)
# =============================================================================

LE_FACTOR = 0.865**(2.0 / 3.0)  # VDI N4 Eq. (25): 0.865^(2/3) = 0.9079 (Kloppers Eq. B.15 writes 0.865^0.667;
                                 # Kröger Eq. 4.2.13 uses 0.866^0.667, about 0.08 % higher)


def _bosnjakovic(Y_s: float, Y: float) -> float:
    """0.865^(2/3) (a - 1) / ln(a) with a = (0.622 + Y_s)/(0.622 + Y)."""
    a = (0.622 + Y_s) / (0.622 + Y)
    if abs(a - 1.0) < 1e-12:            # limit (a - 1) / ln(a) -> 1 for Y -> Y_s (avoids 0/0)
        return LE_FACTOR
    return LE_FACTOR * (a - 1.0) / math.log(a)


def Le_f(Y_sat_W: float, Y: float) -> float:
    """Lewis factor of unsaturated air after Bosnjakovic, VDI-Wärmeatlas N4 Eq. (25):
    alpha / (beta_X c_pLD) = 0.865^(2/3) [(0.622 + X''(theta_W))/(0.622 + X) - 1] / ln[(0.622 + X''(theta_W))/(0.622 + X)].
    Gives the ratio directly -- alpha and beta are never needed on their own."""
    return _bosnjakovic(Y_sat_W, Y)


def Le_f_supersat(Y_sat_W: float, Y_sat_L_: float) -> float:
    """Lewis factor of supersaturated air: Eq. (25) with the saturation humidity ratio at the air temperature in place
    of X (only the vapour counts, not the mist), as in Kröger Eq. (4.2.33), p. 246 -- with the VDI factor for
    consistency with Le_f (the VDI gives no equation for supersaturated air)."""
    return _bosnjakovic(Y_sat_W, Y_sat_L_)


# =============================================================================
# 5. Water-side table: everything that depends only on the water temperature
# =============================================================================

@dataclass
class WaterSideTable:
    """Properties at the water temperatures of one integration (2N + 1 points for N Runge-Kutta intervals:
    the interval ends and midpoints). They depend only on T_W and p, so they are computed ONCE per guess of the
    water outlet temperature and reused by every iteration on the outlet humidity Y_o (poppe.py)."""
    T_W: np.ndarray            # water temperature [°C]
    c_pL: np.ndarray           # [J/(kg K)], all specific heats at T_eval_K(T_W)
    c_pD: np.ndarray
    c_pW: np.ndarray
    Y_sat_W: np.ndarray        # saturation humidity ratio at T_W [kg/kg dry air]
    h_D: np.ndarray            # enthalpy of water vapour at T_W [J/kg]
    h_L_sat_W: np.ndarray      # enthalpy of saturated air at T_W [J/kg dry air]


def water_side_table(T_W_grid, p: float, saturation: str = 'coolprop') -> WaterSideTable:
    """Builds the WaterSideTable for the water temperatures T_W_grid [°C] at pressure p [Pa].
    saturation: 'coolprop' (operation) or 'kroeger' (validation against Kloppers, App. G)."""
    if saturation == 'coolprop':
        Y_sat_W_fn = Y_sat_W_coolprop
    elif saturation == 'kroeger':
        Y_sat_W_fn = Y_sat_W_kroeger
    else:
        raise ValueError(f"saturation must be 'coolprop' or 'kroeger', got {saturation!r}")

    T_W = np.asarray(T_W_grid, dtype=float)
    for T in (T_W.min(), T_W.max()):
        _check_liquid_range(float(T), "water_side_table")
    # every value comes from the scalar functions above, so each formula is defined only once
    c_pL, c_pD, c_pW = (np.array(v) for v in zip(*(specific_heats(float(T)) for T in T_W)))
    Y_sat_W = np.array([Y_sat_W_fn(float(T), p) for T in T_W])
    h_D_arr = np.array([h_D(float(T)) for T in T_W])
    h_L_sat_W_arr = np.array([h_L_sat_W(float(T), float(Y)) for T, Y in zip(T_W, Y_sat_W)])
    return WaterSideTable(T_W=T_W, c_pL=c_pL, c_pD=c_pD, c_pW=c_pW, Y_sat_W=Y_sat_W, h_D=h_D_arr,
                          h_L_sat_W=h_L_sat_W_arr)


# =============================================================================
# 6. Helpers
# =============================================================================

def _check_liquid_range(T: float, where: str) -> None:
    """Kröger's water and vapour equations (A.2, A.4) are given for 273.15-380 K."""
    if not 0.0 <= T <= 380.0 - T0_K:
        warnings.warn(f"{where}: T = {T:.2f} °C outside the range 0-106.85 °C of Kröger App. A.2/A.4",
                      stacklevel=3)
