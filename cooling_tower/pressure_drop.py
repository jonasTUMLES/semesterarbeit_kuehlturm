"""
pressure_drop.py -- air-side pressure drop of the forced-draft counterflow tower (fan at the bottom, blowing).

Framework: VDI-Wärmeatlas N4, Sec. 6.3/6.4 (pp. 1951 f.): loss coefficient K = Delta_p / (rho v^2/2) (Gl. 36); the
component losses are referred to the mean velocity in the fill and added, K_tot = sum K_i (Gl. 40);
Delta_p = K_tot rho_fill v_fill^2 / 2 (Gl. 41); the static pressure rise of the fan balances Delta_p (Sec. 6.4).

Reference state = mean conditions in the fill (Kloppers, dissertation, App. D):
  m_av,i = m_L (1 + Y_i) below, m_av,o = m_L (1 + Y_o) above the fill (humid air, incl. mist); G_av = m_av / A_fr
  G_av,m = (G_av,i + G_av,o)/2,  1/rho_av,m = 0.5 (1/rho_av,i + 1/rho_av,o)
Conversion of a coefficient K given for (rho, m_av, A) to the fill, Kloppers Eq. (D.5) -- the exact form of the VDI
conversion (Sec. 6.3), also with the changes of density and mass flow:
  K_fi = K (rho_av,m / rho) (m_av / m_av,m)^2 (A_fr / A)^2

Components (VDI 6.3, items 1-9), in the order the air passes them:
  1a fan inlet            zeta_E = 0.05 (rounded inlet), 0.5 (sharp-edged)   VDI L1.3, Sec. 1.3, Fig. 4 (p. 1365)
                          referred to the fan (inlet air, fan area A_F)
  1b fan -> plenum        (1 - A_F/A_plenum)^2 (Borda-Carnot)               VDI L1.3, Eq. (7) = Kloppers (D.16)
                          referred to the fan
  2  rain zone            neglected: VDI N4 6.2 (forced draft, film fill); range of Kloppers (D.9) (H_i 2-8 m) not met
  3  fill supports        K_fs = 0.5 (incl. contraction)                     Kloppers App. J (after Kröger)
                          referred to the state below the fill
  4  fill                 K_fdm = K_fdm1 L_fi, K_fdm1 from Kloppers (T.4), p. T.13 (film fill, three heights)
                          + momentum, Kloppers (D.19): K_fi = K_fdm + (G_av,o^2/rho_o - G_av,i^2/rho_i)/(G_av,m^2/rho_m)
                          already referred to the mean fill state (definition of K_fdm: Kloppers Eqs. 3.2-3.4)
                          buoyancy neglected (forced draft, as in Kloppers App. J)
     expansion after fill (D.16) with sigma = A_fr/A_fr = 1 -> 0 (same cross-section above the fill)
  5  spray zone           K_sp = L_sp [0.4 (G_W/G_L) + 1]                    Kloppers (D.6) (after Cale)
     water distribution   K_wd = 0.5                                         Kloppers D.2.3 (after Kröger)
                          both referred to the state above the fill
  6+7 drift eliminator    K_de = 27.4892 Ry^-0.14247, Ry = m_av / (mu_av A_fr)  Kloppers (D.7), type c: App. I, p. I.23
                          referred to the state above the fill
  8  outlet shell         none (open outlet above the drift eliminator assumed)
  9  exit (dynamic pressure)  K = 1 referred to the outlet: limit of Borda-Carnot for A_out/A_ambient -> 0 (VDI L1.3
                          Eq. 7 / Kloppers D.16); VDI N4 6.3 item 9

Not used: Kloppers (D.10)-(D.14) (inlet of towers with an open inlet around the circumference: round towers,
d_i = 80-120 m, resp. induced-draft rectangular towers), (D.15) (supports of a tower shell), (D.8)/(D.9) (rain zone).

Properties: density and viscosity of humid air after Kröger App. A (moist_air.rho_av, mu_av), as in Kloppers'
examples. Supersaturated outlet air (mist): the gas phase holds Y_sat(T_L), the mist adds mass but no volume,
rho = rho_av(T_L, Y_sat) (1 + Y)/(1 + Y_sat) (own assumption, consistent with G_av incl. mist).

Fan power: P_F = V_F Delta_p / eta_F with the volume flow at the fan V_F = m_av,i / rho_av,i (cold inlet air) and
eta_F = 0.7 as in the predecessor's economics.calc_fan_power (Towler, Chemical Engineering Design, Sec. 19.16).
"""

import warnings
from dataclasses import dataclass, field

from cooling_tower import moist_air as ma
from cooling_tower import poppe as pp


# =============================================================================
# 1. Conversion to the mean fill state (Kloppers D.5)
# =============================================================================

def refer_to_fill(K: float, rho: float, m_av: float, A: float, rho_m: float, m_av_m: float, A_fr: float) -> float:
    """Loss coefficient K given for (rho, m_av, A) referred to the mean fill state (rho_m, m_av_m, A_fr),
    Kloppers Eq. (D.5): K_fi = K (rho_m/rho) (m_av/m_av_m)^2 (A_fr/A)^2. The pressure drop stays the same."""
    return K * (rho_m / rho) * (m_av / m_av_m)**2 * (A_fr / A)**2


# =============================================================================
# 2. Loss coefficients of the components (each in its own reference state)
# =============================================================================

ZETA_INLET_ROUNDED = 0.05       # VDI L1.3, Fig. 4: rounded inlet, 'normal'
ZETA_INLET_SHARP = 0.5          # VDI L1.3, Fig. 4: sharp-edged inlet, 'normal gebrochen'
K_FILL_SUPPORT = 0.5            # Kloppers App. J: fill support and contraction
K_WATER_DISTRIBUTION = 0.5      # Kloppers D.2.3
K_EXIT = 1.0                    # dynamic pressure of the leaving air (VDI N4 6.3, item 9)
DRIFT_ELIMINATOR_TYPE_C = (27.4892, -0.14247)   # Kloppers (D.7) with the values of type c, App. I p. I.23

# range of Kloppers' fill tests (App. T), the basis of (T.4)
T4_RANGE = {'G_W': (2.687, 6.831), 'G_L': (1.181, 4.186), 'L_fi': (0.6, 1.2)}


def K_expansion(A_small: float, A_large: float) -> float:
    """Sudden expansion (Borda-Carnot), referred to the velocity in the SMALL cross-section:
    K = (1 - A_small/A_large)^2 -- VDI L1.3 Eq. (7), Kloppers (D.16)."""
    if A_small > A_large:
        raise ValueError(f"A_small = {A_small} larger than A_large = {A_large}")
    return (1.0 - A_small / A_large)**2


def K_fdm1_kloppers(G_W: float, G_L: float, L_fi: float) -> float:
    """Fill loss coefficient per metre [1/m] (friction and drag, without momentum and buoyancy), Kloppers (T.4),
    film fill: K_fdm1 = (5.154914 G_W^0.877646 G_L^-1.462034 + 10.806728 G_W^0.226578 G_L^0.293222) L_fi^-0.236292.
    G_W = m_W/A_fr, G_L = m_L/A_fr (dry air) [kg/(m^2 s)]. r^2 = 0.904."""
    for value, what in ((G_W, 'G_W'), (G_L, 'G_L'), (L_fi, 'L_fi')):
        lo, hi = T4_RANGE[what]
        if not lo <= value <= hi:
            warnings.warn(f"fill loss coefficient (Kloppers T.4): {what} = {value:.3f} outside the tested range "
                          f"{lo}-{hi} (extrapolation)", stacklevel=3)
    return ((5.154914 * G_W**0.877646 * G_L**-1.462034 + 10.806728 * G_W**0.226578 * G_L**0.293222)
            * L_fi**-0.236292)


def K_fill(K_fdm: float, G_av_i: float, rho_i: float, G_av_o: float, rho_o: float) -> float:
    """Fill loss coefficient in the tower, with the momentum change of the air, Kloppers (D.19), referred to the mean
    fill state: K_fi = K_fdm + (G_av,o^2/rho_o - G_av,i^2/rho_i) / (G_av,m^2/rho_m)."""
    G_m = 0.5 * (G_av_i + G_av_o)
    rho_m = 2.0 / (1.0 / rho_i + 1.0 / rho_o)
    return K_fdm + (G_av_o**2 / rho_o - G_av_i**2 / rho_i) / (G_m**2 / rho_m)


def K_spray_zone(L_sp: float, G_W: float, G_L: float) -> float:
    """Spray zone above the fill, Kloppers (D.6) (after Cale): K_sp = L_sp [0.4 (G_W/G_L) + 1]."""
    return L_sp * (0.4 * G_W / G_L + 1.0)


def K_drift_eliminator(m_av: float, mu_av: float, A_fr: float, coeff: tuple = DRIFT_ELIMINATOR_TYPE_C) -> float:
    """Drift eliminator, Kloppers (D.7): K_de = a Ry^b with Ry = m_av / (mu_av A_fr) [1/m]; default type c."""
    a, b = coeff
    return a * (m_av / (mu_av * A_fr))**b


# =============================================================================
# 3. Air states and tower geometry
# =============================================================================

@dataclass(frozen=True)
class AirState:
    """Humid air at one cross-section: m_av incl. mist [kg/s], rho [kg/m^3], mu [kg/(m s)]."""
    m_av: float
    rho: float
    mu: float


def air_state(m_dot_L: float, T_L: float, Y: float, p: float, Y_sat_fn) -> AirState:
    """Mass flow, density and viscosity of humid air with water content Y (vapour + mist). If Y exceeds the
    saturation humidity at T_L, the gas holds Y_sat and the mist adds mass but (practically) no volume."""
    Y_v = min(Y, Y_sat_fn(T_L, p))
    rho = ma.rho_av(T_L, Y_v, p) * (1.0 + Y) / (1.0 + Y_v)
    return AirState(m_av=m_dot_L * (1.0 + Y), rho=rho, mu=ma.mu_av(T_L, Y_v))


@dataclass(frozen=True)
class FlowPath:
    """Air path of the tower. Areas in m^2, heights in m.
    A_plenum and A_out default to A_fr (plenum and open outlet with the cross-section of the fill).
    L_sp = 0.15 m: the spray zone height of Kloppers' fill tests, whose Merkel numbers (fill.py) include it."""
    A_fr: float                               # frontal area of the fill
    A_F: float                                # fan area (pi d_F^2 / 4)
    A_plenum: float | None = None             # cross-section of the plenum below the fill
    A_out: float | None = None                # outlet cross-section
    L_sp: float = 0.15                        # height of the spray zone above the fill
    zeta_inlet: float = ZETA_INLET_ROUNDED    # fan inlet (rounded 0.05 / sharp-edged 0.5)
    K_fs: float = K_FILL_SUPPORT
    K_wd: float = K_WATER_DISTRIBUTION
    drift_eliminator: tuple = DRIFT_ELIMINATOR_TYPE_C

    @property
    def plenum(self) -> float:
        return self.A_fr if self.A_plenum is None else self.A_plenum

    @property
    def outlet(self) -> float:
        return self.A_fr if self.A_out is None else self.A_out


# =============================================================================
# 4. Pressure drop of the tower
# =============================================================================

@dataclass
class PressureDropResult:
    dp: float                  # total pressure drop the fan has to provide [Pa] (VDI Gl. 41)
    K_total: float             # sum of the loss coefficients referred to the mean fill state (VDI Gl. 40)
    K: dict                    # loss coefficients per component, referred to the mean fill state
    dp_parts: dict             # pressure drop per component [Pa]
    V_F: float                 # volume flow through the fan [m^3/s] (inlet air)
    v_fill: float              # mean velocity in the fill [m/s]
    rho_m: float               # mean density in the fill [kg/m^3]
    states: dict = field(default_factory=dict)   # AirState below ('i') and above ('o') the fill


def tower_pressure_drop(path: FlowPath, m_dot_W_i: float, m_dot_L: float, L_fi: float,
                        T_L_i: float, Y_i: float, T_L_o: float, Y_o: float, p: float,
                        saturation: str = 'coolprop') -> PressureDropResult:
    """Pressure drop of the forced-draft tower for the air states below (inlet, T_L_i, Y_i) and above the fill
    (outlet of poppe.design, T_L_o, Y_o). The fan and the plenum carry the inlet air (heating by the fan neglected).
    m_dot_W_i: water inlet flow (G_W as in Kloppers' tests), m_dot_L: dry air flow."""
    Y_sat_fn = pp.saturation_function(saturation)
    s_i = air_state(m_dot_L, T_L_i, Y_i, p, Y_sat_fn)
    s_o = air_state(m_dot_L, T_L_o, Y_o, p, Y_sat_fn)
    A_fr = path.A_fr
    m_m = 0.5 * (s_i.m_av + s_o.m_av)
    rho_m = 2.0 / (1.0 / s_i.rho + 1.0 / s_o.rho)
    G_W, G_L = m_dot_W_i / A_fr, m_dot_L / A_fr

    def ref(K, s, A):
        return refer_to_fill(K, s.rho, s.m_av, A, rho_m, m_m, A_fr)

    K = {
        'fan inlet': ref(path.zeta_inlet, s_i, path.A_F),                                  # 1a
        'fan -> plenum': ref(K_expansion(path.A_F, path.plenum), s_i, path.A_F),           # 1b
        'fill supports': ref(path.K_fs, s_i, A_fr),                                        # 3
        'fill': K_fill(K_fdm1_kloppers(G_W, G_L, L_fi) * L_fi,                             # 4
                       s_i.m_av / A_fr, s_i.rho, s_o.m_av / A_fr, s_o.rho),
        'spray zone': ref(K_spray_zone(path.L_sp, G_W, G_L), s_o, A_fr),                   # 5
        'water distribution': ref(path.K_wd, s_o, A_fr),                                   # 5
        'drift eliminator': ref(K_drift_eliminator(s_o.m_av, s_o.mu, A_fr, path.drift_eliminator), s_o, A_fr),
        'exit': ref(K_EXIT, s_o, path.outlet),                                             # 9
    }
    dyn = (m_m / A_fr)**2 / (2.0 * rho_m)            # dynamic pressure in the fill, G_av,m^2 / (2 rho_m)
    K_total = sum(K.values())
    return PressureDropResult(dp=K_total * dyn, K_total=K_total, K=K,
                              dp_parts={k: v * dyn for k, v in K.items()},
                              V_F=s_i.m_av / s_i.rho, v_fill=m_m / (rho_m * A_fr), rho_m=rho_m,
                              states={'i': s_i, 'o': s_o})


ETA_FAN = 0.7    # as in src/auxiliaries/economics.calc_fan_power (Towler, Chemical Engineering Design, Sec. 19.16)


def fan_power(result: PressureDropResult, eta_F: float = ETA_FAN) -> float:
    """Fan power [W]: P_F = V_F Delta_p / eta_F."""
    return result.V_F * result.dp / eta_F
