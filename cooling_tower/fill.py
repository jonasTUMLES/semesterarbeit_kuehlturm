"""
fill.py -- fill performance characteristic: Merkel number per metre of fill, evaluated with POPPE.

Form (Kloppers, dissertation, Eq. T.1; Kloppers & Kröger 2005, Heat Transfer Eng. 26(4); VDI N4 Gl. 35 without the
temperature term):

    Me / L_fi = c1 * G_W^c2 * G_L^c3 * L_fi^c4          G_W = m_dot_W / A_fr,  G_L = m_dot_L / A_fr  [kg/(m^2 s)]

    (G_L with the DRY air flow, as in Kloppers' data; L_fi in m; Me = Merkel number of the whole fill incl. the
    0.15 m spray zone of the test rig.)

Data base: Kloppers' 54 tests on a cross-corrugated film fill (dissertation, App. T, three fill heights), each
re-evaluated with poppe.design and CoolProp (fill_evaluation.py -> data/kloppers2003_film_fill_poppe.csv, column
Me_P_calc). Fill data and tower design therefore come from the same code and property model (consistency rule,
Kloppers & Kröger 2005). Kloppers' own published film correlation (Eq. T.1 = Kloppers & Kröger 2005, HTE, used e.g.
by Rubio-Castro et al. 2011, Tab. 1) is MERKEL-evaluated and must not be combined with poppe.design.

Fit: least squares on ln(Me/L_fi) = ln c1 + c2 ln G_W + c3 ln G_L + c4 ln L_fi, i.e. the RELATIVE deviation is
minimised (every test weighs the same, whether Me/L_fi is 0.6 or 2.4 1/m).

No term for the water inlet temperature: with this data it is not identifiable (exponent +0.32 +- 0.33, T_wi
correlated with L_fi and G_W at r = -0.73 / -0.55, no improvement of the fit), and Kloppers & Kröger (2005) publish
the film correlation without it as well (docs/cooling-tower.md).

Validity range = range of the tests: G_W 2.687-6.831, G_L 1.181-4.186 kg/(m^2 s), L_fi 0.6-1.2 m, water inlet
37.3-47.6 °C. Outside, check_range() warns (extrapolation).

Use in the design: the required Merkel number from poppe.design and the fill must agree,
    Me_req = c1 G_W^c2 G_L^c3 L_fi^(1 + c4)   ->   L_fi = (Me_req / (c1 G_W^c2 G_L^c3))^(1 / (1 + c4))
(fill_height(); closed form, no iteration needed).

Planned here as well: loss coefficient of the fill K_fi(G_W, G_L, L_fi) (Kloppers App. T, Eq. T.4) for pressure_drop.py.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_POPPE = Path(__file__).parent / 'data' / 'kloppers2003_film_fill_poppe.csv'


# =============================================================================
# 1. Correlation
# =============================================================================

@dataclass(frozen=True)
class FillCorrelation:
    """Me / L_fi = c1 G_W^c2 G_L^c3 L_fi^c4 with its validity range."""
    c1: float
    c2: float                  # exponent of G_W
    c3: float                  # exponent of G_L
    c4: float                  # exponent of L_fi
    G_W_range: tuple           # (min, max) [kg/(m^2 s)]
    G_L_range: tuple           # (min, max) [kg/(m^2 s)]
    L_fi_range: tuple          # (min, max) [m]
    name: str = ''

    def Me_per_m(self, G_W: float, G_L: float, L_fi: float) -> float:
        """Merkel number per metre of fill [1/m]."""
        self.check_range(G_W, G_L, L_fi)
        return self.c1 * G_W**self.c2 * G_L**self.c3 * L_fi**self.c4

    def Me(self, G_W: float, G_L: float, L_fi: float) -> float:
        """Merkel number the fill provides [-]."""
        return self.Me_per_m(G_W, G_L, L_fi) * L_fi

    def fill_height(self, Me_req: float, G_W: float, G_L: float) -> float:
        """Fill height [m] that provides the required Merkel number Me_req at the loadings G_W, G_L:
        Me_req = c1 G_W^c2 G_L^c3 L_fi^(1 + c4) solved for L_fi."""
        L_fi = (Me_req / (self.c1 * G_W**self.c2 * G_L**self.c3))**(1.0 / (1.0 + self.c4))
        self.check_range(G_W, G_L, L_fi)
        return L_fi

    def check_range(self, G_W: float, G_L: float, L_fi: float) -> None:
        """Warns if the point lies outside the tested range (extrapolation of the correlation)."""
        for value, (lo, hi), what in ((G_W, self.G_W_range, 'G_W'), (G_L, self.G_L_range, 'G_L'),
                                      (L_fi, self.L_fi_range, 'L_fi')):
            if not lo <= value <= hi:
                warnings.warn(f"fill correlation {self.name}: {what} = {value:.3f} outside the tested range "
                              f"{lo}-{hi} (extrapolation)", stacklevel=3)


# =============================================================================
# 2. Fit
# =============================================================================

@dataclass(frozen=True)
class FitStatistics:
    n: int                     # number of tests
    std_err: tuple             # standard errors of (ln c1, c2, c3, c4)
    R2: float                  # coefficient of determination of ln(Me/L_fi)
    rms_rel: float             # rms of the relative deviation Me_fit / Me_data - 1
    max_rel: float             # largest absolute relative deviation


def fit_correlation(data: pd.DataFrame, y: str = 'Me_P_calc', name: str = '') -> tuple:
    """Fits Me/L_fi = c1 G_W^c2 G_L^c3 L_fi^c4 to the column y of data (columns G_w, G_a, L_fi as in Kloppers'
    tables). Returns (FillCorrelation, FitStatistics)."""
    ln_y = np.log(data[y].to_numpy())
    X = np.column_stack([np.ones(len(data)), np.log(data['G_w']), np.log(data['G_a']), np.log(data['L_fi'])])
    b, *_ = np.linalg.lstsq(X, ln_y, rcond=None)
    resid = ln_y - X @ b
    n, k = X.shape
    cov = (resid @ resid) / (n - k) * np.linalg.inv(X.T @ X)
    rel = np.exp(-resid) - 1.0                            # Me_fit / Me_data - 1
    stats = FitStatistics(n=n, std_err=tuple(float(x) for x in np.sqrt(np.diag(cov))),
                          R2=float(1.0 - (resid @ resid) / ((ln_y - ln_y.mean()) @ (ln_y - ln_y.mean()))),
                          rms_rel=float(np.sqrt(np.mean(rel**2))), max_rel=float(np.abs(rel).max()))
    corr = FillCorrelation(c1=float(np.exp(b[0])), c2=float(b[1]), c3=float(b[2]), c4=float(b[3]),
                           G_W_range=(float(data['G_w'].min()), float(data['G_w'].max())),
                           G_L_range=(float(data['G_a'].min()), float(data['G_a'].max())),
                           L_fi_range=(float(data['L_fi'].min()), float(data['L_fi'].max())), name=name)
    return corr, stats


def load_poppe_data(path: Path = DATA_POPPE) -> pd.DataFrame:
    """The re-evaluated tests (written by fill_evaluation.py)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run  python -m cooling_tower.fill_evaluation  first")
    return pd.read_csv(path, comment='#')


@lru_cache(maxsize=1)
def film_fill_poppe() -> FillCorrelation:
    """Poppe correlation of Kloppers' film fill, fitted to the own re-evaluation (data/kloppers2003_film_fill_poppe.csv).
    Computed from the data instead of typed-in coefficients, so it always matches the current evaluation."""
    corr, _ = fit_correlation(load_poppe_data(), name='film fill (Kloppers 2003 App. T, Poppe)')
    return corr


if __name__ == '__main__':
    data = load_poppe_data()
    for y, label in (('Me_P_calc', 'own Poppe evaluation (CoolProp)'), ('Me_P', "Kloppers' Poppe values"),
                     ('Me_M', "Kloppers' Merkel values")):
        c, s = fit_correlation(data, y)
        se = s.std_err
        print(f"{label}:")
        print(f"  Me/L_fi = {c.c1:.6f} G_W^{c.c2:.6f} G_L^{c.c3:.6f} L_fi^{c.c4:.6f}")
        print(f"  std. errors: ln c1 {se[0]:.4f}, c2 {se[1]:.4f}, c3 {se[2]:.4f}, c4 {se[3]:.4f}")
        print(f"  R2 = {s.R2:.4f}, rms = {100 * s.rms_rel:.2f} %, max = {100 * s.max_rel:.2f} %  (n = {s.n})")
    print(f"\nvalidity: G_W {c.G_W_range}, G_L {c.G_L_range}, L_fi {c.L_fi_range}")
