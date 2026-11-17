"""
fill_evaluation.py -- evaluation of fill performance tests with poppe.design.

Why: the Merkel number the fill provides (Me per metre fill height, as a function of G_W, G_L, ...) has to be
evaluated with the same method as the tower design (consistency rule, Kloppers & Kröger 2005; Kloppers'
dissertation, Ch. 4). Here each test is evaluated with exactly the code and the property model of the design
(poppe.design, saturation='coolprop'), so the fill correlation fitted from these values (fill.py) is consistent
with the required Me of the design by construction.

Data: cooling_tower/data/kloppers2003_film_fill.csv -- Kloppers' dissertation, App. T, cross-corrugated film fill,
54 tests at three fill heights (measurements Tables T.1/T.5/T.9, his own evaluation Tables T.2/T.6/T.10).

Per test:
  1. inlet humidity Y_i from dry and wet bulb (moist_air.Y_inlet_wb, CoolProp)
  2. poppe.design with the measured water inlet / outlet temperatures, water and dry air flows, pressure
     -> required Merkel number of the test = the Merkel number the fill delivered in the test
  3. Me / L_fi and the computed air outlet temperature, compared with
       - Kloppers' Poppe evaluation Me_P / L_fi and T_ao_P (checks the implementation and the property model)
       - the measured air outlet temperature T_ao (checks the model itself)

Run as a script for the full table:  python -m cooling_tower.fill_evaluation
"""

from pathlib import Path

import pandas as pd

from cooling_tower import moist_air as ma
from cooling_tower import poppe as pp

DATA_KLOPPERS = Path(__file__).parent / 'data' / 'kloppers2003_film_fill.csv'


def load_tests(path: Path = DATA_KLOPPERS) -> pd.DataFrame:
    """Fill test table (one row per test), comment lines starting with '#' skipped."""
    return pd.read_csv(path, comment='#')


def evaluate_test(row, saturation: str = 'coolprop', n_intervals: int = 20) -> dict:
    """Poppe evaluation of one fill test. row needs the columns L_fi, p_a, T_ai, T_wb, T_wi, T_wo, m_a, m_w."""
    Y_i = ma.Y_inlet_wb(row.T_ai, row.T_wb, row.p_a)
    r = pp.design(T_W_i=row.T_wi, T_W_o=row.T_wo, m_dot_W_i=row.m_w, m_dot_L=row.m_a, T_L_i=row.T_ai, Y_i=Y_i,
                  p=row.p_a, n_intervals=n_intervals, saturation=saturation)
    return {'Y_i': Y_i,
            'Me_P_calc': r.Me / row.L_fi,              # Merkel number per metre fill height [1/m]
            'T_ao_P_calc': r.T_L_o,                    # air outlet temperature [°C]
            'Y_o_calc': r.Y_o,
            'm_ev_calc': r.m_dot_ev,                   # evaporated water [kg/s]
            'supersat_calc': r.supersaturated}


def evaluate_tests(tests: pd.DataFrame | None = None, saturation: str = 'coolprop',
                   n_intervals: int = 20) -> pd.DataFrame:
    """Evaluates all tests; returns the table with the computed columns and the deviations
    dMe_rel = Me_P_calc / Me_P - 1 (against Kloppers), dT_ao_P = T_ao_P_calc - T_ao_P (against Kloppers),
    dT_ao_meas = T_ao_P_calc - T_ao (against the measurement)."""
    tests = load_tests() if tests is None else tests
    calc = pd.DataFrame([evaluate_test(row, saturation, n_intervals) for row in tests.itertuples(index=False)],
                        index=tests.index)
    out = pd.concat([tests, calc], axis=1)
    out['dMe_rel'] = out['Me_P_calc'] / out['Me_P'] - 1.0
    out['dT_ao_P'] = out['T_ao_P_calc'] - out['T_ao_P']
    out['dT_ao_meas'] = out['T_ao_P_calc'] - out['T_ao']
    return out


def summary(out: pd.DataFrame) -> pd.DataFrame:
    """Mean, mean absolute and maximum absolute deviation per fill height and in total."""
    cols = ['dMe_rel', 'dT_ao_P', 'dT_ao_meas']
    rows = {}
    for key, grp in list(out.groupby('L_fi')) + [('alle', out)]:
        rows[key] = {**{f'{c} mean': grp[c].mean() for c in cols},
                     **{f'{c} |max|': grp[c].abs().max() for c in cols},
                     'supersaturated': int(grp['supersat_calc'].sum())}
    return pd.DataFrame(rows).T


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 100)
    result = evaluate_tests()
    print(result[['L_fi', 'test', 'Me_P', 'Me_P_calc', 'dMe_rel', 'T_ao', 'T_ao_P', 'T_ao_P_calc',
                  'supersat_calc']].round(4).to_string(index=False))
    print()
    print(summary(result).round(4).to_string())
