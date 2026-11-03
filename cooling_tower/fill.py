"""
fill.py -- fill performance characteristics.

Planned:
  - Merkel number per metre of fill as a function of G_w and G_a, EVALUATED WITH POPPE (consistency rule: the method
    used to evaluate the fill test must also predict the tower, Kloppers & Kröger 2005, IJHMT 48, p. 774;
    Kloppers dissertation Sec. 4.5, p. 4.10). Source: Kloppers dissertation (Stellenbosch 2003), App. T, Tab. T.7,
    p. T.6 (film fill, test height 0.9 m), e.g. Me_P / L_fi = 1.193519 G_w^-0.397740 G_a^0.724438.
  - loss coefficient per metre of fill K_fi(G_w, G_a), Poppe variant, same appendices (R-T).
  - the validity ranges of both, with a warning outside them (like pressure_drop.check_wang_range).

Alternative / cross-check: Rubio-Castro et al. (2011), Eq. 8 + Tab. 1 (film fill, with an explicit dependence on L_fi
and T_c_i); it is not stated there whether that correlation was evaluated with Poppe or Merkel.
"""
