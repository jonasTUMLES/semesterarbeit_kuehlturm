"""
geometry.py -- the tower geometry.

Planned: a dataclass with the fill cross-section A_fr [m2], the fill height L_fi [m] and the fill type (key into
fill.py), plus derived quantities: the mass velocities G_w = m_dot_c / A_fr and G_a = m_dot_a / A_fr [kg/(m2 s)] that
the fill correlations and their validity ranges are written in.

Optional later: rain-zone and spray-zone heights (their transfer characteristics may stay Merkel-based, Kloppers
dissertation App. I, p. I.9), the height of the spray nozzles above the basin (static head of the pump).

Note on size: at 1 kg/s and G_w about 2.9-6 kg/(m2 s) (validity range of the fill correlation, Rubio-Castro et al.
2011, Eq. 35) the cross-section is only about 0.17-0.35 m2 -- far smaller than the test facility the fill data come
from; to be named as a model limitation.
"""
