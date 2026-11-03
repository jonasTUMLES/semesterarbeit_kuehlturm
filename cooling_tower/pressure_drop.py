"""
pressure_drop.py -- air-side pressure drop of the tower.

Planned: the sum of the component losses -- inlet (louvres), rain zone, fill, water distribution, drift eliminator,
outlet (dynamic pressure) (VDI-Wärmeatlas N4, Sec. 6.3, Eq. 40). Fill: Delta_p_fi = K_fi L_fi m_dot_av^2 / (2 rho_m A_fr^2)
with K_fi from fill.py; other losses e.g. after Rubio-Castro et al. 2011, Eqs. 16-23.

The fan power then comes from src/auxiliaries/economics.calc_fan_power (P = m_dot_a Delta_p / (eta_fan rho), the same
as VDI N4 Eq. 41) -- no new formula needed.

Pump: unlike the dry cooler, the open tower's pump lifts the water to the spray nozzles (static head + nozzle
pressure); the friction in the tubes hardly matters. To be added here or in water_balance.py.
"""
