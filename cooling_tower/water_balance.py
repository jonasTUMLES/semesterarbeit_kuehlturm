"""
water_balance.py -- the water balance of the open tower.

Planned:
  - evaporation m_dot_ev = m_dot_da (Y_o - Y_i), from poppe.py (dry-air flow, as the pad water in
    src/auxiliaries/economics.calc_water_usage)
  - drift as a fraction of the circulating water (Rubio-Castro et al. 2011: 0.2 %; Perry's: below 0.02 % with modern
    eliminators -- the chosen value to be justified)
  - blowdown from the cycles of concentration n: m_dot_b = m_dot_ev / (n - 1) - m_dot_d (Perry's Eqs. 12-21/22)
  - make-up m_dot_mu = m_dot_ev + m_dot_d + m_dot_b = n / (n - 1) m_dot_ev (Rubio-Castro Eq. 28)

For a fair comparison the blowdown goes to the sewer (wastewater fee), the pad of the hybrid cooler is charged for its
evaporated water only (src/parameters.py WASTEWATER_EUR_PER_M3 = 0) -- whether pads also need a bleed-off is to be
checked in the literature (ASHRAE Handbook, Ch. 41).
"""
