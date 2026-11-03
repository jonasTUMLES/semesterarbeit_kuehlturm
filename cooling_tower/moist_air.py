"""
moist_air.py -- the moist-air relations of the Poppe method.

Planned:
  - saturation humidity ratio Y''(T_c) and enthalpy h''(T_c) of air saturated at the LOCAL WATER temperature, per kg
    DRY air (CoolProp 'Hda', as src/thermo/precooling.calc_h since its 2026-10 fix)
  - enthalpy of the bulk air h_a(T_a, Y) and the inverse T_a(h_a, Y)
  - Lewis factor after Bosnjakovic: Le_f = 0.865^(2/3) * ((0.622 + Y'')/(0.622 + Y) - 1) / ln((0.622 + Y'')/(0.622 + Y))
    (VDI-Wärmeatlas N4, Eq. 25; Rubio-Castro et al. 2011, Eq. 4)
  - check for supersaturation (Y above saturation at the air temperature)

Reuses src/thermo/fluid_properties (CoolProp, cached) and the station pressure p_a of the climate year (about 955 hPa)
-- the saturation humidity ratio depends on p_a.
"""
