"""
config.py -- the settings of a cooling-tower run.

Planned: a frozen dataclass `Settings`, read ONCE per run and passed as the first argument (`cfg`) to every function that
needs a setting -- the same pattern as comparison/config.py, so a run is reproducible from its `cfg`.

Planned contents:
  - cooling task: m_dot_c [kg/s], T_c_i [°C], T_c_target [°C] (reused from src/parameters.py: T_COOLANT_IN,
    T_COOLANT_TARGET_OUT, COMPARE_M_COOLANT, so all three systems solve the same task)
  - fill: fill type (key into fill.py), validity ranges of the correlation
  - design: L/G range of the design sweep, design ambient state (wet-bulb based, see below)
  - operation: fan speed / air flow range for the control (analogous to w_fr_min / w_fr_design of the comparison)
  - water: drift fraction, cycles of concentration
  - prices: electricity and water as in src/parameters.py; wastewater fee for the blowdown (it reaches the sewer,
    unlike the evaporated pad water, for which WASTEWATER_EUR_PER_M3 = 0)
  - numerics: number of Runge-Kutta intervals, tolerances of the outlet-humidity and outlet-temperature iterations

Kept separate from src/parameters.py on purpose: the predecessor's inputs and pinned test numbers stay untouched.
Whether the tower settings later appear in the GUI (sections tagged in parameters.py) is decided at the end.

Open point: the design ambient. The comparison's design points are dry-bulb percentiles (20 °C / 60 %, P95, P99); the
tower needs a wet-bulb design value (e.g. 0.4 % or 1 % exceedance, ASHRAE Fundamentals, Munich), cross-checked with
src/inputs/weather.add_wet_bulb on the climate year.
"""
