"""
test_cooling_tower.py -- validation of the cooling-tower package (planned, no tests yet).

Planned checks:
  - Poppe: Kloppers dissertation, App. G (counterflow fill analysis), Tab. H.2: Me_P = 0.741 (Merkel 0.685)
  - limiting case: Poppe with Le_f = 1 and no evaporation loss (poppe.py validation switches) reproduces
    Perry's Handbook, Example 12-10 (41 -> 29.4 °C, wet bulb 25 °C, L/G = 1.2): KaV/L = 2.32 (p. 12-16)
  - the predecessor's tests (tests/test_solvers_golden.py etc.) must keep passing unchanged
"""
