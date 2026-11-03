"""
design.py -- design of the tower for the design ambient state.

Planned:
  1. for a range of L/G: required Merkel number from poppe.py (and the minimum air flow below which it diverges,
     Picardo & Variyar 2012, pp. 168 f.)
  2. cross-section from a chosen G_a inside the correlation's range, fill height L_fi = Me_required / (Me / L_fi)
     from fill.py (iterative if the correlation contains L_fi)
  3. pressure drop and fan power per L/G (pressure_drop.py)
  4. annual cost per L/G: capital + electricity + water; the optimum L/G (Picardo & Variyar 2012, pp. 167, 171;
     Rubio-Castro et al. 2011, Eqs. 37-39)

Fairness of the comparison (open point): the dry coolers' capital cost is only the raw-metal write-off (no construction
cost model, docs/model-notes.md Sec. 3). The tower needs the SAME cost basis, otherwise the capital costs of the systems
are not comparable.
"""
