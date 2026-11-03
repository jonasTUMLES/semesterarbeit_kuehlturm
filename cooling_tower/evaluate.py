"""
evaluate.py -- one operating point of the tower -> one flat result row (dict).

The key to the comparison: the row has the SAME columns as comparison/evaluate.run_point (Q_kW, T_c_o, T_a_o,
pinch, w_fr / air flow, Delta_p_a, P_fan, P_pump, P_total, m_dot_ev, target_met, freeze, ...), so that the existing
year loops, summaries and cost columns (comparison/operation.year_control, annual_summary, add_costs) work for the
tower without changes. Tower-only columns (Me, Le_f mean, Y_o, m_dot_b, m_dot_d, m_dot_mu, supersaturated) are added.

Planned:
  - run_point(cfg, geo, ambient state, air flow) -> row  (rating mode of poppe.py)
  - control_point: per hour the smallest air flow that reaches the target, analogous to
    comparison/operation.control_point (brentq), with the same target tolerance (comparison/config.TARGET_TOL)
"""
