import numpy as np
from scipy.optimize import minimize

class CCQNUphillSolver:
    def solve(self, g, B, e_vec, trust_radius_uphill, cos_phi):
        s0 = e_vec * trust_radius_uphill
        if np.linalg.norm(s0) < 1e-9:
            return np.zeros_like(g)
        def objective(s):
            return g.T @ s + 0.5 * s.T @ B @ s
        def jac_objective(s):
            return g + B @ s
        def eq_constraint_fun(s):
            return s.T @ s - trust_radius_uphill ** 2
        def jac_eq_constraint(s):
            return 2 * s
        def ineq_constraint_fun(s):
            return e_vec.T @ s - cos_phi * trust_radius_uphill
        def jac_ineq_constraint(s):
            return e_vec
        constraints = [
            {'type': 'eq', 'fun': eq_constraint_fun, 'jac': jac_eq_constraint},
            {'type': 'ineq', 'fun': ineq_constraint_fun, 'jac': jac_ineq_constraint}
        ]
        res = minimize(objective, s0, jac=jac_objective, constraints=constraints, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-6})
        if res.success:
            return res.x
        s_fallback = e_vec * trust_radius_uphill
        return s_fallback
