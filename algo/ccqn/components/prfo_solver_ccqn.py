import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

class _PRFOSolver:
    def _solve_rfo_subproblem(self, lambdas, g_tilde, mode, a_sq=1.0):
        dim = len(lambdas)
        if dim == 0:
            return np.array([]), 0.0
        eps_machine = 1e-15
        a = np.sqrt(max(a_sq, eps_machine))

        # Optimization: Use secular equation for large dimensions to avoid O(N^3) diagonalization
        if dim > 5:
            try:
                v = g_tilde / a
                v_sq = v**2
                
                def secular_eq(mu):
                    # f(mu) = -mu + sum(v_i^2 / (lambda_i - mu))
                    return -mu + np.sum(v_sq / (lambdas - mu))

                if mode == 'min':
                    # Find smallest root in (-inf, min(lambdas))
                    lambda_min = np.min(lambdas)
                    upper = lambda_min - 1e-9
                    lower = lambda_min - np.linalg.norm(v) - 1.0
                    
                    # Ensure bracketing: f(lower) should be positive (left of root)
                    # f(upper) approaches -inf
                    cnt = 0
                    while secular_eq(lower) < 0 and cnt < 20:
                        lower = lower * 2.0 if lower < 0 else lower - 10.0
                        cnt += 1
                    
                    if secular_eq(lower) > 0:
                        mu = brentq(secular_eq, lower, upper, xtol=1e-12)
                        s_tilde = (v / (mu - lambdas)) * a
                        epsilon = mu / 2.0
                        return s_tilde, epsilon
                
                else: # mode == 'max'
                    # Find largest root in (max(lambdas), +inf)
                    lambda_max = np.max(lambdas)
                    lower = lambda_max + 1e-9
                    upper = lambda_max + np.linalg.norm(v) + 1.0
                    
                    # Ensure bracketing: f(upper) should be negative (right of root)
                    # f(lower) approaches +inf
                    cnt = 0
                    while secular_eq(upper) > 0 and cnt < 20:
                        upper = upper * 2.0 if upper > 0 else upper + 10.0
                        cnt += 1

                    if secular_eq(upper) < 0:
                        mu = brentq(secular_eq, lower, upper, xtol=1e-12)
                        s_tilde = (v / (mu - lambdas)) * a
                        epsilon = mu / 2.0
                        return s_tilde, epsilon
            except (ValueError, RuntimeError):
                # Fallback to full diagonalization if secular solver fails
                pass

        H_aug = np.zeros((dim + 1, dim + 1))
        H_aug[:dim, :dim] = np.diag(lambdas)
        H_aug[:dim, dim] = g_tilde / a
        H_aug[dim, :dim] = g_tilde / a
        try:
            aug_eigvals, aug_eigvecs = eigh(H_aug)
            idx = -1 if mode == 'max' else 0
            chosen_eigvec, chosen_eigval = aug_eigvecs[:, idx], aug_eigvals[idx]
            scale = chosen_eigvec[-1]
            if abs(scale) < eps_machine:
                lambda_diag = np.diag(lambdas)
                s_tilde = -np.linalg.pinv(lambda_diag, rcond=eps_machine) @ g_tilde
                return s_tilde, 0.0
            s_tilde = (chosen_eigvec[:dim] / scale) * a
            epsilon = chosen_eigval / 2.0
            if not np.all(np.isfinite(s_tilde)):
                raise ValueError("non-finite")
            return s_tilde, epsilon
        except (np.linalg.LinAlgError, ValueError):
            grad_norm = np.linalg.norm(g_tilde)
            if grad_norm > eps_machine:
                direction = 1.0 if mode == 'max' else -1.0
                s_tilde = direction * g_tilde / grad_norm * a
            else:
                s_tilde = np.zeros_like(g_tilde)
            return s_tilde, 0.0
    def solve(self, g, eigvals, eigvecs, trust_radius_saddle, logfile):
        tol = 1e-15
        g_tilde = eigvecs.T @ g
        g_tilde_max = g_tilde[:1]
        g_tilde_min = g_tilde[1:]
        eigvals_max = eigvals[:1]
        eigvals_min = eigvals[1:]
        s_max_tilde_unc = -np.linalg.pinv(np.diag(eigvals_max), rcond=tol) @ g_tilde_max
        s_min_tilde_unc = -np.linalg.pinv(np.diag(eigvals_min), rcond=tol) @ g_tilde_min
        s_tilde_unc = np.concatenate([s_max_tilde_unc, s_min_tilde_unc])
        s_unc = eigvecs @ s_tilde_unc
        norm_s_unc = np.linalg.norm(s_unc)
        if norm_s_unc <= trust_radius_saddle:
            return s_unc
        def constraint_residual(alpha_sq):
            a2 = max(alpha_sq, tol)
            try:
                s_max_tilde, _ = self._solve_rfo_subproblem(eigvals[:1], g_tilde[:1], 'max', a2)
                s_min_tilde, _ = self._solve_rfo_subproblem(eigvals[1:], g_tilde[1:], 'min', a2)
                norm_sq = np.sum(s_max_tilde ** 2) + np.sum(s_min_tilde ** 2)
                return norm_sq - trust_radius_saddle ** 2
            except (ZeroDivisionError, np.linalg.LinAlgError, ValueError):
                return 1e6
        try:
            alpha_sq_opt = brentq(constraint_residual, 1e-20, 1e6, xtol=tol)
            s_max_tilde, _ = self._solve_rfo_subproblem(eigvals[:1], g_tilde[:1], 'max', alpha_sq_opt)
            s_min_tilde, _ = self._solve_rfo_subproblem(eigvals[1:], g_tilde[1:], 'min', alpha_sq_opt)
            s_tilde = np.concatenate([s_max_tilde, s_min_tilde])
            s = eigvecs @ s_tilde
        except (RuntimeError, ValueError):
            logfile.write("  Warning: Trust radius constraint failed, using simple scaling fallback.\n")
            if norm_s_unc > tol:
                s = s_unc * (trust_radius_saddle / norm_s_unc)
            else:
                s = np.zeros_like(g)
        s_norm_final = np.linalg.norm(s)
        if s_norm_final > (trust_radius_saddle * 1.05):
            logfile.write(f"  Warning: PRFO step norm {s_norm_final:.4f} numerically exceeded trust radius {trust_radius_saddle:.4f}. Rescaling step.\n")
            s *= (trust_radius_saddle / s_norm_final)
        return s
