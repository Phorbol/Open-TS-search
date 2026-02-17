import numpy as np

class PRFOPhase:
    def __init__(self, prfo_solver, trust_manager):
        self.prfo_solver = prfo_solver
        self.trust_manager = trust_manager
    def _rho(self, pred, actual):
        eta = 1e-4
        if abs(pred) < eta:
            return 1.0 if abs(actual) < eta else 0.0
        return actual / pred
    def run(self, ctx):
        if (
            ctx.prev_pos is not None
            and ctx.prev_grad is not None
            and ctx.prev_energy is not None
            and ctx.prev_B is not None
            and ctx.prev_mode == 'prfo'
        ):
            s_prev = ctx.x_k - ctx.prev_pos
            pred = ctx.prev_grad.T @ s_prev + 0.5 * s_prev.T @ ctx.prev_B @ s_prev
            actual = ctx.e_k - ctx.prev_energy
            rho = self._rho(float(pred), float(actual))
            ctx.logfile.write(f"  PRFO Quality: pred={float(pred):.4e}, actual={float(actual):.4e}, rho={rho:.4f}\n")
            s_norm = np.linalg.norm(s_prev)
            new_radius = self.trust_manager.update(rho, s_norm, ctx.logfile)
            ctx.trust_radius_saddle = new_radius
        s_k = self.prfo_solver.solve(ctx.g_k, ctx.eigvals, ctx.eigvecs, ctx.trust_radius_saddle, ctx.logfile)
        return s_k, ctx.trust_radius_saddle
