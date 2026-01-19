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
        if ctx.prev_pos is not None:
            s_prev = ctx.x_k - ctx.prev_pos
            pred = ctx.prev_grad.T @ s_prev + 0.5 * s_prev.T @ ctx.B @ s_prev
            actual = ctx.e_k - ctx.prev_energy
            rho = self._rho(pred, actual)
            ctx.logfile.write(f"  PRFO Quality: rho={rho:.4f}\n")
            new_radius = self.trust_manager.update_saddle(ctx.atoms, ctx.prev_pos, ctx.trust_radius_saddle, rho, ctx.logfile)
            ctx.trust_radius_saddle = new_radius
        s_k = self.prfo_solver.solve(ctx.g_k, ctx.eigvals, ctx.eigvecs, ctx.trust_radius_saddle, ctx.logfile)
        return s_k, ctx.trust_radius_saddle
