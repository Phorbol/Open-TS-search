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
            
            # Sync trust manager with context (if not already managed internally)
            # The trust manager is stateful now.
            if hasattr(self.trust_manager, 'set_radius'):
                 # Ensure it starts from correct radius if context has override?
                 # Actually context radius usually comes from manager.
                 # But let's assume manager is the source of truth.
                 pass

            s_norm = np.linalg.norm(s_prev)
            new_radius = self.trust_manager.update(rho, s_norm, ctx.logfile)
            ctx.trust_radius_saddle = new_radius
        s_k = self.prfo_solver.solve(ctx.g_k, ctx.eigvals, ctx.eigvecs, ctx.trust_radius_saddle, ctx.logfile)
        return s_k, ctx.trust_radius_saddle
