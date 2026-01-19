import numpy as np

class UphillPhase:
    def __init__(self, direction_provider, uphill_solver, params):
        self.direction_provider = direction_provider
        self.uphill_solver = uphill_solver
        self.params = params
    def run(self, ctx):
        if self.params["e_vector_method"] == "interp":
            e_vec = self.direction_provider.evec_interp(ctx.atoms, self.params["product_atoms"], self.params["idpp_images"], self.params["use_idpp"], logfile=ctx.logfile)
        else:
            e_vec = self.direction_provider.evec_ic(ctx.atoms, self.params["reactive_bonds"], self.params["ic_mode"], logfile=ctx.logfile)
        s_k = self.uphill_solver.solve(ctx.g_k, ctx.B, e_vec, ctx.trust_radius_uphill, ctx.cos_phi)
        if np.linalg.norm(e_vec) < 1e-9 and np.linalg.norm(s_k) < 1e-9:
            ctx.logfile.write("  Warning: Uphill e-vector is zero. Returning zero step.\n")
        return s_k
