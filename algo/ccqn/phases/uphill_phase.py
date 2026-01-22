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
        
        # Log overlap with eigenvectors (Debug info)
        if np.linalg.norm(e_vec) > 1e-9 and ctx.eigvecs is not None:
             e_vec_norm = e_vec / np.linalg.norm(e_vec)
             # ctx.eigvecs columns are eigenvectors. 
             # overlap[i] = dot(e_vec, v_i)
             overlaps = np.abs(np.dot(e_vec_norm, ctx.eigvecs))[:3]
             ctx.logfile.write(f"  e_vector Overlaps (v0, v1, v2): [{overlaps[0]:.4f}, {overlaps[1]:.4f}, {overlaps[2]:.4f}]\n")

        s_k = self.uphill_solver.solve(ctx.g_k, ctx.B, e_vec, ctx.trust_radius_uphill, ctx.cos_phi)
        if np.linalg.norm(e_vec) < 1e-9 and np.linalg.norm(s_k) < 1e-9:
            ctx.logfile.write("  Warning: Uphill e-vector is zero. Returning zero step.\n")
        return s_k
