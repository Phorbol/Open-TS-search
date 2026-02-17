import numpy as np

class TrustRegionLinear:
    def __init__(self, min_radius, max_radius):
        self.min_radius = min_radius
        self.max_radius = max_radius
    def update_saddle(self, atoms, pos_k_minus_1, old_radius, rho, logfile):
        rho_inc = 1.035
        rho_dec = 5.0
        sigma_inc = np.sqrt(1.15)
        sigma_dec = np.sqrt(0.65)
        new_radius = old_radius
        update_reason = "no change"
        if rho < 1.0 / rho_dec or rho > rho_dec:
            new_radius = max(self.min_radius, old_radius * sigma_dec)
            update_reason = "poor agreement" if rho < 1.0 / rho_dec else "excessive agreement"
        elif 1.0 / rho_inc < rho < rho_inc:
            s_norm = np.linalg.norm(atoms.get_positions().flatten() - pos_k_minus_1)
            if abs(s_norm - old_radius) < 1e-3:
                new_radius = min(self.max_radius, old_radius * sigma_inc)
                update_reason = "good agreement (boundary step)"
            else:
                update_reason = "good agreement (interior step)"
        if abs(new_radius - old_radius) > 1e-6:
            logfile.write(f"  Trust radius update: {old_radius:.4e} -> {new_radius:.4e} (rho={rho:.4f}, reason: {update_reason})\n")
            logfile.flush()
        return new_radius
