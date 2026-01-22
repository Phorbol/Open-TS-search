import numpy as np

class _TrustRegionManager:
    def __init__(self, initial_radius, min_radius, max_radius):
        self.radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius

    def update(self, rho, s_norm, logfile):
        rho_inc = 1.035
        rho_dec = 5.0
        sigma_inc = np.sqrt(1.15)
        sigma_dec = np.sqrt(0.65)
        
        old_radius = self.radius
        new_radius = old_radius
        update_reason = "no change"
        
        if rho < 1.0 / rho_dec or rho > rho_dec:
            new_radius = max(self.min_radius, old_radius * sigma_dec)
            update_reason = "poor agreement" if rho < 1.0 / rho_dec else "excessive agreement"
        elif 1.0 / rho_inc < rho < rho_inc:
            # s_norm is passed directly now
            if abs(s_norm - old_radius) < 1e-3:
                new_radius = min(self.max_radius, old_radius * sigma_inc)
                update_reason = "good agreement (boundary step)"
            else:
                update_reason = "good agreement (interior step)"
        
        self.radius = new_radius
        
        if abs(new_radius - old_radius) > 1e-6 and logfile:
            logfile.write(f"  Trust radius update: {old_radius:.4e} -> {new_radius:.4e} (rho={rho:.4f}, reason: {update_reason})\n")
            logfile.flush()
        
        return new_radius

    def get_radius(self):
        return self.radius

    def set_radius(self, radius):
        self.radius = radius

    # Deprecated / Adapter for old code if needed, but we will update optimizer
    def update_saddle(self, atoms, pos_k_minus_1, old_radius, rho, logfile):
        # This was the old signature. We should avoid using it.
        # But to be safe, if called, we can adapt.
        s_norm = np.linalg.norm(atoms.get_positions().flatten() - pos_k_minus_1)
        self.radius = old_radius # Sync state
        return self.update(rho, s_norm, logfile)
