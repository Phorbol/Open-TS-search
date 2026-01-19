import numpy as np

class GPUTrustRegionManager:
    def __init__(self, initial_radius=0.05, min_radius=5e-3, max_radius=0.2):
        self.radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Parameters for update strategy
        self.rho_inc = 1.035
        self.rho_dec = 5.0      
        self.sigma_inc = np.sqrt(1.15)
        self.sigma_dec = np.sqrt(0.65) 

    def update(self, rho, s_norm, logfile=None):
        """
        Update trust region radius based on the ratio of actual to predicted energy change (rho).
        """
        old_radius = self.radius
        
        if rho < 1.0/self.rho_dec or rho > self.rho_dec:
            self.radius = max(self.min_radius, old_radius * self.sigma_dec)
        elif 1.0/self.rho_inc < rho < self.rho_inc:
            # Only increase if we are pushing the boundary of the current trust region
            if abs(s_norm - old_radius) < 1e-3:
                self.radius = min(self.max_radius, old_radius * self.sigma_inc)
        
        if logfile and self.radius != old_radius:
            # Optional logging if needed, though Driver usually handles high-level logging
            pass
            
        return self.radius

    def get_radius(self):
        return self.radius

    def set_radius(self, radius):
        self.radius = radius

    def reset(self, radius=None):
        if radius is not None:
            self.radius = radius
