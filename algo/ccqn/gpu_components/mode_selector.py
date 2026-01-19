import torch

class GPUModeSelector:
    def __init__(self, trust_radius_reset_value=0.05):
        self.trust_radius_reset_value = trust_radius_reset_value

    def select(self, current_mode, eigvals, logfile=None):
        """
        Determine the next mode based on Hessian eigenvalues.
        
        Returns:
            new_mode (str): The next mode ('uphill' or 'prfo').
            trust_reset (float or None): If not None, the trust radius should be reset to this value.
            reason (str): The reason for the mode switch (or None if no switch).
        """
        min_eig = eigvals[0].item() # Scalar sync
        new_mode = current_mode
        trust_reset = None
        reason = None

        if current_mode == 'uphill':
            if min_eig < -1e-6:
                new_mode = 'prfo'
                trust_reset = self.trust_radius_reset_value
                reason = f"Min eigenvalue ({min_eig:.4e}) < -1e-6"
                if logfile: logfile.write(f" -> Switching to PRFO. Reason: {reason}\n")
        elif current_mode == 'prfo':
            if min_eig > 1e-2:
                new_mode = 'uphill'
                reason = f"Min eigenvalue ({min_eig:.4e}) > 1e-2"
                if logfile: logfile.write(f" -> Switching to Uphill. Reason: {reason}\n")
        
        return new_mode, trust_reset, reason
