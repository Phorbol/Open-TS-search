import torch

class GPUModeSelector:
    def __init__(self, trust_radius_reset_value=0.05, prfo_patience=5):
        self.trust_radius_reset_value = trust_radius_reset_value
        self.prfo_patience = prfo_patience
        self.prfo_counter = 0

    def select(self, current_mode, eigvals, logfile=None):
        """
        Determine the next mode based on Hessian eigenvalues.
        
        Returns:
            new_mode (str): The next mode ('uphill' or 'prfo').
            trust_reset (float or None): If not None, the trust radius should be reset to this value.
            reason (str): The reason for the mode switch (or None if no switch).
        """
        min_eig = eigvals[0].item()
        new_mode = current_mode
        trust_reset = None
        reason = None

        if current_mode == 'uphill':
            self.prfo_counter = 0
            if min_eig < -1e-6:
                new_mode = 'prfo'
                trust_reset = self.trust_radius_reset_value
                reason = f"Min eigenvalue ({min_eig:.4e}) < -1e-6"
                if logfile: logfile.write(f" -> Switching to PRFO. Reason: {reason}\n")
        elif current_mode == 'prfo':
            self.prfo_counter += 1
            if min_eig > 1e-2:
                if self.prfo_counter < self.prfo_patience and min_eig < 0.1:
                    if logfile:
                        logfile.write(f" -> [DEBUG] Keeping PRFO despite min_eig={min_eig:.4e} (Step {self.prfo_counter}/{self.prfo_patience})\n")
                else:
                    new_mode = 'uphill'
                    reason = f"Min eigenvalue ({min_eig:.4e}) > 1e-2 (PRFO steps: {self.prfo_counter})"
                    if logfile: logfile.write(f" -> Switching to Uphill. Reason: {reason}\n")
        
        return new_mode, trust_reset, reason
