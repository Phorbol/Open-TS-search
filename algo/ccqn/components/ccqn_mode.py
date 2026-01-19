from scipy.linalg import eigh

class CCQNModeSelector:
    def select(self, B, mode, logfile, trust_radius_saddle_initial):
        try:
            eigvals, eigvecs = eigh(B)
            logfile.write(f"\nThe smallest eigenvalue of Hessian is {eigvals[0]:.4e}")
            trust_reset = None
            if mode == 'uphill':
                if eigvals[0] < -1e-6:
                    mode = 'prfo'
                    logfile.write(f"\nSwitching to 'prfo' mode.")
                    trust_reset = trust_radius_saddle_initial
            elif mode == 'prfo':
                if eigvals[0] > 1e-2:
                    mode = 'uphill'
                    logfile.write(f"\nSwitching to 'uphill' mode.")
            return eigvals, eigvecs, mode, trust_reset
        except Exception:
            logfile.write("Hessian diagonalization failed. Resetting Hessian.\n")
            raise
