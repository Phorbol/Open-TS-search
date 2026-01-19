import torch
import numpy as np

class GPUHessianManager:
    def __init__(self, atoms, hessian=False, device=None):
        self.atoms = atoms
        self.use_calc = hessian
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self):
        natoms = len(self.atoms)
        if self.use_calc:
            # Note: get_hessian() might be expensive, so we do it on CPU then transfer
            try:
                h_np = self.atoms.calc.get_hessian(self.atoms).reshape(3 * natoms, 3 * natoms)
                return torch.from_numpy(h_np).to(self.device, dtype=torch.float64)
            except AttributeError:
                # Fallback if calculator doesn't support get_hessian
                pass
        
        # Initialize as diagonal matrix, using double precision
        return torch.eye(3 * natoms, device=self.device, dtype=torch.float64) * 70.0

    def update(self, B, s, y, logfile=None):
        """
        Generic update method (defaults to TS-BFGS).
        """
        return self.update_ts_bfgs(B, s, y, logfile)

    def update_ts_bfgs(self, B, s, y, logfile=None):
        """
        Pure GPU implementation of TS-BFGS update.
        """
        try:
            eigvals, eigvecs = torch.linalg.eigh(B)
            # B_tilde construction ensuring positive definiteness proxy
            B_tilde = eigvecs @ (torch.diag(torch.abs(eigvals)) @ eigvecs.T)
        except RuntimeError:
            if logfile:
                logfile.write("Warning: Hessian diagonalization failed in update. Using B directly.\n")
            B_tilde = B
        
        # Calculate update terms
        # s and y are 1D tensors. @ operator works as dot product for 1D.
        
        # Term: s^T * B_tilde * s (scalar)
        s_B_s = s @ (B_tilde @ s)
        
        M_k = torch.outer(y, y) + s_B_s * B_tilde
        j_k = y - (B @ s)
        sMs = s @ (M_k @ s)
        
        if abs(sMs) < 1e-12:
            return B
        
        u_k = (M_k @ s) / sMs
        jTs = j_k @ s
        
        delta_B = torch.outer(j_k, u_k) + torch.outer(u_k, j_k) - jTs * torch.outer(u_k, u_k)
        
        return B + delta_B
