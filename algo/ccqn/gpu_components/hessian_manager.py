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

    def update(self, B, s, y, logfile=None, eigvals=None, eigvecs=None):
        """
        Generic update method (defaults to TS-BFGS).
        """
        return self.update_ts_bfgs(B, s, y, logfile, eigvals, eigvecs)

    def update_ts_bfgs(self, B, s, y, logfile=None, eigvals=None, eigvecs=None):
        """
        Pure GPU implementation of TS-BFGS update.
        """
        try:
            if eigvals is None or eigvecs is None:
                eigvals, eigvecs = torch.linalg.eigh(B)
            # Optimized z = |B| s calculation without forming full B_tilde
            s_proj = eigvecs.T @ s
            z = eigvecs @ (torch.abs(eigvals) * s_proj)
        except RuntimeError:
            if logfile:
                logfile.write("Warning: Hessian diagonalization failed in update. Using B directly.\n")
            z = B @ s
        
        # sMs = s.T @ M_k @ s = (s.T y)^2 + (s.T z)^2
        sTy = torch.dot(s, y)
        sTz = torch.dot(s, z)
        
        sMs = sTy**2 + sTz**2
        
        if abs(sMs) < 1e-12:
            return B
        
        # u_k = (M_k @ s) / sMs
        # M_k @ s = (s.T y) y + (s.T z) z
        Mks = sTy * y + sTz * z
        u_k = Mks / sMs
        
        j_k = y - (B @ s)
        jTs = torch.dot(j_k, s)
        
        term1 = torch.outer(j_k, u_k)
        term2 = torch.outer(u_k, j_k)
        term3 = jTs * torch.outer(u_k, u_k)
        
        delta_B = term1 + term2 - term3
        
        return B + delta_B
