import numpy as np
from scipy.linalg import eigh
from ase import Atoms

class HessianTSBFGS:
    def __init__(self, atoms: Atoms, hessian: bool):
        self.atoms = atoms
        self.use_calc = hessian
    def initialize(self):
        natoms = len(self.atoms)
        if self.use_calc:
            return self.atoms.calc.get_hessian(self.atoms).reshape(3 * natoms, 3 * natoms)
        return np.eye(3 * natoms) * 70.0
    def update_ts_bfgs(self, B, s, y, logfile):
        try:
            eigvals_B, eigvecs_B = eigh(B)
            # Optimized computation of z = |B| s without forming full B_tilde
            s_proj = eigvecs_B.T @ s
            z = eigvecs_B @ (np.abs(eigvals_B) * s_proj)
        except np.linalg.LinAlgError:
            logfile.write("Warning: Diagonalization failed in TS-BFGS update. Using B directly.\n")
            z = B @ s

        # sMs = s.T @ M_k @ s = (s.T y)^2 + (s.T z)^2
        sTy = np.dot(s, y)
        sTz = np.dot(s, z)
        sMs = sTy**2 + sTz**2

        if abs(sMs) < 1e-12:
            return B
        
        # u_k = (M_k @ s) / sMs
        # M_k @ s = (s.T y) y + (s.T z) z
        Mks = sTy * y + sTz * z
        u_k = Mks / sMs
        
        j_k = y - B @ s
        jTs = np.dot(j_k, s)
        
        term1 = np.outer(j_k, u_k)
        term2 = np.outer(u_k, j_k)
        term3 = jTs * np.outer(u_k, u_k)
        delta_B = term1 + term2 - term3
        return B + delta_B
