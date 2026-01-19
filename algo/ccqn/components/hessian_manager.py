import numpy as np
from scipy.linalg import eigh
from ase import Atoms

class _HessianManager:
    def __init__(self, atoms: Atoms, hessian: bool):
        self.atoms = atoms
        self.use_calc = hessian
    def initialize(self):
        natoms = len(self.atoms)
        if self.use_calc:
            return self.atoms.calc.get_hessian(self.atoms).reshape(3 * natoms, 3 * natoms)
        return np.eye(3 * natoms) * 70.0
    def update_ts_bfgs(self, B, s, y, logfile, eigvals=None, eigvecs=None):
        try:
            if eigvals is None or eigvecs is None:
                eigvals_B, eigvecs_B = eigh(B)
            else:
                eigvals_B, eigvecs_B = eigvals, eigvecs
            B_tilde = eigvecs_B @ np.diag(np.abs(eigvals_B)) @ eigvecs_B.T
        except np.linalg.LinAlgError:
            logfile.write("Warning: Diagonalization failed in TS-BFGS update. Using B directly.\n")
            B_tilde = B
        M_k = np.outer(y, y) + (s.T @ B_tilde @ s) * B_tilde
        j_k = y - B @ s
        sMs = s.T @ M_k @ s
        if abs(sMs) < 1e-12:
            return B
        u_k = (M_k @ s) / sMs
        jTs = j_k.T @ s
        term1 = np.outer(j_k, u_k)
        term2 = np.outer(u_k, j_k)
        term3 = jTs * np.outer(u_k, u_k)
        delta_B = term1 + term2 - term3
        return B + delta_B
