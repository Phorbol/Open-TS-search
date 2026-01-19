import numpy as np

class CCQNConvergenceChecker:
    def converged(self, atoms, forces, fmax_threshold, mode):
        if forces is None:
            forces = atoms.get_forces()
        elif hasattr(forces, 'ndim') and forces.ndim == 1:
            forces = forces.reshape(-1, 3)
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        return (fmax < fmax_threshold) and (mode == 'prfo')
