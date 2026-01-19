import numpy as np
from ase.geometry import find_mic
from shared.interp import robust_interpolate, Vectorized_ASE_IDPPSolver

class _DirectionProvider:
    def evec_interp(self, atoms, product_atoms, idpp_images, use_idpp, logfile=None):
        if logfile is not None:
            logfile.write(f"  Calculating e-vector via path interpolation ({'IDPP' if use_idpp else 'Linear'})...\n")
        path_func = Vectorized_ASE_IDPPSolver.from_endpoints if use_idpp else robust_interpolate
        path_tmp = path_func(atoms.copy(), product_atoms.copy(), idpp_images)
        if use_idpp:
            path = path_tmp.run()
        else:
            path = path_tmp
        midpoint_index = len(path) // 2
        x_mid = path[midpoint_index].get_positions()
        x_current = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        raw = x_mid - x_current
        mic, _ = find_mic(raw, cell, pbc)
        e_vec = mic.flatten()
        n = np.linalg.norm(e_vec)
        if n > 1e-8:
            return e_vec / n
        return np.zeros_like(e_vec)
    def evec_ic(self, atoms, reactive_bonds, ic_mode, logfile=None):
        if logfile is not None:
            logfile.write(f"  Calculating e-vector via IC (Mode: {ic_mode})...\n")
        coords = atoms.get_positions()
        natoms = len(atoms)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        bonds = np.array(reactive_bonds, dtype=int)
        i_idx, j_idx = bonds[:, 0], bonds[:, 1]
        raw_v_ij = coords[j_idx] - coords[i_idx]
        v_ij, _ = find_mic(raw_v_ij, cell, pbc)
        norm_v = np.linalg.norm(v_ij, axis=1)
        valid = norm_v > 1e-8
        forces = atoms.get_forces()
        v_ij = v_ij[valid]
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        if v_ij.shape[0] == 0:
            if logfile is not None:
                logfile.write("  Warning: No valid reactive bonds found for IC mode.\n")
            return np.zeros(natoms * 3)
        f_i = forces[i_idx]
        f_j = forces[j_idx]
        dot_vj = np.sum(v_ij * f_j, axis=1)
        dot_vi = np.sum(v_ij * f_i, axis=1)
        dot_vv = np.sum(v_ij * v_ij, axis=1)
        p_ij_num = v_ij * (dot_vj / dot_vv)[:, None] - v_ij * (dot_vi / dot_vv)[:, None]
        E = np.zeros_like(coords)
        if ic_mode == 'democratic':
            norm_p = np.linalg.norm(p_ij_num, axis=1)
            valid2 = norm_p > 1e-8
            if np.sum(valid2) == 0:
                if logfile is not None:
                    logfile.write("  Warning: All force projections are zero in IC-democratic mode.\n")
                return np.zeros(natoms * 3)
            p_ij = p_ij_num[valid2] / norm_p[valid2][:, None]
            np.add.at(E, i_idx[valid2], p_ij)
            np.add.at(E, j_idx[valid2], -p_ij)
        else:
            np.add.at(E, i_idx, p_ij_num)
            np.add.at(E, j_idx, -p_ij_num)
        e_vec = E.flatten()
        n = np.linalg.norm(e_vec)
        if n > 1e-8:
            return e_vec / n
        return np.zeros_like(e_vec)
