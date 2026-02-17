import numpy as np
import warnings
from ase import Atoms
from typing import List

def robust_interpolate(start_atoms: Atoms, end_atoms: Atoms, nimages: int) -> List[Atoms]:
    scaled_start = start_atoms.get_scaled_positions()
    scaled_end = end_atoms.get_scaled_positions()
    delta_scaled = scaled_end - scaled_start
    delta_scaled_mic = delta_scaled - np.round(delta_scaled)
    path = [start_atoms.copy()]
    total_steps = nimages + 1
    for i in range(1, total_steps):
        alpha = i / total_steps
        current_scaled_pos = scaled_start + alpha * delta_scaled_mic
        image = start_atoms.copy()
        image.set_scaled_positions(current_scaled_pos)
        path.append(image)
    path.append(end_atoms.copy())
    return path

class Vectorized_ASE_IDPPSolver:
    def __init__(self, images: list[Atoms]):
        self.images = [img.copy() for img in images]
        self.cell = images[0].get_cell()
        self.inv_cell = np.linalg.inv(self.cell)
        self.natoms = len(images[0])
        self.nimages = len(images) - 2
        start_atoms = images[0]
        end_atoms = images[-1]
        d_start = start_atoms.get_all_distances(mic=True)
        d_end = end_atoms.get_all_distances(mic=True)
        factors = np.linspace(0, 1, self.nimages + 2)[1:-1]
        self.target_dists = d_start[None, :, :] + factors[:, None, None] * (d_end - d_start)[None, :, :]
        initial_dists = np.array([img.get_all_distances(mic=True) for img in images[1:-1]])
        avg_dists = (self.target_dists + initial_dists) / 2.0
        self.weights = 1.0 / (avg_dists**4 + np.eye(self.natoms)[None, :, :] * 1e-12)
        self.init_coords = np.array([img.get_positions() for img in images])
    def _get_mic_vectors_all_images(self, coords_stack: np.ndarray) -> np.ndarray:
        scaled_coords = np.dot(coords_stack, self.inv_cell)
        diffs = scaled_coords[:, :, None, :] - scaled_coords[:, None, :, :]
        diffs -= np.round(diffs)
        return np.dot(diffs, self.cell)
    def _get_funcs_and_forces(self, coords_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = self._get_mic_vectors_all_images(coords_stack)
        trial_dist = np.linalg.norm(vectors, axis=3)
        dist_plus_eye = trial_dist + np.eye(self.natoms)[None, :, :]
        diff_dist_sq = (trial_dist - self.target_dists)**2
        funcs = 0.5 * np.sum(self.weights * diff_dist_sq, axis=(1, 2))
        aux_mat = self.weights * (trial_dist - self.target_dists) / dist_plus_eye
        forces_on_atoms = np.einsum('...ij,...ijk->...ik', aux_mat, vectors)
        return funcs, -2 * forces_on_atoms
    def _get_inter_image_mic_vectors(self, coords_after, coords_before):
        delta_cart = coords_after - coords_before
        delta_scaled = np.dot(delta_cart, self.inv_cell)
        delta_scaled_mic = delta_scaled - np.round(delta_scaled)
        return np.dot(delta_scaled_mic, self.cell)
    def _get_total_forces(self, coords: np.ndarray, true_forces: np.ndarray, spring_const: float) -> np.ndarray:
        vecs_after = self._get_inter_image_mic_vectors(coords[2:], coords[1:-1])
        vecs_before = self._get_inter_image_mic_vectors(coords[1:-1], coords[:-2])
        vecs_after_flat = vecs_after.reshape(self.nimages, -1)
        vecs_before_flat = vecs_before.reshape(self.nimages, -1)
        len_after = np.linalg.norm(vecs_after_flat, axis=1)
        len_before = np.linalg.norm(vecs_before_flat, axis=1)
        unit_vecs_after = vecs_after_flat / (len_after[:, None] + 1e-12)
        unit_vecs_before = vecs_before_flat / (len_before[:, None] + 1e-12)
        tangents = unit_vecs_after + unit_vecs_before
        tangent_norms = np.linalg.norm(tangents, axis=1)
        tangents /= (tangent_norms[:, None] + 1e-12)
        spring_force_magnitudes = spring_const * (len_after - len_before)
        spring_forces = tangents * spring_force_magnitudes[:, None]
        true_forces_flat = true_forces.reshape(self.nimages, -1)
        true_force_parallel_mags = np.einsum('...i,...i->...', true_forces_flat, tangents)
        true_force_parallel = tangents * true_force_parallel_mags[:, None]
        total_forces_flat = true_forces_flat - true_force_parallel + spring_forces
        return total_forces_flat.reshape(self.nimages, self.natoms, 3)
    def run(self, maxiter=500, tol=1e-4, gtol=1e-3, step_size=0.05, max_disp=0.05, spring_const=5.0):
        coords = self.init_coords.copy()
        old_funcs = np.zeros(self.nimages)
        print("Starting FULLY VECTORIZED IDPP optimization...")
        for n in range(maxiter):
            funcs, true_forces = self._get_funcs_and_forces(coords[1:-1])
            tot_forces = self._get_total_forces(coords, true_forces, spring_const)
            displacements = step_size * tot_forces
            norms = np.linalg.norm(displacements, axis=2)
            scale = np.minimum(1.0, max_disp / (norms + 1e-12))
            displacements *= scale[:, :, np.newaxis]
            coords[1:-1] += displacements
            max_force_component = np.max(np.abs(tot_forces))
            func_change = np.sum(np.abs(funcs - old_funcs))
            if (n > 5 and func_change < tol and max_force_component < gtol):
                print(f"IDPP converged in {n+1} steps.")
                break
            old_funcs = funcs
            if n % 20 == 0:
                print(f"Step {n:4d}: Max Force = {max_force_component:.4f}, Func Change = {func_change:.4g}")
        else:
            warnings.warn("IDPP did not converge within maxiter.", UserWarning)
        final_images = [self.images[0].copy()]
        for i in range(self.nimages):
            img = self.images[i+1].copy()
            img.set_positions(coords[i+1])
            final_images.append(img)
        final_images.append(self.images[-1].copy())
        return final_images
    @classmethod
    def from_endpoints(cls, start: Atoms, end: Atoms, nimages: int):
        initial_images = robust_interpolate(start, end, nimages)
        return cls(initial_images)
