import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import FIRE, LBFGS
from ase.calculators.singlepoint import SinglePointCalculator

class TSDescentOptimizer:
    """
    Fake-IRC Optimizer: Performs bidirectional geometric optimization along a given imaginary frequency mode.
    This serves as an approximate IRC (Intrinsic Reaction Coordinate) path.
    """
    def __init__(self, ts_atoms, vib_mode, constraint=None, delta=0.1, log_prefix="descent"):
        """
        Initialize the Descent Optimizer.

        Args:
            ts_atoms (ase.Atoms): The transition state structure.
            vib_mode (np.ndarray): The vibration mode vector (eigenvector of the imaginary frequency).
            constraint (ase.constraints.Constraint): Constraints to apply during optimization.
            delta (float): The initial displacement step size along the mode.
            log_prefix (str): Prefix for log and trajectory files.
        """
        self.ts_atoms = ts_atoms
        if constraint is not None:
            self.ts_atoms.set_constraint(constraint)
            
        self.vib_mode = self._normalize_mode(vib_mode)
        self.delta = delta
        self.log_prefix = log_prefix
        self.full_path = []

    def _normalize_mode(self, mode):
        norm = np.linalg.norm(mode)
        return mode / norm if norm > 1e-6 else mode

    def run(self, fmax=0.05, steps=200, optimizer_cls=FIRE):
        """
        Run the bidirectional descent optimization.

        Args:
            fmax (float): Maximum force convergence criterion.
            steps (int): Maximum number of optimization steps.
            optimizer_cls (class): ASE optimizer class (default: FIRE).

        Returns:
            list[ase.Atoms]: The full path (reverse + TS + forward).
        """
        # 1. Forward
        print(f"\n>>> [Descent] Forward Optimization (+{self.delta})...")
        atoms_fwd = self.ts_atoms.copy()
        atoms_fwd.calc = self.ts_atoms.calc
        atoms_fwd.set_positions(self.ts_atoms.get_positions() + self.delta * self.vib_mode)
        
        # Ensure constraints are preserved
        if self.ts_atoms.constraints:
            atoms_fwd.set_constraint(self.ts_atoms.constraints)
        
        opt_fwd = optimizer_cls(atoms_fwd, trajectory=f"{self.log_prefix}_fwd.traj", logfile=f"{self.log_prefix}_fwd.log")
        opt_fwd.run(fmax=fmax, steps=steps)
        
        # Read back the trajectory
        if os.path.exists(f"{self.log_prefix}_fwd.traj"):
            path_fwd = read(f"{self.log_prefix}_fwd.traj", index=':')
        else:
            path_fwd = [atoms_fwd]

        # 2. Reverse
        print(f"\n>>> [Descent] Reverse Optimization (-{self.delta})...")
        atoms_rev = self.ts_atoms.copy()
        atoms_rev.calc = self.ts_atoms.calc
        atoms_rev.set_positions(self.ts_atoms.get_positions() - self.delta * self.vib_mode)
        
        if self.ts_atoms.constraints:
            atoms_rev.set_constraint(self.ts_atoms.constraints)
        
        opt_rev = optimizer_cls(atoms_rev, trajectory=f"{self.log_prefix}_rev.traj", logfile=f"{self.log_prefix}_rev.log")
        opt_rev.run(fmax=fmax, steps=steps)
        
        if os.path.exists(f"{self.log_prefix}_rev.traj"):
            path_rev = read(f"{self.log_prefix}_rev.traj", index=':')
        else:
            path_rev = [atoms_rev]

        # 3. Combine
        ts_snap = self.ts_atoms.copy()
        # Create a SinglePointCalculator to preserve energy/forces if original calculator is expensive or transient
        # However, if ts_atoms already has results, we try to use them.
        try:
            energy = self.ts_atoms.get_potential_energy()
            forces = self.ts_atoms.get_forces()
            ts_snap.calc = SinglePointCalculator(ts_snap, energy=energy, forces=forces)
        except Exception:
            # If calculation not available, just keep the calculator or leave it
            pass

        # Reverse path should be reversed to go from end to TS
        self.full_path = path_rev[::-1] + [ts_snap] + path_fwd
        
        write(f"{self.log_prefix}_full.traj", self.full_path)
        write(f"{self.log_prefix}_full.xyz", self.full_path)
        print(f">>> [Descent] Done. Full path length: {len(self.full_path)}")
        
        return self.full_path

def plot_descent_profile(full_path, title="Reaction_Profile", save_path=None):
    """
    Plot the energy profile of the Fake-IRC path.
    
    Args:
        full_path (list[Atoms]): The full path list of Atoms objects.
        title (str): Chart title.
        save_path (str): Path to save the image (e.g., "profile.png").
    """
    if not full_path or len(full_path) < 3:
        print("Path too short to plot.")
        return

    # 1. Extract energies
    try:
        energies = np.array([atoms.get_potential_energy() for atoms in full_path])
    except Exception as e:
        print(f"Error getting energies: {e}")
        return

    # 2. Calculate cumulative geometric distance
    dist = [0.0]
    for i in range(1, len(full_path)):
        d = np.linalg.norm(full_path[i].positions - full_path[i-1].positions)
        dist.append(dist[-1] + d)
    dist = np.array(dist)

    # 3. Process data
    ts_idx = np.argmax(energies)
    
    # Use Reactant (start) as zero reference
    base_energy = energies[0]
    rel_energies = energies - base_energy
    
    E_barrier_fwd = rel_energies[ts_idx] # Forward barrier
    E_reaction = rel_energies[-1]        # Reaction energy (Delta E)
    E_barrier_rev = rel_energies[ts_idx] - rel_energies[-1] # Reverse barrier

    # 4. Plot
    plt.figure(figsize=(8, 5))
    
    plt.plot(dist, rel_energies, 'o-', color='tab:blue', markersize=4, linewidth=1.5, label='Path')
    
    # Mark special points
    plt.scatter(dist[0], rel_energies[0], color='green', s=100, label='IS', zorder=5)
    plt.scatter(dist[-1], rel_energies[-1], color='orange', s=100, label='FS', zorder=5)
    plt.scatter(dist[ts_idx], rel_energies[ts_idx], color='red', marker='*', s=200, label='TS', zorder=5)

    # Annotations
    plt.text(dist[ts_idx], rel_energies[ts_idx] + 0.05, f"$E_a={E_barrier_fwd:.2f}$ eV", 
             ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
    
    plt.text(dist[-1], rel_energies[-1] - 0.1, f"$\Delta E={E_reaction:.2f}$ eV", 
             ha='right', va='top', fontsize=10, color='orange')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.xlabel("Cumulative Distance ($\AA$)")
    plt.ylabel("Relative Energy (eV)")
    plt.title(f"{title}\nForward Barrier: {E_barrier_fwd:.3f} eV | Reverse: {E_barrier_rev:.3f} eV")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    
    # plt.show() # Avoid showing blocking window in non-interactive environments
