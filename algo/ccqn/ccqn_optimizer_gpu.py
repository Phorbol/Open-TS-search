import torch
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.geometry import find_mic
from algo.ccqn.gpu_components.ccqn_gpu_driver import CCQNGPUDriver
from algo.shared.reproducibility import set_deterministic

class CCQNGPUOptimizer(Optimizer):
    """
    CCQN Optimizer (GPU Version)
    
    A wrapper around CCQNGPUDriver that adapts it to the ASE Optimizer interface.
    
    Args:
        atoms: The ASE Atoms object to optimize.
        restart: Filename for restart file.
        logfile: Filename for log file.
        trajectory: Filename for trajectory file.
        master: Master process for parallel execution.
        e_vector_method: Method for calculating e_vector ('interp', 'ic', etc.).
        product_atoms: Product atoms for interpolation.
        reactive_bonds: Reactive bonds for IC mode.
        ic_mode: IC mode ('democratic', etc.).
        cos_phi: Cosine of the cone angle for uphill step.
        trust_radius_uphill: Trust radius for uphill step.
        trust_radius_saddle_initial: Initial trust radius for saddle step.
        hessian_update: Hessian update method.
        idpp_images: Number of IDPP images.
        use_idpp: Whether to use IDPP.
        hessian: Whether to use Hessian.
        gpu_hessian_manager: Custom GPU Hessian Manager.
        gpu_uphill_solver: Custom GPU Uphill Solver.
        gpu_prfo_solver: Custom GPU PRFO Solver.
        driver: Custom CCQNGPUDriver.
        uphill_max_iter: Max iterations for Uphill PGD solver.
        uphill_use_slsqp: Whether to use SLSQP (CPU) for uphill step instead of PGD.
        uphill_use_alm: Whether to use ALM (GPU) for uphill step instead of PGD/SLSQP.
        uphill_use_adam: Whether to use Projected Adam (GPU) for uphill step instead of PGD/SLSQP.
        uphill_lr: Learning rate for Uphill Solver (e.g. Adam).
    """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, master=None,
                 e_vector_method='interp', product_atoms=None, reactive_bonds=None,
                 ic_mode='democratic', cos_phi=0.5, trust_radius_uphill=0.1,
                 trust_radius_saddle_initial=0.05, hessian_update='ts-bfgs',
                 idpp_images=7, use_idpp=False, hessian=False,
                 gpu_hessian_manager=None, gpu_uphill_solver=None, gpu_prfo_solver=None,
                 driver=None,
                 # Uphill Solver Hyperparameters (Reverted momentum)
                 uphill_max_iter=200,
                 uphill_use_slsqp=False,
                 uphill_use_alm=False,
                 uphill_use_adam=False,
                 uphill_lr=0.01,
                 debug_mode=False):
        super().__init__(atoms, restart, logfile, trajectory, master)
        
        # Enforce deterministic behavior
        set_deterministic(seed=42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.logfile:
            self.logfile.write(f"CCQN-GPU (Driver-based) initialized on device: {self.device}\n")

        # Parameters
        self.e_vector_method = e_vector_method
        self.hessian = hessian
        self.ic_mode = ic_mode.lower()
        self.product_atoms = product_atoms
        self.reactive_bonds = reactive_bonds
        self.idpp_images = idpp_images
        self.use_idpp = use_idpp
        
        # Initialize Driver
        if driver:
            self.driver = driver
        else:
            self.driver = CCQNGPUDriver(
                device=self.device,
                cos_phi=cos_phi,
                trust_radius_uphill=trust_radius_uphill,
                trust_radius_saddle_initial=trust_radius_saddle_initial,
                gpu_hessian_manager=gpu_hessian_manager,
                gpu_uphill_solver=gpu_uphill_solver,
                gpu_prfo_solver=gpu_prfo_solver,
                hessian=hessian,
                atoms=atoms,
                uphill_max_iter=uphill_max_iter,
                uphill_use_slsqp=uphill_use_slsqp,
                uphill_use_alm=uphill_use_alm,
                uphill_use_adam=uphill_use_adam,
                uphill_lr=uphill_lr,
                debug_mode=debug_mode
            )
        
        # Initialize Driver State
        self.driver.initialize()

    def converged(self, forces=None):
        """Check if the optimization has converged."""
        if forces is None:
            forces = self.atoms.get_forces()
        elif hasattr(forces, 'ndim') and forces.ndim == 1:
            forces = forces.reshape(-1, 3)
            
        # Calculate fmax
        fmax_val = np.sqrt((forces ** 2).sum(axis=1).max())
        
        # Check standard fmax convergence
        # self.fmax is set by Optimizer.run(), default to 0.05 if not set
        threshold = getattr(self, 'fmax', 0.05)
        force_converged = fmax_val < threshold
        
        # Check mode convergence (must be in PRFO mode)
        # The driver maintains the mode state
        mode_converged = (self.driver.mode == 'prfo')
        
        return force_converged and mode_converged

    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()
        
        # Prepare inputs for driver
        x_k_np = self.atoms.get_positions()
        e_k = self.atoms.get_potential_energy()
        
        # Determine if we can use GPU e_vector calculation
        # This requires x_k and f to be available on GPU
        # Currently ASE provides them as numpy, but we can convert them once
        
        # use_gpu_evec = (self.e_vector_method == 'ic') # Only IC method implemented for GPU for now
        # Reverting GPU e_vector: always use CPU path for now as requested
        use_gpu_evec = False 
        
        e_vec_np = None
        
        if not use_gpu_evec:
            # Legacy CPU path
            if self.driver.mode == 'uphill':
                 e_vec_np = self._calculate_e_vector_cpu(f, x_k_np)
            
            def e_vector_provider():
                return self._calculate_e_vector_cpu(f, x_k_np)
                
            # Execute step via Driver
            s_k = self.driver.compute_step(f, x_k_np, e_k, e_vec_np, e_vector_provider, self.logfile)

        else:
            # GPU Path (Disabled/Reverted)
            pass

        # Apply Step
        x_new_np = (torch.from_numpy(x_k_np.flatten()).to(self.device) + s_k).cpu().numpy()
        self.atoms.set_positions(x_new_np.reshape(-1, 3))

    def save_trajectory_log(self, filename="optimization_log.npz"):
        """Save the optimization trajectory log from the driver."""
        if hasattr(self, 'driver'):
            self.driver.save_trajectory_log(filename)

    def _calculate_e_vector_cpu(self, forces=None, coords=None):
         # CPU Geometric Logic (relying on ASE)
         if self.e_vector_method == 'ic': 
              if coords is None: coords = self.atoms.get_positions() 
              if forces is None: forces = self.atoms.get_forces() 
              
              cell = self.atoms.get_cell() 
              pbc = self.atoms.get_pbc() 
              if not self.reactive_bonds: return np.zeros(len(coords)*3)
              bonds = np.array(self.reactive_bonds, dtype=int) 
              if len(bonds) == 0: return np.zeros(len(coords)*3) 

              i_idx, j_idx = bonds[:,0], bonds[:,1] 
              raw_v_ij = coords[j_idx] - coords[i_idx] 
              v_ij, _ = find_mic(raw_v_ij, cell, pbc) 
              norm_v = np.linalg.norm(v_ij, axis=1) 
              valid = norm_v > 1e-8 
              if not np.any(valid): return np.zeros(len(coords)*3) 
              
              v_ij = v_ij[valid]; i_idx = i_idx[valid]; j_idx = j_idx[valid] 
              f_i, f_j = forces[i_idx], forces[j_idx] 
              dot_vj = np.sum(v_ij * f_j, axis=1) 
              dot_vi = np.sum(v_ij * f_i, axis=1) 
              dot_vv = np.sum(v_ij * v_ij, axis=1) 
              p_ij_num = v_ij * (dot_vj/dot_vv)[:,None] - v_ij * (dot_vi/dot_vv)[:,None] 
              
              E = np.zeros_like(coords) 
              if self.ic_mode == 'democratic': 
                  norm_p = np.linalg.norm(p_ij_num, axis=1) 
                  valid2 = norm_p > 1e-8 
                  if np.sum(valid2) > 0: 
                      p_ij = p_ij_num[valid2] / norm_p[valid2][:,None] 
                      np.add.at(E, i_idx[valid2], p_ij) 
                      np.add.at(E, j_idx[valid2], -p_ij) 
              else: 
                  np.add.at(E, i_idx, p_ij_num) 
                  np.add.at(E, j_idx, -p_ij_num) 
                  
              e = E.flatten() 
              n = np.linalg.norm(e) 
              return e/n if n > 1e-8 else e 
         return np.zeros(len(self.atoms)*3)

    # def _calculate_e_vector_gpu(self, forces_flat, coords_flat):
    #     """
    #     GPU implementation of IC-based e_vector calculation.
    #     (Reverted/Commented out to restore CPU-only behavior for geometry logic)
    #     """
    #     pass
