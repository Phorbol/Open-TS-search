import torch
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.geometry import find_mic
from algo.ccqn.gpu_components.ccqn_gpu_driver import CCQNGPUDriver
from algo.shared.reproducibility import set_deterministic
from algo.ccqn.gpu_components.e_vector_generator import EVectorGenerator

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
                 ic_mode='democratic', ic_gamma=None, cos_phi=0.5, trust_radius_uphill=0.1,
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
                 debug_mode=False,
                 # New Parameter for Robust E-Vector
                 use_robust_e_vector=False):
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

        # Resolve ic_gamma for unified e-vector calculation
        if ic_gamma is not None:
            self.ic_gamma = float(ic_gamma)
        else:
            # Backward compatibility: Map mode to gamma
            if self.ic_mode == 'democratic':
                self.ic_gamma = 1.0
            elif self.ic_mode == 'weighted':
                self.ic_gamma = 0.0
            else:
                # Default to democratic if unknown, matching default ic_mode
                self.ic_gamma = 1.0
                
        self.product_atoms = product_atoms
        self.reactive_bonds = reactive_bonds
        self.idpp_images = idpp_images
        self.use_idpp = use_idpp
        
        # Initialize E-Vector Generator (Optional)
        self.e_vector_generator = None
        # Enable if explicitly requested OR if reactive_bonds has 3-element tuples (new format)
        has_new_format = reactive_bonds and len(reactive_bonds) > 0 and len(reactive_bonds[0]) == 3
        if use_robust_e_vector or has_new_format:
            if reactive_bonds:
                self.e_vector_generator = EVectorGenerator(reactive_bonds)
                if self.logfile:
                    self.logfile.write("CCQN: Enabled Robust Markovian E-Vector Generator.\n")
        
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
            # Legacy CPU path or New Generator path
            if self.driver.mode == 'uphill':
                 if self.e_vector_generator:
                     e_vec_np = self.e_vector_generator.compute(self.atoms, f)
                 else:
                     e_vec_np = self._calculate_e_vector_cpu(f, x_k_np)
            
            def e_vector_provider():
                if self.e_vector_generator:
                    return self.e_vector_generator.compute(self.atoms, f)
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
              
              # --- DEBUG: Bond Analysis ---
              # Only print if debug_mode is enabled in the driver
              if hasattr(self, 'driver') and getattr(self.driver, 'debug_mode', False):
                  # Calculate p_k norms for scale analysis
                  p_norms = np.linalg.norm(p_ij_num, axis=1)
                  max_p = np.max(p_norms) if len(p_norms) > 0 else 0.0
                  min_p = np.min(p_norms) if len(p_norms) > 0 else 0.0
                  R_val = max_p / (min_p + 1e-8)

                  # Check current mode from driver
                  current_mode = getattr(self.driver, 'mode', 'unknown').upper()
                  print(f"\n[CCQN Debug] Step {getattr(self.driver, 'nsteps', '?')} ({current_mode}) | E-Vector Construction (R={R_val:.2f}):")
                  for k, (i, j) in enumerate(zip(i_idx, j_idx)):
                      # Current Bond Length
                      dist = np.linalg.norm(v_ij[k])
                      v_hat = v_ij[k] / dist
                      
                      # Force Projection: (f_j - f_i) . v_hat
                      # Positive = Repulsive/Expanding Force (Downhill is to expand)
                      # Negative = Attractive/Restoring Force (Downhill is to contract)
                      f_diff = forces[j] - forces[i]
                      f_proj = np.sum(f_diff * v_hat)
                      
                      # E-vector logic:
                      # CCQN constructs E to oppose the force? 
                      # p_ij aligns with v_hat if f_proj > 0.
                      # E contribution: E_j += p_ij, E_i -= p_ij => Delta(r_j - r_i) ~ 2 * p_ij
                      # So if f_proj > 0 (Repulsive), E points to Expand.
                      # Wait! Standard logic: E aligns with force? 
                      # Let's check p_ij sign: p_ij ~ v * (f_diff . v) / v^2
                      # So p_ij is PARALLEL to force difference.
                      # So E vector points DOWNHILL along the force.
                      # BUT! The Solver minimizes Energy.
                      # To climb, we usually need E to point UPHILL?
                      # Let's verify observation: "Normally E points along mode".
                      # If E follows force, it points to reactant well.
                      # Constraint: s . e > 0. 
                      # If e points downhill, s must have downhill component.
                      # This implies CCQN assumes e points to PRODUCT?
                      # Actually: In standard CCQN/P-RFO, the mode is the Hessian eigenvector with negative curvature.
                      # Here we construct it from forces. 
                      # Empirically: If bond is stretched, force pulls back (Negative f_proj).
                      # p_ij opposes v_hat. E points to CONTRACT.
                      # This seems to contradict "breaking bond".
                      # UNLESS: The force used here is the *negative* of the gradient?
                      # ASE get_forces() returns -Gradient. 
                      # So if atoms attract, Force is negative? No.
                      # Force on J from I is attractive -> points to I (-v_ij).
                      # Force on I from J is attractive -> points to J (+v_ij).
                      # f_j - f_i = (-v) - (v) = -2v.
                      # f_proj is Negative.
                      # p_ij is Negative (opposes v).
                      # E_j += p_ij (moves J towards I). E_i -= p_ij (moves I towards J).
                      # So E points to CONTRACT.
                      # This means s . e > 0 enforces CONTRACTION?
                      # This suggests for bond breaking, we might need to FLIP e_vector 
                      # OR the force is actually repulsive at the start?
                      
                      status = "EXPANDING (Repulsive Force)" if f_proj > 0 else "CONTRACTING (Restoring Force)"
                      print(f"  Bond {i}-{j}: L={dist:.4f} A, F_proj={f_proj:.4f} eV/A, |p_k|={p_norms[k]:.4f} -> {status}")
              # ----------------------------

              E = np.zeros_like(coords) 
              
              # Unified Gamma Approach: p_ij = p_ij_num / ||p_ij_num||^gamma
              # gamma=1 -> Democratic (Normalized)
              # gamma=0 -> Weighted (Raw Force Projection)
              norm_p = np.linalg.norm(p_ij_num, axis=1) 
              valid2 = norm_p > 1e-8 
              
              if np.sum(valid2) > 0: 
                  p_subset = p_ij_num[valid2]
                  n_subset = norm_p[valid2]
                  
                  if abs(self.ic_gamma) < 1e-6:
                      # Weighted (gamma=0): scale = 1
                      p_ij = p_subset
                  elif abs(self.ic_gamma - 1.0) < 1e-6:
                      # Democratic (gamma=1): scale = 1/norm
                      p_ij = p_subset / n_subset[:, None]
                  else:
                      # General Gamma: scale = 1/(norm^gamma)
                      scale = 1.0 / np.power(n_subset, self.ic_gamma)
                      p_ij = p_subset * scale[:, None]

                  np.add.at(E, i_idx[valid2], p_ij) 
                  np.add.at(E, j_idx[valid2], -p_ij) 
              
              # --- DEBUG: E-Vector Effect Analysis ---
              if hasattr(self, 'driver') and getattr(self.driver, 'debug_mode', False):
                   print(f"  E-Vector Net Effect on Bonds:")
                   e_temp = E.flatten()
                   # Predict step along E
                   step_test = e_temp * 0.1 # Small step
                   step_test_Rs = step_test.reshape(-1, 3)
                   
                   for k, (i, j) in enumerate(zip(i_idx, j_idx)):
                       r_i, r_j = coords[i], coords[j]
                       dr = r_j - r_i
                       dist_old = np.linalg.norm(dr)
                       
                       # New positions
                       r_i_new = r_i + step_test_Rs[i]
                       r_j_new = r_j + step_test_Rs[j]
                       dist_new = np.linalg.norm(r_j_new - r_i_new)
                       
                       delta = dist_new - dist_old
                       effect = "LENGTHENING" if delta > 0 else "SHORTENING"
                       print(f"    Bond {i}-{j}: E-vec points to {effect} (Delta ~ {delta:.1e})")
              # ---------------------------------------

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
