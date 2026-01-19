import torch
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.geometry import find_mic
from algo.ccqn.gpu_components.ccqn_gpu_driver import CCQNGPUDriver

class CCQNGPUOptimizer(Optimizer):
    """
    CCQN Optimizer (GPU Version)
    
    A wrapper around CCQNGPUDriver that adapts it to the ASE Optimizer interface.
    """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, master=None,
                 e_vector_method='interp', product_atoms=None, reactive_bonds=None,
                 ic_mode='democratic', cos_phi=0.5, trust_radius_uphill=0.1,
                 trust_radius_saddle_initial=0.05, hessian_update='ts-bfgs',
                 idpp_images=7, use_idpp=False, hessian=False,
                 gpu_hessian_manager=None, gpu_uphill_solver=None, gpu_prfo_solver=None,
                 driver=None):
        super().__init__(atoms, restart, logfile, trajectory, master)
        
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
                atoms=atoms
            )
        
        # Initialize Driver State
        self.driver.initialize()

    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()
        
        # Prepare inputs for driver
        x_k_np = self.atoms.get_positions()
        e_k = self.atoms.get_potential_energy()
        
        # Calculate e_vector (CPU side logic)
        # We pass it if we are already in uphill mode, otherwise we provide a callback
        e_vec_np = None
        if self.driver.mode == 'uphill':
             # Pass cached f and x_k_np to avoid re-calling get_forces/get_positions
             e_vec_np = self._calculate_e_vector_cpu(f, x_k_np)

        # Define provider callback for lazy evaluation if mode switches to uphill
        def e_vector_provider():
            return self._calculate_e_vector_cpu(f, x_k_np)

        # Execute step via Driver
        s_k = self.driver.compute_step(f, x_k_np, e_k, e_vec_np, e_vector_provider, self.logfile)

        # Apply Step (Transfer back to CPU for ASE)
        # Optimization: use driver.pos_k_minus_1 which holds the new x_curr (old x_k) on GPU?
        # No, compute_step updates pos_k_minus_1 to be x_curr at the END.
        # So pos_k_minus_1 holds x_k.
        # But we need to add s_k.
        # To avoid re-uploading x_k, we can use the driver's stored history if available.
        # However, accessing internal state like that is brittle. 
        # The re-upload of N*3 floats is negligible compared to the logic fix above.
        
        x_new_np = (torch.from_numpy(x_k_np.flatten()).to(self.device) + s_k).cpu().numpy()
        self.atoms.set_positions(x_new_np.reshape(-1, 3))

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
