import torch
import numpy as np
from open_ts_search.algo.ccqn.gpu_components.hessian_manager import GPUHessianManager
from open_ts_search.algo.ccqn.gpu_components.uphill_solver import GPUUphillSolver
from open_ts_search.algo.ccqn.gpu_components.prfo_solver import GPUPRFOSolver
from open_ts_search.algo.ccqn.gpu_components.trust_region import GPUTrustRegionManager
from open_ts_search.algo.ccqn.gpu_components.mode_selector import GPUModeSelector

class CCQNGPUDriver:
    """
    Core Logic Driver for CCQN-GPU.
    
    This class encapsulates the state and logic of the CCQN algorithm on GPU.
    It is decoupled from ASE's Optimizer interface, allowing for standalone usage
    where inputs (gradients, positions, energy) are provided manually.
    """
    def __init__(self, 
                 device=None,
                 cos_phi=0.5, 
                 trust_radius_uphill=0.1,
                 trust_radius_saddle_initial=0.05,
                 trust_radius_saddle_min=5e-3,
                 trust_radius_saddle_max=0.2,
                 gpu_hessian_manager=None, 
                 gpu_uphill_solver=None, 
                 gpu_prfo_solver=None,
                 gpu_trust_manager=None,
                 gpu_mode_selector=None,
                 hessian=False,
                 atoms=None,
                 # Uphill Solver Hyperparameters (Reverted momentum args)
                 uphill_max_iter=200,
                 uphill_use_slsqp=False,
                 uphill_use_alm=False,
                 uphill_use_adam=False,
                 uphill_lr=0.01,
                 debug_mode=False): 
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug_mode = debug_mode
        
        # Parameters
        self.cos_phi = cos_phi
        self.trust_radius_uphill = trust_radius_uphill
        
        # Components
        self._hessian_mgr = gpu_hessian_manager or GPUHessianManager(atoms, hessian, self.device)
        self._uphill_solver = gpu_uphill_solver or GPUUphillSolver(
            max_iter=uphill_max_iter,
            use_slsqp=uphill_use_slsqp,
            use_alm=uphill_use_alm,
            use_adam=uphill_use_adam,
            lr=uphill_lr
        )
        self._prfo_solver = gpu_prfo_solver or GPUPRFOSolver(self.device)
        self._trust_mgr = gpu_trust_manager or GPUTrustRegionManager(trust_radius_saddle_initial, 
                                                                    trust_radius_saddle_min, 
                                                                    trust_radius_saddle_max)
        self._mode_selector = gpu_mode_selector or GPUModeSelector()

        # Internal State
        self.B = None # Will be initialized in initialize()
        self.mode = 'uphill'
        self.nsteps = 0
        
        # History
        self.g_k_minus_1 = None
        self.pos_k_minus_1 = None
        self.energy_k_minus_1 = None
        self.rho = 0.0
        self.eigvals = None
        self.eigvecs = None
        
        # Trajectory Logging
        self.log_data = {
            'steps': [],       # Actual steps taken
            'e_vectors': [],   # e_vectors used
            'gradients': [],   # Gradients
            'modes': [],       # Optimization modes
            'energies': []     # Energies
        }

    def initialize(self, natoms=None, B_init=None):
        """
        Initialize Hessian and other state.
        If B_init is provided (torch tensor), it is used.
        Otherwise, initialized via HessianManager.
        """
        self.eigvals = None
        self.eigvecs = None
        if B_init is not None:
            self.B = B_init.to(self.device, dtype=torch.float64)
        else:
            # If atoms was not provided in __init__, this might fail if HessianManager needs it
            # But standard GPUHessianManager defaults to diagonal if calc is missing
            self.B = self._hessian_mgr.initialize()
        
        self.mode = 'uphill'
        self.nsteps = 0
        self._trust_mgr.reset()

    def compute_step(self, f_k, x_k, e_k, e_vec_np=None, e_vector_provider=None, logfile=None):
        """
        Compute the next step.
        """
        # 1. Prepare Data on GPU
        if isinstance(f_k, np.ndarray):
            g_k = torch.from_numpy(-f_k.flatten()).to(self.device, dtype=torch.float64)
        else:
            g_k = -f_k.flatten().to(self.device, dtype=torch.float64)
            
        if isinstance(x_k, np.ndarray):
            x_curr = torch.from_numpy(x_k.flatten()).to(self.device, dtype=torch.float64)
        else:
            x_curr = x_k.flatten().to(self.device, dtype=torch.float64)

        B_prev = self.B

        # 2. Update saddle trust radius quality (previous step)
        if self.nsteps > 0 and self.mode == 'prfo':
            s_k_prev = x_curr - self.pos_k_minus_1
            s_norm_prev = torch.norm(s_k_prev).item()
            term2 = 0.5 * (s_k_prev @ (B_prev @ s_k_prev))
            term1 = self.g_k_minus_1 @ s_k_prev
            pred_change = (term1 + term2).item()
            actual_change = e_k - self.energy_k_minus_1
            self.rho = self._calculate_rho(pred_change, actual_change)
            if logfile:
                logfile.write(f"  PRFO Quality: pred={pred_change:.4e}, actual={actual_change:.4e}, rho={self.rho:.4f}\n")
                logfile.flush()
            old_radius = self._trust_mgr.get_radius()
            new_radius = self._trust_mgr.update(self.rho, s_norm_prev, logfile)
            if logfile and new_radius != old_radius:
                logfile.write(f"  PRFO Trust: {old_radius:.4e} -> {new_radius:.4e}\n")
                logfile.flush()

        # 3. Hessian Update (Generic)
        if self.nsteps > 0:
            s_k_prev = x_curr - self.pos_k_minus_1
            y_k_prev = g_k - self.g_k_minus_1
            if torch.norm(s_k_prev) > 1e-7:
                self.B = self._hessian_mgr.update(B_prev, s_k_prev, y_k_prev, logfile, self.eigvals, self.eigvecs)

        # 4. Mode Selection
        try:
            eigvals, eigvecs = torch.linalg.eigh(self.B)
            new_mode, trust_reset, switch_reason = self._mode_selector.select(self.mode, eigvals, logfile)
            self.mode = new_mode
            if trust_reset is not None:
                self._trust_mgr.reset(trust_reset)
        except RuntimeError:
            if logfile: logfile.write("Hessian diagonalization failed. Resetting.\n")
            self.B = self._hessian_mgr.initialize()
            eigvals, eigvecs = torch.linalg.eigh(self.B)
            self.mode = 'uphill'
        
        # Store for next step
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        # 5. Prepare e_vector (early for logging)
        # If e_vec_np is not provided (e.g. mode changed to uphill), try provider
        # Also compute for PRFO mode if available, for overlap debugging
        if e_vec_np is None and e_vector_provider is not None:
             e_vec_np = e_vector_provider()

        if e_vec_np is None:
            e_vec = torch.zeros_like(g_k)
        else:
            e_vec = torch.from_numpy(e_vec_np).to(self.device, dtype=torch.float64)

        # Logging helper
        if logfile:
             # Optimization: Calculate fmax on CPU if available to avoid GPU sync
             if isinstance(f_k, np.ndarray):
                 # Ensure f_k is (N, 3) before calculation, handling flattened case if present
                 f_reshaped = f_k.reshape(-1, 3)
                 fmax = np.sqrt((f_reshaped**2).sum(axis=1).max())
             else:
                 fmax = torch.sqrt((g_k.reshape(-1, 3)**2).sum(dim=1).max()).item()
                 
             min_eig = eigvals[0].item()
             # Clarify that this is the Input state for the current step calculation
             logfile.write(f"Driver Step {self.nsteps:3d}: {self.mode.upper()} | Input State: E={e_k:.4f}, Fmax={fmax:.4f}, MinEig={min_eig:.4e}, Trust={self._trust_mgr.get_radius():.4e}\n")
             logfile.flush()
             
             # Overlap Analysis (User Request: Debug e_vector direction)
             # Note: This adds O(N^2) GPU compute and 1 CPU sync per step.
             # Only calculate if e_vector is significant.
             if torch.norm(e_vec) > 1e-6:
                 # Ensure e_vector is normalized for meaningful overlap
                 e_vec_norm = e_vec / torch.norm(e_vec)
                 # Optimization: Transfer all 3 overlaps to CPU in one sync using .tolist()
                 overlaps = torch.abs(eigvecs.T @ e_vec_norm)[:3].tolist()
                 # Ensure we have enough elements
                 while len(overlaps) < 3: overlaps.append(0.0)
                 logfile.write(f"  e_vector Overlaps (v0, v1, v2): [{overlaps[0]:.4f}, {overlaps[1]:.4f}, {overlaps[2]:.4f}]\n")

        # 6. Step Calculation
        s_k = None
        
        if self.mode == 'uphill':
            # e_vec is already prepared above
            s_k = self._uphill_solver.solve(g_k, self.B, e_vec, self.trust_radius_uphill, self.cos_phi)
            
            # Detailed Logging for Uphill
            if logfile and torch.norm(s_k) > 1e-9 and torch.norm(e_vec) > 1e-9:
                s_norm = s_k / torch.norm(s_k)
                e_norm = e_vec / torch.norm(e_vec)
                cos_sim = torch.dot(s_norm, e_norm).item()
                
                g_norm = g_k / torch.norm(g_k)
                cos_sim_g = torch.dot(s_norm, g_norm).item()
                
                logfile.write(f"  Uphill Direction Analysis:\n")
                logfile.write(f"    Cos(step, e_vec) = {cos_sim:.4f} (Target > {self.cos_phi:.4f})\n")
                logfile.write(f"    Cos(step, grad)  = {cos_sim_g:.4f}\n")
                logfile.flush()
        
        else: # PRFO
            s_k = self._prfo_solver.solve(g_k, eigvals, eigvecs, self._trust_mgr.get_radius(), logfile)

        # 7. Store History
        self.pos_k_minus_1 = x_curr.clone()
        self.g_k_minus_1 = g_k.clone()
        self.energy_k_minus_1 = e_k
        self.nsteps += 1
        
        # 6. Record Trajectory Data (Only in debug mode)
        if self.debug_mode:
            self._record_step_data(s_k, e_vec, g_k, e_k)
        
        return s_k

    def _calculate_rho(self, pred, actual):
        if abs(pred) < 1e-4: return 1.0 if abs(actual) < 1e-4 else 0.0
        return actual / pred

    def _record_step_data(self, s_k, e_vec, g_k, e_k):
        """Record step data for debugging/trajectory analysis."""
        # Helper to convert to numpy safely
        def to_np(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy().flatten()
            elif isinstance(t, np.ndarray):
                return t.flatten()
            return np.zeros_like(self.log_data['gradients'][-1]) if self.log_data['gradients'] else np.array([])

        # Handle s_k potentially being None (though unlikely if compute_step completes)
        if s_k is None:
             s_val = np.zeros_like(to_np(g_k))
        else:
             s_val = to_np(s_k)

        self.log_data['steps'].append(s_val)
        self.log_data['e_vectors'].append(to_np(e_vec))
        self.log_data['gradients'].append(to_np(g_k))
        self.log_data['modes'].append(self.mode)
        self.log_data['energies'].append(float(e_k))

    def save_trajectory_log(self, filename="optimization_log.npz"):
        """Save the recorded trajectory log to a .npz file."""
        try:
            np.savez(filename,
                     steps=np.array(self.log_data['steps']),
                     e_vectors=np.array(self.log_data['e_vectors']),
                     gradients=np.array(self.log_data['gradients']),
                     modes=np.array(self.log_data['modes']),
                     energies=np.array(self.log_data['energies']))
            print(f"Trajectory log saved to {filename}")
        except Exception as e:
            print(f"Failed to save trajectory log: {e}")
