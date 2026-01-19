import numpy as np
import warnings
from scipy.linalg import eigh
from scipy.optimize import minimize, brentq
from ase.optimize.optimize import Optimizer
from ase.geometry import find_mic
from algo.ccqn.components import _Config, _StateTracker, _HessianManager, _PRFOSolver, _TrustRegionManager
from algo.ccqn.components.ccqn_mode import CCQNModeSelector
from algo.ccqn.components.components import _DirectionProvider as CCQNDirectionProvider
from algo.ccqn.components.ccqn_uphill import CCQNUphillSolver
from algo.ccqn.components.ccqn_convergence import CCQNConvergenceChecker
from algo.ccqn.contexts.step_context import StepContext
from algo.ccqn.phases.uphill_phase import UphillPhase
from algo.ccqn.phases.prfo_phase import PRFOPhase

class CCQNOptimizer(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, master=None,
                 e_vector_method='interp', product_atoms=None, reactive_bonds=None,
                 ic_mode='democratic',
                 cos_phi=0.5, trust_radius_uphill=0.1, trust_radius_saddle_initial=0.05,
                 hessian_update='ts-bfgs', idpp_images=7, use_idpp=False, hessian=False,
                 hessian_manager=None, mode_selector=None, direction_provider=None,
                 uphill_solver=None, prfo_solver=None, trust_manager=None, convergence_checker=None):
        super().__init__(atoms, restart, logfile, trajectory, master)
        self.e_vector_method = e_vector_method
        self.hessian = hessian
        self.ic_mode = ic_mode.lower()
        if e_vector_method == 'interp':
            if product_atoms is None:
                raise ValueError("`product_atoms` must be provided for 'interp' method.")
            self.product_atoms = product_atoms
            self.idpp_images = idpp_images
            self.use_idpp = use_idpp
        elif e_vector_method == 'ic':
            if reactive_bonds is None:
                raise ValueError("`reactive_bonds` must be provided for 'ic' method.")
            if self.ic_mode not in ['democratic', 'weighted']:
                raise ValueError(f"Unknown ic_mode: '{self.ic_mode}'. Must be 'democratic' or 'weighted'.")
            self.reactive_bonds = reactive_bonds
        else:
            raise ValueError(f"Unknown e_vector_method: {e_vector_method}")
        self.cos_phi = cos_phi
        self.trust_radius_uphill = trust_radius_uphill
        self.trust_radius_saddle_initial = trust_radius_saddle_initial
        self.trust_radius_saddle = self.trust_radius_saddle_initial
        self.trust_radius_saddle_max = 0.2
        self.trust_radius_saddle_min = 5e-3
        self._config = _Config(self.e_vector_method, self.ic_mode, self.cos_phi, self.trust_radius_uphill, self.trust_radius_saddle_initial, getattr(self, 'idpp_images', 7), getattr(self, 'use_idpp', False), self.hessian)
        self._state = _StateTracker()
        self._hessian_mgr = hessian_manager or _HessianManager(self.atoms, self.hessian)
        self._mode_selector = mode_selector or CCQNModeSelector()
        self._dir_provider = direction_provider or CCQNDirectionProvider()
        self._uphill_solver = uphill_solver or CCQNUphillSolver()
        self._prfo_solver = prfo_solver or _PRFOSolver()
        self._trust_mgr = trust_manager or _TrustRegionManager(self.trust_radius_saddle_min, self.trust_radius_saddle_max)
        self._conv_checker = convergence_checker or CCQNConvergenceChecker()
        self._uphill_phase = UphillPhase(self._dir_provider, self._uphill_solver, {
            "e_vector_method": self.e_vector_method,
            "product_atoms": getattr(self, "product_atoms", None),
            "idpp_images": getattr(self, "idpp_images", 7),
            "use_idpp": getattr(self, "use_idpp", False),
            "reactive_bonds": getattr(self, "reactive_bonds", None),
            "ic_mode": self.ic_mode,
        })
        self._prfo_phase = PRFOPhase(self._prfo_solver, self._trust_mgr)
        self.B = self._initialize_hessian()
        self.mode = self._state.mode
        self.g_k_minus_1 = None
        self.pos_k_minus_1 = None
        self.energy_k_minus_1 = None
        self.rho = 0.0
    def converged(self, forces=None):
        return self._conv_checker.converged(self.atoms, forces, self.fmax, self.mode)
    def _initialize_hessian(self):
        return self._hessian_mgr.initialize()
    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()
        g_k = -f.flatten()
        x_k = self.atoms.get_positions().flatten()
        e_k = self.atoms.get_potential_energy()
        if self.nsteps > 0:
            s_k_prev = x_k - self.pos_k_minus_1
            y_k_prev = g_k - self.g_k_minus_1
            if np.linalg.norm(s_k_prev) > 1e-7:
                self.B = self._hessian_mgr.update_ts_bfgs(self.B, s_k_prev, y_k_prev, self.logfile)
        try:
            eigvals, eigvecs, new_mode, trust_reset = self._mode_selector.select(self.B, self.mode, self.logfile, self.trust_radius_saddle_initial)
            self.mode = new_mode
            if trust_reset is not None:
                self.trust_radius_saddle = trust_reset
        except Exception:
            self.logfile.write("Hessian diagonalization failed. Resetting Hessian.\n")
            self.B = self._initialize_hessian()
            eigvals, eigvecs = eigh(self.B)
            self.mode = 'uphill'
        fmax = np.sqrt((f**2).sum(axis=1).max())
        self.logfile.write(f"Step {self.nsteps:3d}: Mode='{self.mode}', E={e_k:.4f}, Fmax={fmax:.4f}, Trust(Saddle)={self.trust_radius_saddle:.4e}\n")
        ctx = StepContext(self.atoms, self.B, g_k, x_k, e_k, eigvals, eigvecs, self.trust_radius_uphill, self.trust_radius_saddle, self.cos_phi, self.e_vector_method, getattr(self, "product_atoms", None), getattr(self, "idpp_images", 7), getattr(self, "use_idpp", False), getattr(self, "reactive_bonds", None), self.ic_mode, self.pos_k_minus_1, self.g_k_minus_1, self.energy_k_minus_1, self.logfile)
        if self.mode == 'uphill':
            s_k = self._uphill_phase.run(ctx)
        else:
            s_k, self.trust_radius_saddle = self._prfo_phase.run(ctx)
        self.atoms.set_positions((x_k + s_k).reshape(-1, 3))
        self.pos_k_minus_1 = x_k
        self.g_k_minus_1 = g_k
        self.energy_k_minus_1 = e_k
