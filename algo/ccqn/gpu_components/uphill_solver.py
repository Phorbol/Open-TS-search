import torch
import numpy as np
from scipy.optimize import minimize
from algo.ccqn.gpu_components.gpu_kernels import jit_solve_pgd
from algo.ccqn.gpu_components.constrained_opt import ConstrainedOptimizer

class GPUUphillSolver:
    def __init__(self, max_iter=200, lr=0.01, momentum=0.8, tol=1e-6, use_slsqp=False, use_alm=False, use_adam=False):
        self.max_iter = max_iter
        self.lr = lr
        self.momentum = momentum
        self.tol = tol
        self.use_slsqp = use_slsqp
        self.use_alm = use_alm
        self.use_adam = use_adam

    def solve(self, g: torch.Tensor, B: torch.Tensor, e_vec: torch.Tensor, 
              trust_radius_uphill: float, cos_phi: float) -> torch.Tensor:
        """
        Solve the uphill step problem.
        
        Args:
            g: Gradient tensor (GPU)
            B: Hessian tensor (GPU)
            e_vec: Eigenvector tensor (GPU)
            trust_radius_uphill: Trust region radius
            cos_phi: Cone constraint cosine
            
        Returns:
            Step tensor s (GPU)
        """
        # Strategy 1: Use Robust SLSQP (CPU offload)
        if self.use_slsqp:
            return self._solve_slsqp(g, B, e_vec, trust_radius_uphill, cos_phi)
            
        # Strategy 2: Use Native GPU ALM (Robust + Fast)
        if self.use_alm:
            return self._solve_alm(g, B, e_vec, trust_radius_uphill, cos_phi)

        # Strategy 3: Use Projected Adam (GPU Native + Momentum)
        if self.use_adam:
            return self._solve_projected_adam(g, B, e_vec, trust_radius_uphill, cos_phi)

        # Strategy 4: Use Fast PGD (GPU Native, JIT)
        return self._solve_pgd(g, B, e_vec, trust_radius_uphill, cos_phi)

    def _solve_pgd(self, g, B, e_vec, trust_radius_uphill, cos_phi):
        """Native GPU PGD Solver (JIT, Fixed LR)"""
        # Initial guess: along e_vec
        s0 = e_vec * trust_radius_uphill
        
        # Call JIT kernel
        s = jit_solve_pgd(s0, g, B, e_vec, 
                          float(trust_radius_uphill), 
                          float(cos_phi), 
                          self.max_iter)
        return s

    def _solve_projected_adam(self, g, B, e_vec, trust_radius_uphill, cos_phi):
        """
        Projected Adam Solver.
        Uses torch.optim.Adam to minimize the quadratic model, projecting onto constraints at each step.
        """
        # Initial guess: along e_vec
        s = (e_vec * trust_radius_uphill).clone().detach().requires_grad_(True)
        
        # Initialize Adam
        # Using a relatively high LR because we are solving a local quadratic model
        optimizer = torch.optim.Adam([s], lr=self.lr)
        
        # Projection Helper
        def project(s_curr):
            with torch.no_grad():
                # 1. Trust Region Projection (Sphere)
                norm_s = torch.norm(s_curr)
                if norm_s > 1e-9:
                     s_curr.data.mul_(trust_radius_uphill / norm_s)
                else:
                     s_curr.data.copy_(e_vec * trust_radius_uphill)
                
                # 2. Cone Projection
                proj_len = torch.dot(s_curr, e_vec)
                target_proj = trust_radius_uphill * cos_phi
                
                if proj_len < target_proj:
                    s_par = proj_len * e_vec
                    s_perp = s_curr - s_par
                    norm_perp = torch.norm(s_perp)
                    
                    # Target perpendicular length: sqrt(tr^2 - target_proj^2)
                    # FIX: Ensure calculation happens in Tensor space
                    tr_sq = torch.tensor(trust_radius_uphill**2, device=s_curr.device, dtype=s_curr.dtype)
                    tp_sq = torch.tensor(target_proj**2, device=s_curr.device, dtype=s_curr.dtype)
                    target_perp = torch.sqrt(torch.abs(tr_sq - tp_sq)) # abs to prevent nan from small errors
                    
                    if norm_perp > 1e-9:
                        # Reassemble: Fixed parallel + Scaled perpendicular
                        s_curr.data.copy_((target_proj * e_vec) + (target_perp * (s_perp / norm_perp)))
                    else:
                        # Fallback to axis
                        s_curr.data.copy_(e_vec * trust_radius_uphill)
        
        # Optimization Loop
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            
            # Quadratic Objective: 0.5 * s.T @ B @ s + g.T @ s
            # Note: We minimize this subject to constraints
            loss = 0.5 * torch.dot(s, B @ s) + torch.dot(g, s)
            loss.backward()
            
            optimizer.step()
            
            # Project back to feasible region
            project(s)
            
        return s.detach()

    def _solve_alm(self, g, B, e_vec, trust_radius_uphill, cos_phi):
        """Native GPU Augmented Lagrangian Method Solver"""
        
        # Initial guess: along e_vec
        s0 = e_vec * trust_radius_uphill
        
        # Objective: 0.5 * s.T @ B @ s + g.T @ s
        def objective(s):
            return 0.5 * torch.dot(s, B @ s) + torch.dot(g, s)
            
        # Constraint 1 (Equality): ||s||^2 - tr^2 = 0
        def eq_con(s):
            return torch.dot(s, s) - trust_radius_uphill**2
            
        # Constraint 2 (Inequality): e.T @ s - cos_phi * tr >= 0
        # ALM Lib expects fun(x) >= 0 convention for inequality
        def ineq_con(s):
            return torch.dot(e_vec, s) - cos_phi * trust_radius_uphill
            
        constraints = [
            {'type': 'eq', 'fun': eq_con},
            {'type': 'ineq', 'fun': ineq_con}
        ]
        
        # Solve using ALM
        s_final = ConstrainedOptimizer.minimize_alm(
            objective, 
            s0, 
            constraints,
            max_outer_iter=10,  # Usually converges in 3-5 outer steps
            max_inner_iter=20,  # LBFGS steps per outer step
            rho=10.0,
            tol=1e-5
        )
        
        return s_final

    def _solve_slsqp(self, g, B, e_vec, trust_radius_uphill, cos_phi):
        """CPU SLSQP Solver (Robust)"""
        device = g.device
        dtype = g.dtype
        
        # 1. Offload to CPU
        g_np = g.detach().cpu().numpy().flatten()
        B_np = B.detach().cpu().numpy()
        e_vec_np = e_vec.detach().cpu().numpy().flatten()
        
        # 2. Setup SLSQP
        s0 = e_vec_np * trust_radius_uphill
        
        def objective(s):
            # 0.5 * s.T @ B @ s + g.T @ s
            return 0.5 * s.dot(B_np.dot(s)) + g_np.dot(s)
            
        def jac_objective(s):
            # B @ s + g
            return B_np.dot(s) + g_np
            
        def eq_constraint_fun(s):
            # ||s||^2 - tr^2 = 0
            return s.dot(s) - trust_radius_uphill**2
            
        def jac_eq_constraint(s):
            return 2 * s
            
        def ineq_constraint_fun(s):
            # e.T @ s - cos_phi * tr >= 0
            return e_vec_np.dot(s) - cos_phi * trust_radius_uphill
            
        def jac_ineq_constraint(s):
            return e_vec_np
            
        constraints = [
            {'type': 'eq', 'fun': eq_constraint_fun, 'jac': jac_eq_constraint},
            {'type': 'ineq', 'fun': ineq_constraint_fun, 'jac': jac_ineq_constraint}
        ]
        
        # 3. Solve
        # Use a tight tolerance to ensure high quality steps
        res = minimize(objective, s0, jac=jac_objective, 
                       constraints=constraints, method='SLSQP', 
                       options={'maxiter': 500, 'ftol': 1e-9})
        
        if res.success:
            s_final = res.x
        else:
            # Fallback if SLSQP fails (rare): project s0
            s_final = s0
            
        # 4. Upload to GPU
        return torch.from_numpy(s_final).to(device=device, dtype=dtype)
