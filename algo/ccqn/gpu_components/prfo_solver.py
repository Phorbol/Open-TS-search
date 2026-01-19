import torch
import numpy as np
from scipy.optimize import brentq

class GPUPRFOSolver:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def solve(self, g, eigvals, eigvecs, trust_radius_saddle, logfile=None):
        tol = 1e-15
        
        # Matrix ops on GPU
        g_tilde = eigvecs.T @ g
        g_tilde_max, g_tilde_min = g_tilde[:1], g_tilde[1:]
        ev_max, ev_min = eigvals[:1], eigvals[1:]
        
        # Unconstrained step
        # Note: pinv on GPU
        s_max = -torch.linalg.pinv(torch.diag(ev_max), rcond=tol) @ g_tilde_max
        s_min = -torch.linalg.pinv(torch.diag(ev_min), rcond=tol) @ g_tilde_min
        s_unc = eigvecs @ torch.cat([s_max, s_min])
        
        norm_s_unc = torch.norm(s_unc)
        if norm_s_unc <= trust_radius_saddle:
            return s_unc
            
        # Trust Region Scaling
        # Optimization: Move scalar data to CPU once for Secular Equation solver
        # This avoids CPU-GPU ping-pong in brentq loop
        
        try: 
            # Prepare CPU data for secular equation
            # g_tilde_max is size 1, g_tilde_min is size N-1
            # ev_max is size 1, ev_min is size N-1
            
            # Combine them back to full vectors for secular equation logic
            # Structure: [max_mode, min_modes...]
            # Note: g_tilde and eigvals are already sorted/arranged by ModeSelector?
            # ModeSelector sorts by eigenvalue. 
            # If mode='prfo', eigvals[0] is negative (max mode), others are positive (min modes).
            # This matches g_tilde_max (index 0) and g_tilde_min (index 1:).
            
            lambdas_cpu = eigvals.detach().cpu().numpy()
            g_tilde_cpu = g_tilde.detach().cpu().numpy()
            
            # Secular Equation Solver on CPU
            alpha_sq = self._solve_trust_region_cpu(lambdas_cpu, g_tilde_cpu, trust_radius_saddle, tol)
            
            # Compute final step on GPU using the optimal alpha_sq
            s1, _ = self._rfo_gpu(ev_max, g_tilde_max, 'max', alpha_sq) 
            s2, _ = self._rfo_gpu(ev_min, g_tilde_min, 'min', alpha_sq) 
            s = eigvecs @ torch.cat([s1, s2]) 
            
        except Exception as e: 
            if logfile: logfile.write(f"  Warning: Trust constraint failed ({e}), using scaling.\n") 
            if norm_s_unc > tol: 
                s = s_unc * (trust_radius_saddle / norm_s_unc) 
            else: 
                s = torch.zeros_like(g) 

        s_norm_final = torch.norm(s) 
        if s_norm_final > (trust_radius_saddle * 1.05): 
            s *= (trust_radius_saddle / s_norm_final) 
        return s 

    def _solve_trust_region_cpu(self, lambdas, g_tilde, delta, tol):
        """
        Solves the trust region subproblem using Secular Equation on CPU.
        Returns alpha_sq (squared shift parameter).
        """
        # We need to find alpha_sq such that ||s(alpha_sq)|| = delta
        # The step s is composed of s_max (index 0) and s_min (indices 1:)
        # s_i = g_tilde_i / (lambda_i - mu_i) * sqrt(alpha_sq)
        # where mu_i depends on alpha_sq implicitly via the RFO subproblem structure.
        # Wait, the RFO subproblem structure is:
        #   [ diag(lam)   g/a ] [s] = mu [s]
        #   [ g.T/a       0   ] [1]      [1]
        # This is equivalent to (lam - mu) * s = -g/a * scale -> s = -g/a / (lam - mu) * scale
        # And the secular equation for the augmented matrix eigenvalues is:
        #   -mu + sum( (g_i/a)^2 / (lam_i - mu) ) = 0
        
        # Let v = g_tilde / a.
        # For index 0 (max mode): we want the largest eigenvalue of the augmented matrix.
        # For indices 1: (min mode): we want the smallest eigenvalue.
        
        # However, we are solving for 'a' (alpha) such that total step norm is delta.
        # It's easier to keep using brentq on 'a_sq' but implement the inner RFO solver
        # using secular equation on CPU.
        
        def compute_step_norm_sq(a_sq):
            a = np.sqrt(max(a_sq, 1e-15))
            v = g_tilde / a
            
            # 1. Max Mode (Index 0)
            # Find largest root mu > lam[0]
            lam0 = lambdas[0]
            v0 = v[0]
            
            # Secular eq for 1D case is simple: -mu + v0^2/(lam0 - mu) = 0
            # => -mu(lam0 - mu) + v0^2 = 0 => mu^2 - lam0*mu - v0^2 = 0
            # mu = (lam0 + sqrt(lam0^2 + 4v0^2))/2  (Positive root for largest eigval)
            mu_max = (lam0 + np.sqrt(lam0**2 + 4*v0**2)) / 2.0
            
            # s0 = v0 / (mu_max - lam0) * a = v0 / ( (lam0 + sqrt...)/2 - lam0 ) * a
            #    = v0 / ( (sqrt... - lam0)/2 ) * a
            s0 = (v0 / (mu_max - lam0)) * a
            
            # 2. Min Modes (Indices 1:)
            lam_min = lambdas[1:]
            v_min = v[1:]
            
            # Secular eq: f(mu) = -mu + sum( vi^2 / (lami - mu) ) = 0
            # We want smallest root mu < min(lam_min)
            
            # Optimization: If len(lam_min) is small, just diagonalize? 
            # No, standard is consistent. But for performance, secular is O(N).
            
            # Define secular function for min block
            def secular_min(mu):
                return -mu + np.sum(v_min**2 / (lam_min - mu))
            
            l_min_val = np.min(lam_min)
            
            # Bracket for smallest root
            upper = l_min_val - 1e-9
            # Lower bound estimate: Gershgorin or just sufficiently small
            lower = l_min_val - np.linalg.norm(v_min) - 1.0
            
            # Binary search / Brentq to find mu
            # Check bounds first
            if secular_min(upper) == 0: return np.inf # Singularity
            
            # Extend lower if needed
            cnt = 0
            while secular_min(lower) < 0 and cnt < 20:
                 lower = lower * 2.0 if lower < 0 else lower - 10.0
                 cnt += 1
            
            try:
                mu_min_val = brentq(secular_min, lower, upper, xtol=1e-12)
                s_min = (v_min / (mu_min_val - lam_min)) * a
            except ValueError:
                 # Fallback if bracketing fails (rare)
                 s_min = np.zeros_like(v_min)

            return s0**2 + np.sum(s_min**2)

        def residual(a_sq):
             return compute_step_norm_sq(a_sq) - delta**2

        try:
            alpha_sq = brentq(residual, 1e-6, 1e6, xtol=1e-4)
        except ValueError:
            # Fallback
            alpha_sq = 1.0
            
        return alpha_sq

    def _rfo_gpu(self, lam, g, mode, a_sq): 
        # GPU RFO Logic 
        # Kept for final step generation (and fallback)
        dim = len(lam) 
        aug = torch.zeros((dim+1, dim+1), device=self.device, dtype=torch.float64) 
        aug[:dim,:dim] = torch.diag(lam) 
        a = np.sqrt(max(a_sq, 1e-15)) 
        aug[:dim, dim] = g/a 
        aug[dim, :dim] = g/a 
        
        vals, vecs = torch.linalg.eigh(aug) 
        idx = -1 if mode == 'max' else 0 
        scale = vecs[-1, idx] 
        
        if abs(scale) < 1e-15: 
            # Newton fallback on GPU 
            s_newton = - (1.0 / lam) * g 
            s_newton = torch.nan_to_num(s_newton, 0.0) 
            return s_newton, 0.0 
        return (vecs[:dim, idx]/scale)*a, vals[idx]/2 
