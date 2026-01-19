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
        # We define residual to keep matrix ops on GPU
        def residual(a_sq): 
            s1, _ = self._rfo_gpu(ev_max, g_tilde_max, 'max', a_sq) 
            s2, _ = self._rfo_gpu(ev_min, g_tilde_min, 'min', a_sq) 
            # Only transfer scalar result to CPU 
            return (torch.sum(s1**2) + torch.sum(s2**2)).item() - trust_radius_saddle**2 

        try: 
            # Relaxed tolerance (1e-4) to reduce CPU-GPU interaction loops 
            alpha_sq = brentq(residual, 1e-20, 1e6, xtol=1e-4) 
            
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

    def _rfo_gpu(self, lam, g, mode, a_sq): 
        # GPU RFO Logic 
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
