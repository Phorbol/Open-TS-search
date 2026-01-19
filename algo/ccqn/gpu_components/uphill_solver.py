import torch
from algo.ccqn.gpu_components.gpu_kernels import jit_solve_pgd

class GPUUphillSolver:
    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def solve(self, g: torch.Tensor, B: torch.Tensor, e_vec: torch.Tensor, 
              trust_radius_uphill: float, cos_phi: float) -> torch.Tensor:
        """
        Solve PGD on GPU using JIT kernel.
        """
        # Initial guess: along e_vec
        s0 = e_vec * trust_radius_uphill
        
        # Call JIT kernel
        s = jit_solve_pgd(s0, g, B, e_vec, 
                          float(trust_radius_uphill), 
                          float(cos_phi), 
                          self.max_iter)
        return s
