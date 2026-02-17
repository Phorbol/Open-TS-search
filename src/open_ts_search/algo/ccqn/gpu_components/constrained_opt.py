import torch
from typing import Callable, List, Dict, Tuple, Optional

class ConstrainedOptimizer:
    """
    A PyTorch-native library for constrained optimization using Augmented Lagrangian Method (ALM).
    Designed to replace Scipy's SLSQP for GPU-resident tensors.
    """

    @staticmethod
    def minimize_alm(
        fun: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        constraints: List[Dict],
        args: tuple = (),
        max_outer_iter: int = 10,
        max_inner_iter: int = 20,
        tol: float = 1e-5,
        rho: float = 10.0,
        gamma: float = 2.0,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Minimize a function subject to constraints using the Augmented Lagrangian Method.
        
        Args:
            fun: Objective function f(x) -> scalar tensor.
            x0: Initial guess tensor.
            constraints: List of dicts {'type': 'eq'|'ineq', 'fun': callable}.
                         Note: 'ineq' means fun(x) >= 0 in Scipy, but ALM standard is usually <= 0.
                         HERE WE ADOPT SCIPY CONVENTION: 'ineq' means fun(x) >= 0.
                         So constraint is -fun(x) <= 0.
            args: Extra arguments for fun.
            max_outer_iter: Max ALM iterations.
            max_inner_iter: Max LBFGS steps per ALM iter.
            tol: Constraint violation tolerance.
            rho: Initial penalty parameter.
            gamma: Penalty growth factor.
        """
        x = x0.clone().detach().requires_grad_(True)
        device = x.device
        dtype = x.dtype
        
        # 1. Parse Constraints
        # We convert Scipy style (fun(x) >= 0) to ALM canonical form (c(x) <= 0)
        # So if type='ineq', c(x) = -fun(x) <= 0
        eq_cons = [c for c in constraints if c['type'] == 'eq']
        ineq_cons = [c for c in constraints if c['type'] == 'ineq']
        
        # 2. Initialize Multipliers
        # lambda for equality, mu for inequality
        lambda_eq = [torch.zeros(1, device=device, dtype=dtype) for _ in eq_cons]
        mu_ineq = [torch.zeros(1, device=device, dtype=dtype) for _ in ineq_cons]
        
        current_rho = rho
        
        optimizer = torch.optim.LBFGS([x], 
                                      lr=1.0, 
                                      max_iter=max_inner_iter, 
                                      history_size=10, 
                                      line_search_fn='strong_wolfe')

        for k in range(max_outer_iter):
            
            # --- Inner Optimization Loop (LBFGS) ---
            def closure():
                optimizer.zero_grad()
                
                # 1. Objective
                loss = fun(x, *args)
                
                # 2. Equality Constraints: h(x) = 0
                # Term: lambda*h + (rho/2)*h^2
                for i, con in enumerate(eq_cons):
                    val = con['fun'](x)
                    loss = loss + lambda_eq[i] * val + 0.5 * current_rho * (val ** 2)
                
                # 3. Inequality Constraints: g(x) >= 0  <==>  c(x) = -g(x) <= 0
                # Term: (1/2rho) * ( max(0, mu + rho*c)^2 - mu^2 )
                for i, con in enumerate(ineq_cons):
                    # Scipy: fun(x) >= 0
                    g_val = con['fun'](x) 
                    c_val = -g_val # Canonical: c(x) <= 0
                    
                    # Augmented Lagrangian term for inequality
                    # P(x) = (1/2rho) * [ (max(0, mu + rho*c))^2 - mu^2 ]
                    augmented_term = torch.clamp(mu_ineq[i] + current_rho * c_val, min=0.0)
                    loss = loss + (1.0 / (2.0 * current_rho)) * (augmented_term**2 - mu_ineq[i]**2)
                
                if loss.requires_grad:
                    loss.backward()
                return loss

            # Run LBFGS step (it performs multiple internal iterations)
            optimizer.step(closure)
            
            # --- Outer Loop Updates ---
            with torch.no_grad():
                max_violation = 0.0
                
                # Update Lambda (Equality)
                # lambda <- lambda + rho * h(x)
                for i, con in enumerate(eq_cons):
                    val = con['fun'](x)
                    lambda_eq[i] += current_rho * val
                    max_violation = max(max_violation, torch.abs(val).item())
                
                # Update Mu (Inequality)
                # mu <- max(0, mu + rho * c(x))
                for i, con in enumerate(ineq_cons):
                    g_val = con['fun'](x)
                    c_val = -g_val
                    mu_ineq[i] = torch.clamp(mu_ineq[i] + current_rho * c_val, min=0.0)
                    # Violation is positive part of c(x)
                    max_violation = max(max_violation, torch.clamp(c_val, min=0.0).item())
                
                if verbose:
                    print(f"ALM Iter {k}: Max Violation = {max_violation:.2e}, Rho = {current_rho:.2e}")
                
                # Convergence Check
                if max_violation < tol:
                    if verbose: print("ALM Converged.")
                    break
                
                # Update Penalty Parameter
                # Strategy: increase rho if violation is not decreasing fast enough, or just simple schedule
                current_rho *= gamma
                # Cap rho to avoid numerical issues
                current_rho = min(current_rho, 1e6)
                
        return x.detach()
