import numpy as np

class EVectorGenerator:
    """
    Robust Markovian E-Vector Generator for CCQN.
    
    This class implements a regularized least squares projection method to construct
    the uphill direction vector (e_vector). It supports explicit chemical intent
    (bond breaking/forming) via a "Sign Lock" mechanism, ensuring the optimization
    strictly follows the user's desired reaction coordinate direction.
    
    Algorithm:
        1. Construct Jacobian J mapping Cartesian displacements to bond length changes.
        2. Solve (JJ^T + lambda*I) * dq = Jg to find optimal bond changes matching the gradient.
        3. Project back to Cartesian space: e = J^T * dq.
        4. Apply Chemical Sign Lock: If e contradicts the intent S, flip e.
    """
    
    def __init__(self, reactive_bonds, regularization=1e-3):
        """
        Initialize the generator.
        
        Args:
            reactive_bonds: List of bonds. Can be:
                - [(i, j), ...] -> Default intent +1 (Breaking)
                - [(i, j, intent), ...] where intent is '+' (Break), '-' (Form), or 'auto'
            regularization (float): Tikhonov regularization parameter (lambda).
        """
        self.bond_indices, self.target_signs = self._parse_bonds(reactive_bonds)
        self.lam = regularization

    def _parse_bonds(self, bonds):
        """Parse reactive bonds into indices and sign vectors."""
        indices = []
        signs = []
        
        for item in bonds:
            if len(item) == 2:
                # Default case: (i, j) -> Assume Breaking (+1)
                # Note: User can explicitly use 'auto' to disable sign lock if needed,
                # but for simple tuples, +1 is a safe default for TS search.
                i, j = item
                intent = '+' 
            elif len(item) == 3:
                i, j, intent = item
            else:
                raise ValueError(f"Invalid bond format: {item}")
            
            indices.append((i, j))
            
            if intent == '+' or intent == 'break' or intent == 'stretch':
                signs.append(1.0)
            elif intent == '-' or intent == 'form' or intent == 'compress':
                signs.append(-1.0)
            elif intent == 'auto' or intent == 0:
                signs.append(0.0) # 0 means no sign lock enforcement
            else:
                # Default to +1 if unknown, or raise error? 
                # Let's default to +1 to be safe for TS
                signs.append(1.0)
                
        return indices, np.array(signs)

    def compute(self, atoms, forces):
        """
        Compute the cone axis vector 'e' using robust projection.
        
        Args:
            atoms: ASE Atoms object (for positions)
            forces: Force array (N, 3)
            
        Returns:
            e (np.ndarray): Normalized e_vector (N*3,)
        """
        R = atoms.get_positions()
        F = forces
        
        # Ensure F is (N, 3)
        if F.ndim == 1:
            F = F.reshape(-1, 3)
            
        N = R.shape[0]
        K = len(self.bond_indices)
        g = -F.flatten() # Uphill direction (Gradient)
        
        if K == 0:
             return np.zeros(3*N)

        # 1. Construct Jacobian Matrix J (K x 3N)
        J = np.zeros((K, 3 * N))
        
        valid_bonds_count = 0
        
        for k, (i, j) in enumerate(self.bond_indices):
            diff = R[j] - R[i]
            dist = np.linalg.norm(diff)
            
            if dist < 1e-4:
                continue
                
            v_hat = diff / dist
            
            # Fill Jacobian: dL/dRi = -v, dL/dRj = +v
            idx_i = slice(3 * i, 3 * i + 3)
            idx_j = slice(3 * j, 3 * j + 3)
            
            J[k, idx_i] = -v_hat
            J[k, idx_j] = +v_hat
            valid_bonds_count += 1
            
        if valid_bonds_count == 0:
            # Fallback
            norm_g = np.linalg.norm(g)
            return g / norm_g if norm_g > 1e-8 else g

        # 2. Least Squares Projection
        # Solve (JJ^T + lambda*I) * dq = Jg
        Gram = J @ J.T # (K, K)
        
        # Adaptive regularization
        scale = np.trace(Gram) / K if K > 0 else 1.0
        reg_matrix = Gram + (self.lam * scale + 1e-12) * np.eye(K)
        
        rhs = J @ g
        
        try:
            dq_star = np.linalg.solve(reg_matrix, rhs)
        except np.linalg.LinAlgError:
            # Fallback to pure geometric intent if singular
            dq_star = self.target_signs
            
        # Map back to Cartesian
        e_unnormalized = J.T @ dq_star
        
        # 3. Geometric Fallback
        # If force projection is tiny (orthogonal or zero force), use intent
        if np.linalg.norm(e_unnormalized) < 1e-10:
            e_unnormalized = J.T @ self.target_signs
            
        # 4. Chemical Sign Lock
        # Project e back onto bonds to see if it aligns with intent
        # delta_L = J * e
        delta_bond_lengths = J @ e_unnormalized
        
        # Correlation: S . dL
        # If S=+1 (break) and dL > 0 (lengthening), prod > 0. Good.
        # If S=+1 and dL < 0 (shortening), prod < 0. Bad.
        # Only check against non-zero intents
        active_mask = np.abs(self.target_signs) > 0.1
        if np.any(active_mask):
            correlation = np.dot(self.target_signs[active_mask], delta_bond_lengths[active_mask])
            
            if correlation < 0:
                # Flip to enforce intent
                e_unnormalized = -e_unnormalized
                
        # 5. Normalize
        norm_e = np.linalg.norm(e_unnormalized)
        if norm_e > 1e-12:
            e = e_unnormalized / norm_e
        else:
            e = np.zeros_like(e_unnormalized)
            
        return e
