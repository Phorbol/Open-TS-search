
import torch
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from algo.ccqn.gpu_components.ccqn_gpu_driver import CCQNGPUDriver

def main():
    print("Initializing CCQN-GPU Driver Test...")

    # 1. Setup System
    d = 0.95
    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, d)])
    atoms.calc = EMT()

    # 2. Initialize Driver
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    driver = CCQNGPUDriver(
        device=device,
        trust_radius_uphill=0.1,
        hessian=False 
    )
    
    # Initialize state
    driver.initialize(natoms=len(atoms))

    # Open a log file
    with open('ccqn_gpu_debug.log', 'w') as logfile:
        logfile.write("Starting optimization loop...\n")

        # 3. Run Custom Loop
        for step in range(5):
            # Get current properties
            f_k = atoms.get_forces()
            x_k = atoms.get_positions()
            e_k = atoms.get_potential_energy()
            
            # Create a mock e_vector (along Z axis, stretching the bond)
            # H2 is along Z. Atom 1 at 0, Atom 2 at d.
            # Stretching: Atom 1 moves -Z, Atom 2 moves +Z
            e_vec = np.zeros_like(x_k)
            e_vec[0, 2] = -1.0
            e_vec[1, 2] = 1.0
            e_vec = e_vec.flatten()
            e_vec /= np.linalg.norm(e_vec)

            # Compute Step
            print(f"Step {step}...")
            s_k_tensor = driver.compute_step(f_k, x_k, e_k, e_vec_np=e_vec, logfile=logfile)
            
            # Apply Step
            s_k = s_k_tensor.cpu().numpy()
            x_new = x_k.flatten() + s_k
            atoms.set_positions(x_new.reshape(-1, 3))
            
    print("Run finished. Checking log file content:")
    with open('ccqn_gpu_debug.log', 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
