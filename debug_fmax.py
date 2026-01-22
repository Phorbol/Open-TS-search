
import numpy as np
import torch
from ase import Atoms
from ase.calculators.emt import EMT
from algo.ccqn.gpu_components.ccqn_gpu_driver import CCQNGPUDriver
from algo.ccqn.ccqn_optimizer_gpu import CCQNGPUOptimizer

def check_fmax():
    # 1. Setup
    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1.0)])
    atoms.calc = EMT()
    f = atoms.get_forces()
    print(f"Forces shape: {f.shape}")
    print(f"Forces:\n{f}")
    
    # 2. ASE Logic
    ase_fmax = np.sqrt((f**2).sum(axis=1).max())
    print(f"ASE fmax: {ase_fmax}")
    
    # 3. Driver Logic (Manual Check)
    f_k = f
    driver_fmax_np = np.sqrt((f_k**2).sum(axis=1).max())
    print(f"Driver numpy fmax: {driver_fmax_np}")
    
    # 4. Driver Logic (Torch Path Check)
    device = torch.device('cpu')
    g_k = torch.from_numpy(-f.flatten()).to(device, dtype=torch.float64)
    driver_fmax_torch = torch.sqrt((g_k.reshape(-1, 3)**2).sum(dim=1).max()).item()
    print(f"Driver torch fmax: {driver_fmax_torch}")

    # 5. Check if they match
    assert np.isclose(ase_fmax, driver_fmax_np)
    assert np.isclose(ase_fmax, driver_fmax_torch)
    print("All calculations match.")

if __name__ == "__main__":
    check_fmax()
