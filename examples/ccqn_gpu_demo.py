import os
import sys
from ase.io import read

# Ensure we can import from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from mace.calculators import MACECalculator
except ImportError:
    print("MACE not installed. Skipping demo.")
    sys.exit(0)

from open_ts_search.algo.ccqn.ccqn_optimizer_gpu import CCQNGPUOptimizer

def main():
    # Load the initial structure
    cif_path = os.path.join(os.path.dirname(__file__), 'IS.cif')
    if not os.path.exists(cif_path):
        print(f"File not found: {cif_path}")
        return

    image_TS_initial1 = read(cif_path, index=-1)

    # Set up the MACE calculator
    # Note: You may need to adjust the model_paths to point to your specific model file
    model_path = os.path.join(os.path.dirname(__file__), 'mace-omat-0-small.model')
    
    # Check if model exists (optional, but good for a robust demo)
    # if not os.path.exists(model_path):
    #     print(f"Warning: Model file not found at {model_path}. Please ensure the path is correct.")

    try:
        calc = MACECalculator(
            model_paths=model_path,
            device='cuda',
            default_dtype='float32',
        )
        image_TS_initial1.calc = calc
    except Exception as e:
        print(f"Failed to initialize MACECalculator: {e}")
        return

    # Initialize CCQN GPU Optimizer
    print("Initializing CCQN GPU Optimizer...")
    ccqn = CCQNGPUOptimizer(
        image_TS_initial1, 
        uphill_use_slsqp=False, 
        uphill_use_alm=False, 
        uphill_use_adam=False, 
        # PGD parameters
        uphill_max_iter=500,     # PGD iterations
        reactive_bonds=[(23, 24), (24, 317)], 
        e_vector_method='ic', 
        cos_phi=0.3, 
        trajectory='ccqn.traj'
    )

    print("Starting run...")
    ccqn.run(fmax=0.05, steps=500)
    print("Run complete.")

if __name__ == "__main__":
    main()
