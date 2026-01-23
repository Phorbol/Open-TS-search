import os
import sys
import argparse
from ase.io import read
from ase.constraints import FixAtoms

# Ensure we can import from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared import TSDescentOptimizer, get_vib_mode, plot_descent_profile, get_clean_irc_path, plot_irc
from algo.ccqn.ccqn_optimizer_gpu import CCQNGPUOptimizer

try:
    from mace.calculators import MACECalculator
except ImportError:
    print("MACE not installed. Skipping demo.")
    sys.exit(0)

# Try importing Sella, but don't exit yet (only needed for 'true' IRC)
try:
    import sella
    SELLA_AVAILABLE = True
except ImportError:
    SELLA_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Run CCQN Optimization, Frequency Analysis, and IRC (Fake or True).")
    parser.add_argument('--irc-type', type=str, default='fake', choices=['fake', 'true'], 
                        help="Type of IRC to run: 'fake' (TSDescent) or 'true' (Sella). Default: fake")
    args = parser.parse_args()

    irc_type = args.irc_type
    
    if irc_type == 'true' and not SELLA_AVAILABLE:
        print("Error: Sella is required for 'true' IRC but is not installed.")
        return

    # 1. Setup paths and load structure
    cif_path = os.path.join(os.path.dirname(__file__), 'IS.cif')
    if not os.path.exists(cif_path):
        print(f"File not found: {cif_path}")
        return

    # Use a local model path for the demo
    model_path = os.path.join(os.path.dirname(__file__), 'mace-omat-0-small.model')
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), f'irc_{irc_type}_output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Change working directory to output_dir so files are generated there
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    print(f"Working directory changed to: {output_dir}")

    try:
        print(f"Loading IS structure from {cif_path}...")
        image_TS_initial1 = read(cif_path, index=-1)
        
        # Setup Calculator
        print("Initializing MACECalculator...")
        calc = MACECalculator(
            model_paths=model_path,
            device='cuda',
            default_dtype='float32',
        )
        image_TS_initial1.calc = calc
        
        # ---------------------------------------------------------------------
        # Step 0: Run CCQN to find Transition State (TS) from IS
        # ---------------------------------------------------------------------
        print("\n=== Step 0: Running CCQN to find TS from IS ===")
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
            ic_mode='weighted',
            logfile='ccqn_run.log',
            trajectory='ccqn_run.traj'
        )
        ccqn.run(fmax=0.05, steps=500)
        
        ts_structure = image_TS_initial1
        print("CCQN run complete. Assuming current structure is TS candidate.")

        # ---------------------------------------------------------------------
        # Step 1: Analyze Vibrations & Get Mode (on the found TS)
        # ---------------------------------------------------------------------
        print("\n=== Step 1: Running Vibration Analysis on TS candidate ===")
        # Always run freq analysis first to confirm TS character
        mode = get_vib_mode(
            ts_structure,
            calculator=calc,
            # Fix Si and O atoms as requested
            constraint=FixAtoms([atom.index for atom in ts_structure if atom.symbol in ['Si', 'O']]),
            name='vib_check',
            cutoff_cm=50.0,       # Filter imaginary frequencies < 50 cm^-1
            output_all_imag_modes=True # Output animations for significant imaginary modes
        )

        if mode is None:
            print("\nNo significant imaginary mode found. Skipping IRC descent.")
            return

        print("\nSignificant imaginary mode found.")

        # ---------------------------------------------------------------------
        # Step 2: Run IRC (Fake or True)
        # ---------------------------------------------------------------------
        print(f"\n=== Step 2: Running {irc_type.upper()} IRC Descent... ===")
        # Re-ensure calculator is attached
        ts_structure.calc = calc

        if irc_type == 'fake':
            optimizer = TSDescentOptimizer(ts_structure, mode, delta=0.1, log_prefix="irc_check")
            path = optimizer.run(fmax=0.05, steps=500)

            # Step 3: Plot Profile
            print("\n=== Step 3: Plotting Descent Profile ===")
            plot_descent_profile(path, title="Approximate IRC Profile", save_path="profile.png")

        elif irc_type == 'true':
            # 参数对应 Sella IRC 的配置: dx (步长), eta (容差), ninner_iter (内部迭代)
            irc_path = get_clean_irc_path(
                ts_structure,
                irc_log_prefix="my_irc_run",
                fmax=0.05,
                steps=500,
                dx=0.1,
                eta=1e-4
            )
            
            if irc_path:
                print(f"IRC 成功，路径长度: {len(irc_path)}")
                print("\n=== Step 3: 绘制能量曲线 ===")
                plot_irc(irc_path, title="My Reaction Profile")
            else:
                print("IRC 失败 (Sella 可能未安装或未收敛)")

        print(f"Done. Results saved in {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
