import unittest
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
import sys
import os
sys.path.append(os.getcwd())
from shared import TSDescentOptimizer, get_vib_mode, plot_descent_profile

class TestFakeIRC(unittest.TestCase):
    def setUp(self):
        # Create a simple dummy system: H2 molecule
        self.atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
        self.atoms.calc = EMT()
        
        # Mock a vibration mode (just a vector)
        self.vib_mode = np.array([[0, 0, -0.1], [0, 0, 0.1]])

    def test_optimizer_init(self):
        opt = TSDescentOptimizer(self.atoms, self.vib_mode, delta=0.05, log_prefix="test_descent")
        self.assertIsNotNone(opt)
        self.assertEqual(opt.delta, 0.05)

    def test_optimizer_run_mock(self):
        # We won't run full optimization to avoid long runtimes, 
        # but we can try running with steps=0 or 1 to check flow.
        opt = TSDescentOptimizer(self.atoms, self.vib_mode, delta=0.01, log_prefix="test_descent_run")
        try:
            # Run with very loose fmax and few steps
            path = opt.run(fmax=10.0, steps=1) 
            # Note: steps=1 might not suffice for FIRE to start, but should not crash.
            # path might be short.
        except Exception as e:
            self.fail(f"Optimizer run failed with error: {e}")
        
        # Check if files are created
        import os
        self.assertTrue(os.path.exists("test_descent_run_fwd.traj"))
        self.assertTrue(os.path.exists("test_descent_run_rev.traj"))
        
        # Cleanup
        for f in ["test_descent_run_fwd.traj", "test_descent_run_rev.traj", 
                  "test_descent_run_fwd.log", "test_descent_run_rev.log",
                  "test_descent_run_full.traj", "test_descent_run_full.xyz"]:
            if os.path.exists(f):
                os.remove(f)

    def test_plot_mock(self):
        # Create a fake path
        path = [self.atoms.copy() for _ in range(5)]
        for i, a in enumerate(path):
            a.calc = EMT() # Need calculator for potential energy
            pos = a.get_positions()
            pos[1, 2] += i * 0.1
            a.set_positions(pos)
            
        # Test plotting code (non-interactive)
        try:
            plot_descent_profile(path, title="Test Profile", save_path="test_profile.png")
        except Exception as e:
            self.fail(f"Plotting failed: {e}")
            
        import os
        self.assertTrue(os.path.exists("test_profile.png"))
        if os.path.exists("test_profile.png"):
            os.remove("test_profile.png")

if __name__ == '__main__':
    unittest.main()
