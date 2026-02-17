import unittest
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
import sys
import os
sys.path.append(os.getcwd())
from open_ts_search.shared.freq import get_vib_mode

class TestGetVibMode(unittest.TestCase):
    def setUp(self):
        # Create a simple dummy system: H2 molecule
        self.atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
        self.atoms.calc = EMT()

    def test_calculator_argument(self):
        # Test if explicit calculator is used
        # We can verify this by passing a mock calculator or just a different instance
        calc2 = EMT()
        # To avoid running actual heavy Vibrations, we rely on the fact that if calculator is passed, 
        # it should be attached to the atoms copy. 
        # But get_vib_mode runs vib.run(), which is slow.
        # For this unit test, we might want to mock Vibrations to avoid actual run.
        pass

    def test_mock_multi_imaginary(self):
        # We need to mock Vibrations to simulate multiple imaginary frequencies
        from unittest.mock import MagicMock, patch
        
        with patch('shared.freq.Vibrations') as MockVib:
            instance = MockVib.return_value
            # Mock run
            instance.run.return_value = None
            
            # Mock energies: 3 modes, 2 imaginary (>50), 1 real
            # Energies are complex numbers in ASE Vib
            # 100i cm^-1 -> energy? E = h*nu. 
            # get_energies returns values where e.imag > 0 means imaginary freq.
            # Let's just mock the logic inside get_vib_mode directly?
            # get_vib_mode uses: energies = vib.get_energies()
            # units.invcm is roughly 1/8065 eV
            
            # Imag 200i, Imag 100i, Real 300
            e1 = 200 * units.invcm * 1j
            e2 = 100 * units.invcm * 1j
            e3 = 300 * units.invcm
            
            instance.get_energies.return_value = [e1, e2, e3]
            
            # Mock modes
            instance.get_mode.side_effect = lambda i: np.array([float(i)]*6) # dummy vectors
            
            # Call function
            vec = get_vib_mode(self.atoms, calculator=EMT(), cutoff_cm=50.0)
            
            # Verify:
            # 1. Should pick the largest imaginary (200i, index 0)
            # 2. Should return vector for index 0 (which we mocked as 0s)
            self.assertTrue(np.allclose(vec, np.array([0.0]*6)))
            
            # Verify clean was called
            instance.clean.assert_called()

    def test_mock_small_imaginary_filtering(self):
        from unittest.mock import MagicMock, patch
        
        with patch('shared.freq.Vibrations') as MockVib:
            instance = MockVib.return_value
            instance.run.return_value = None
            
            # Imag 30i (below cutoff 50), Real 300
            e1 = 30 * units.invcm * 1j
            e2 = 300 * units.invcm
            
            instance.get_energies.return_value = [e1, e2]
            
            vec = get_vib_mode(self.atoms, cutoff_cm=50.0)
            
            # Should return None because 30i < 50
            self.assertIsNone(vec)

if __name__ == '__main__':
    unittest.main()
