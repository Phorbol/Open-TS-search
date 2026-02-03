import unittest
import numpy as np
from ase import Atoms
from algo.ccqn.gpu_components.e_vector_generator import EVectorGenerator

class TestEVectorGenerator(unittest.TestCase):
    
    def test_parse_bonds(self):
        """Test parsing of different bond formats."""
        # Case 1: Legacy format
        bonds = [(0, 1), (1, 2)]
        gen = EVectorGenerator(bonds)
        self.assertEqual(len(gen.bond_indices), 2)
        # Default should be +1 (Break)
        np.testing.assert_array_equal(gen.target_signs, np.array([1.0, 1.0]))
        
        # Case 2: New format mixed
        bonds = [(0, 1, '+'), (1, 2, '-'), (2, 3, 'auto')]
        gen = EVectorGenerator(bonds)
        np.testing.assert_array_equal(gen.target_signs, np.array([1.0, -1.0, 0.0]))
        
        # Case 3: Aliases
        bonds = [(0, 1, 'break'), (1, 2, 'form')]
        gen = EVectorGenerator(bonds)
        np.testing.assert_array_equal(gen.target_signs, np.array([1.0, -1.0]))

    def test_compute_simple_stretch(self):
        """Test simple H2 stretch with force opposing intent (Sign Lock check)."""
        # H-H aligned on x-axis
        atoms = Atoms('H2', positions=[[0, 0, 0], [1.0, 0, 0]])
        
        # Force: Attractive (restoring force), pulling atoms together
        # f0 = +1 (right), f1 = -1 (left)
        forces = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        
        # Intent: Break (+)
        gen = EVectorGenerator([(0, 1, '+')])
        
        # Gradient = -Forces -> [-1, 0, 0], [+1, 0, 0]
        # This gradient points to COMPRESSION (atoms moving closer)
        # Intent is BREAK (atoms moving apart)
        # Sign Lock should FLIP the vector
        
        e_vec = gen.compute(atoms, forces).reshape(-1, 3)
        
        # e_vec on atom 1 (at x=1) should point RIGHT (+x) to break bond
        # e_vec[1] should be [>0, 0, 0]
        
        self.assertTrue(e_vec[1, 0] > 0.9, f"Expected positive x component, got {e_vec[1, 0]}")
        self.assertTrue(e_vec[0, 0] < -0.9, f"Expected negative x component, got {e_vec[0, 0]}")

    def test_compute_simple_compress(self):
        """Test simple H2 compress with force opposing intent."""
        # H-H aligned on x-axis
        atoms = Atoms('H2', positions=[[0, 0, 0], [1.0, 0, 0]])
        
        # Force: Repulsive, pushing atoms apart
        # f0 = -1 (left), f1 = +1 (right)
        forces = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
        
        # Intent: Form/Compress (-)
        gen = EVectorGenerator([(0, 1, '-')])
        
        # Gradient = -Forces -> [+1, 0, 0], [-1, 0, 0]
        # This gradient points to STRETCH (atoms moving apart)
        # Intent is COMPRESS
        # Sign Lock should FLIP
        
        e_vec = gen.compute(atoms, forces).reshape(-1, 3)
        
        # e_vec on atom 1 (at x=1) should point LEFT (-x) to compress bond
        self.assertTrue(e_vec[1, 0] < -0.9, f"Expected negative x component, got {e_vec[1, 0]}")

    def test_orthogonal_force(self):
        """Test fallback when force is orthogonal to bond."""
        # H-H on x-axis
        atoms = Atoms('H2', positions=[[0, 0, 0], [1.0, 0, 0]])
        
        # Force: Perpendicular (y-axis)
        forces = np.array([[0, 1.0, 0], [0, -1.0, 0]])
        
        # Intent: Break (+)
        gen = EVectorGenerator([(0, 1, '+')])
        
        # Projection of Gradient on Bond (x-axis) is 0.
        # Should fallback to geometric intent
        
        e_vec = gen.compute(atoms, forces).reshape(-1, 3)
        
        # Should still point along bond (x-axis) to break it
        self.assertTrue(abs(e_vec[1, 0]) > 0.9, "Should align with bond axis despite orthogonal force")
        self.assertTrue(abs(e_vec[1, 1]) < 0.1, "Should have no y component")

if __name__ == '__main__':
    unittest.main()
