def test_create_ccqn_v110():
    from registry.factory import create_optimizer
    from ase.build import molecule
    mol = molecule('H2')
    opt = create_optimizer('ccqn', 'v1.10', mol, e_vector_method='interp', product_atoms=mol.copy(), idpp_images=3, use_idpp=False, hessian=False)
    assert opt is not None
