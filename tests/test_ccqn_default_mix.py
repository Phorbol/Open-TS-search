def test_create_ccqn_default_mix():
    from open_ts_search.registry.factory import create_ccqn
    from ase.build import molecule
    mol = molecule('H2')
    opt = create_ccqn(
        mol,
        e_vector_method='interp',
        product_atoms=mol.copy(),
        idpp_images=3,
        use_idpp=False,
        hessian=False
    )
    assert opt is not None
