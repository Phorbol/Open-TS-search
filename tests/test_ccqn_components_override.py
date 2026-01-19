def test_create_ccqn_with_components():
    from registry.factory import create_ccqn
    from algo.ccqn.components import _DirectionProvider
    from ase.build import molecule
    mol = molecule('H2')
    opt = create_ccqn(
        mol,
        components={
            'direction_provider': _DirectionProvider
        },
        e_vector_method='interp',
        product_atoms=mol.copy(),
        idpp_images=3,
        use_idpp=False,
        hessian=False
    )
    assert opt is not None
