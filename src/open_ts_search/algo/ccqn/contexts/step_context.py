class StepContext:
    def __init__(self, atoms, B, g_k, x_k, e_k, eigvals, eigvecs, trust_radius_uphill, trust_radius_saddle, cos_phi, e_vector_method, product_atoms, idpp_images, use_idpp, reactive_bonds, ic_mode, prev_pos, prev_grad, prev_energy, prev_B, prev_mode, logfile):
        self.atoms = atoms
        self.B = B
        self.g_k = g_k
        self.x_k = x_k
        self.e_k = e_k
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.trust_radius_uphill = trust_radius_uphill
        self.trust_radius_saddle = trust_radius_saddle
        self.cos_phi = cos_phi
        self.e_vector_method = e_vector_method
        self.product_atoms = product_atoms
        self.idpp_images = idpp_images
        self.use_idpp = use_idpp
        self.reactive_bonds = reactive_bonds
        self.ic_mode = ic_mode
        self.prev_pos = prev_pos
        self.prev_grad = prev_grad
        self.prev_energy = prev_energy
        self.prev_B = prev_B
        self.prev_mode = prev_mode
        self.logfile = logfile
