class _Config:
    def __init__(self, e_vector_method, ic_mode, cos_phi, trust_radius_uphill, trust_radius_saddle_initial, idpp_images, use_idpp, hessian):
        self.e_vector_method = e_vector_method
        self.ic_mode = ic_mode
        self.cos_phi = cos_phi
        self.trust_radius_uphill = trust_radius_uphill
        self.trust_radius_saddle_initial = trust_radius_saddle_initial
        self.idpp_images = idpp_images
        self.use_idpp = use_idpp
        self.hessian = hessian
