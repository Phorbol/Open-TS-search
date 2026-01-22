class _StateTracker:
    def __init__(self):
        self.mode = 'uphill'
        self.pos_k_minus_1 = None
        self.g_k_minus_1 = None
        self.energy_k_minus_1 = None
        self.rho = 0.0
