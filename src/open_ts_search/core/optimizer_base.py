class OptimizerBase:
    def step(self, f=None):
        raise NotImplementedError
    def converged(self, forces=None):
        raise NotImplementedError
