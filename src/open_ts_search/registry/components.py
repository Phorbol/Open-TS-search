from open_ts_search.core.ts_components.hessian_ts_bfgs import HessianTSBFGS
from open_ts_search.algo.ccqn.components.ccqn_mode import CCQNModeSelector
from open_ts_search.algo.ccqn.components.direction_ccqn import _DirectionProvider as CCQNDirectionProvider
from open_ts_search.core.ts_components.prfo_solver import PRFOSolver
from open_ts_search.core.ts_components.trust_linear import TrustRegionLinear
from open_ts_search.core.ts_components.convergence_prfo import ConvergencePRFOFmax
from open_ts_search.algo.ccqn.components.ccqn_uphill import CCQNUphillSolver
from open_ts_search.algo.ccqn.components.ccqn_convergence import CCQNConvergenceChecker

class ComponentRegistry:
    def __init__(self):
        self._constructors = {}
    def register(self, role, version, constructor):
        self._constructors[(role, version)] = constructor
    def get(self, role, version):
        return self._constructors.get((role, version))

default_component_registry = ComponentRegistry()

def register_defaults():
    default_component_registry.register('hessian_manager', 'ts.v1', HessianTSBFGS)
    default_component_registry.register('mode_selector', 'ccqn.v1', CCQNModeSelector)
    default_component_registry.register('direction_provider', 'ccqn.v1', CCQNDirectionProvider)
    default_component_registry.register('prfo_solver', 'ts.v1', PRFOSolver)
    default_component_registry.register('trust_manager', 'ts.v1', TrustRegionLinear)
    default_component_registry.register('convergence_checker', 'ccqn.v1', CCQNConvergenceChecker)
    default_component_registry.register('uphill_solver', 'ccqn.v1', CCQNUphillSolver)
