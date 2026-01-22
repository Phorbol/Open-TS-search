from registry.registry import default_registry
from registry.components import default_component_registry, register_defaults as register_component_defaults
from algo.ccqn.ccqn_optimizer import CCQNOptimizer
from algo.ccqn.ccqn_optimizer_gpu import CCQNGPUOptimizer

def register_defaults():
    default_registry.register_algorithm('ccqn', 'v1.10', CCQNOptimizer)
    default_registry.register_algorithm('ccqn-gpu', 'v1.0', CCQNGPUOptimizer)
    register_component_defaults()

def create_optimizer(algorithm, version, atoms, **kwargs):
    register_defaults()
    ctor = default_registry.get_algorithm(algorithm, version)
    if ctor is None:
        raise ValueError('algorithm version not found')
    return ctor(atoms, **kwargs)

def create_ccqn(atoms, components=None, **kwargs):
    register_defaults()
    components = components or {}
    def get_instance(role, default_version):
        val = components.get(role)
        if val is None:
            ctor = default_component_registry.get(role, default_version)
            if ctor is None:
                return None
            if role == 'hessian_manager':
                return ctor(atoms, kwargs.get('hessian', False))
            if role == 'trust_manager':
                return ctor(kwargs.get('trust_radius_saddle_min', 5e-3), kwargs.get('trust_radius_saddle_max', 0.2))
            return ctor()
        if isinstance(val, type):
            if role == 'hessian_manager':
                return val(atoms, kwargs.get('hessian', False))
            if role == 'trust_manager':
                return val(kwargs.get('trust_radius_saddle_min', 5e-3), kwargs.get('trust_radius_saddle_max', 0.2))
            return val()
        return val
    hessian_manager = get_instance('hessian_manager', 'ts.v1')
    mode_selector = get_instance('mode_selector', 'ts.v1')
    direction_provider = get_instance('direction_provider', 'ts.v1')
    uphill_solver = get_instance('uphill_solver', 'ccqn.v1')
    prfo_solver = get_instance('prfo_solver', 'ts.v1')
    trust_manager = get_instance('trust_manager', 'ts.v1')
    convergence_checker = get_instance('convergence_checker', 'ccqn.v1')
    return CCQNOptimizer(
        atoms,
        hessian_manager=hessian_manager,
        mode_selector=mode_selector,
        direction_provider=direction_provider,
        uphill_solver=uphill_solver,
        prfo_solver=prfo_solver,
        trust_manager=trust_manager,
        convergence_checker=convergence_checker,
        **kwargs
    )
