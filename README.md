# Open-TS-search

## Overview
- Transition state search tooling with a modular architecture:
  - Shared, algorithm-agnostic utilities (interpolation/IDPP, IRC, frequency)
  - Core TS components (Hessian, PRFO, trust region, convergence)
  - CCQN-specific optimizer, components, phases, and contexts
- Dependencies are imported lazily inside functions where possible to minimize global load.

## Directory Structure
- shared/
  - interp.py: robust_interpolate, Vectorized_ASE_IDPPSolver
  - irc.py: get_clean_irc_path, mass_weighted_path, combine_irc, plot_irc
  - freq.py: ase_vib (imaginary frequency analysis and animation export)
- core/ts_components/
  - hessian_ts_bfgs.py
  - prfo_solver.py
  - trust_linear.py
  - convergence_prfo.py
- algo/ccqn/
  - ccqn_optimizer.py
  - components/
    - config.py, state_tracker.py, hessian_manager.py
    - direction_ccqn.py, prfo_solver_ccqn.py, trust_manager_ccqn.py
    - ccqn_mode.py, ccqn_uphill.py, ccqn_convergence.py
  - contexts/step_context.py
  - phases/uphill_phase.py, phases/prfo_phase.py
- registry/
  - components.py, factory.py
- examples/
  - README.md
  - interp_demo.py, freq_demo.py, irc_demo.py

## Shared Utilities
- Interpolation & IDPP (shared/interp.py)
  - robust_interpolate: PBC-aware linear interpolation
  - Vectorized_ASE_IDPPSolver: fully vectorized IDPP path refinement
- IRC (shared/irc.py)
  - get_clean_irc_path: forward/reverse runs, path stitching
  - mass_weighted_path, plot_irc: analysis and visualization
  - Requires sella; demos skip gracefully if missing
- Frequency (shared/freq.py)
  - ase_vib: imaginary frequency count and animated mode export
  - Uses ASE Vibrations; imports are inside the function

## CCQN Architecture
- CCQNOptimizer orchestrates the flow (Hessian update, mode switching, logging, state)
- Phases:
  - UphillPhase: e-vector selection (interp/IDPP or IC) + constrained step
  - PRFOPhase: rho quality, trust radius update, PRFO step
- Components are single-responsibility files for easy maintenance and replacement.

## Quick Start
```python
from registry.factory import create_ccqn
from ase.build import molecule
from ase.calculators.emt import EMT

mol = molecule('H2')
mol.calc = EMT()
opt = create_ccqn(
    mol,
    e_vector_method='interp',
    product_atoms=mol.copy(),
    idpp_images=5,
    use_idpp=True,
    hessian=False
)
for _ in range(10):
    opt.step()
```

## Examples
- Interp/IDPP demo: examples/interp_demo.py
- Frequency demo: examples/freq_demo.py
- IRC demo: examples/irc_demo.py
  - If sella is not installed, the script will print a message and exit without error.

## Notes
- Keep utilities algorithm-agnostic in shared/ for reuse across TS methods.
- CCQN-specific behavior (cone constraints, logging) lives under algo/ccqn/.
