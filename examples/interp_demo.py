from ase.build import molecule
from ase.io import write
from shared.interp import robust_interpolate, Vectorized_ASE_IDPPSolver

def make_endpoints():
    start = molecule('H2')
    end = start.copy()
    pos = end.get_positions()
    pos[1] = pos[1] + [0.5, 0.0, 0.0]
    end.set_positions(pos)
    return start, end

def run_linear(nimages=5):
    start, end = make_endpoints()
    path = robust_interpolate(start, end, nimages)
    write('interp_linear.xyz', path)
    print(f'Linear path images: {len(path)}')

def run_idpp(nimages=5):
    start, end = make_endpoints()
    solver = Vectorized_ASE_IDPPSolver.from_endpoints(start, end, nimages)
    path = solver.run()
    write('interp_idpp.xyz', path)
    print(f'IDPP path images: {len(path)}')

if __name__ == '__main__':
    run_linear(5)
    run_idpp(5)
