from ase.build import molecule
from ase.calculators.emt import EMT
from open_ts_search.shared.freq import ase_vib

def run_freq():
    mol = molecule('H2')
    mol.calc = EMT()
    num_imag = ase_vib(mol, name='freq_demo', constraint=None, CUTOFF=50, calc=mol.calc)
    print(f'Imaginary frequencies count: {num_imag}')

if __name__ == '__main__':
    run_freq()
