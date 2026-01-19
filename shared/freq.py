import numpy as np

def ase_vib(ts_atoms, name, constraint=None, CUTOFF=50, calc=None):
    from ase.vibrations import Vibrations
    from ase import units
    from ase.io import write
    TS = ts_atoms.copy()
    TS.calc = calc
    if constraint is not None:
        TS.set_constraint(constraint)
    vib = Vibrations(TS, name=name)
    vib.run()
    all_frequencies = vib.get_frequencies()
    imag_indices = np.where(all_frequencies.imag > CUTOFF)[0]
    num_imag_freqs = len(imag_indices)
    if num_imag_freqs > 0:
        for index in imag_indices:
            images_generator = vib.get_vibrations().iter_animated_mode(index, temperature=units.kB * 300, frames=30)
            images_list = list(images_generator)
            output_xyz_file = f'{vib.name}.{index % len(vib.get_energies())}.xyz'
            write(output_xyz_file, images_list)
            vib.clean()
    return num_imag_freqs
