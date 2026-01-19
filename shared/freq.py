import os
import shutil
import numpy as np
from ase.vibrations import Vibrations
from ase import units
from ase.io import write

def write_animated_mode_xyz(vib_object, atoms, index, file_prefix, is_imag):
    """
    Generate multi-frame XYZ animation file using iter_animated_mode and ase.io.write.
    """
    try:
        # Note: Removing temperature/frames args for compatibility as per user code
        # Ideally, we should check ASE version or use default if possible.
        # User code specifically removed them to avoid TypeError, so I will follow that.
        images_generator = vib_object.get_vibrations().iter_animated_mode(index)
    except Exception as e:
        print(f"     [Error] Cannot create animation generator for Mode #{index}. Error: {e}")
        return

    try:
        images_list = list(images_generator)
    except Exception as e:
        print(f"     [Error] Cannot convert animation generator to list. Error: {e}")
        return

    tag = "imag" if is_imag else "real"
    # Ensure directory exists
    if not os.path.exists(vib_object.name):
        os.makedirs(vib_object.name)
        
    output_filename = os.path.join(vib_object.name, f'{file_prefix}.{index}.{tag}.xyz')
        
    write(output_filename, images_list)
    print(f"     > Animation for Mode #{index} output to {output_filename}")

def get_vib_mode(ts_atoms, constraint=None, name='vib_check', cutoff_cm=50.0, 
                 output_all_imag_modes=False, output_all_real_modes=False):
    """
    [XYZ Output Version] Check first-order saddle point and extract imaginary modes and animations.
    
    Args:
        ts_atoms (ase.Atoms): Structure.
        constraint (ase.constraints.Constraint): Constraint.
        name (str): Directory/prefix for output.
        cutoff_cm (float): Cutoff for imaginary frequency (cm^-1).
        output_all_imag_modes (bool): If True, output animation for all imaginary modes > cutoff.
        output_all_real_modes (bool): If True, output animation for all real modes.
        
    Returns:
        numpy.ndarray or None: Eigenvector of the unique significant imaginary frequency if found.
    """
    # 1. Cleanup
    if os.path.exists(name): shutil.rmtree(name)
    # vib is the default cache dir for some ASE versions/configs, clean it if safe?
    # User script had: if os.path.exists("vib"): shutil.rmtree("vib")
    # We will skip cleaning "vib" globally to be safer, unless name='vib'.
    
    print(f"\n>>> [Vib-Check] Starting Hessian calculation for {name}...")
    
    atoms_vib = ts_atoms.copy()
    atoms_vib.calc = ts_atoms.calc
    
    if constraint is not None:
        atoms_vib.set_constraint(constraint)
        print(f"     Constraint {constraint} applied.")

    # 2. Run Vibrations
    vib = Vibrations(atoms_vib, name=name)
    vib.run()
    
    # 3. Analyze frequencies
    energies = vib.get_energies()
    
    imag_candidates = []
    real_modes_to_output = []
    print(f">>> [Vib-Check] Frequency Analysis (Cutoff = {cutoff_cm} cm^-1):")
    
    for i, e in enumerate(energies):
        if e.imag > 0:
            freq_val = abs(e) / units.invcm
            
            if freq_val > cutoff_cm:
                mode_vec = vib.get_mode(i)
                print(f"     [!] Found Imaginary Mode #{i}: {freq_val:.2f}i cm^-1")
                imag_candidates.append((i, freq_val, mode_vec))
                
                if output_all_imag_modes:
                    write_animated_mode_xyz(vib, atoms_vib, i, name, is_imag=True)
            else:
                print(f"     [Ignored] Noise Imaginary Mode #{i}: {freq_val:.2f}i cm^-1")
        
        elif output_all_real_modes and e.real > 0:
            freq_val = e.real / units.invcm
            print(f"     [Real] Mode #{i}: {freq_val:.2f} cm^-1")
            real_modes_to_output.append(i)

    # 4. Logic & Output
    num_imag = len(imag_candidates)
    result_vec = None
    
    print("\n" + "-"*30)
    
    if num_imag == 0:
        print(">>> [Result] No significant imaginary frequencies found. (MINIMUM)")
    
    elif num_imag == 1:
        idx, val, vec = imag_candidates[0]
        print(">>> [Result] Confirmed First-Order Saddle Point.")
        print(f"     Reaction Coordinate: Mode #{idx} ({val:.1f}i cm^-1)")
        
        if not output_all_imag_modes:
             write_animated_mode_xyz(vib, atoms_vib, idx, name, is_imag=True)
        
        result_vec = vec
        
    elif num_imag > 1:
        print(f">>> [Result] Higher-Order Saddle Point found ({num_imag} modes).")
        for idx, val, _ in imag_candidates:
            print(f"     Mode {idx}: {val:.1f}i")
        
        if not output_all_imag_modes:
            print("     > No imaginary mode animations output (set output_all_imag_modes=True to output all).")

    # 5. Real modes
    if output_all_real_modes:
        print("\n>>> [Output] Generating animations for all real modes...")
        for i in real_modes_to_output:
            write_animated_mode_xyz(vib, atoms_vib, i, name, is_imag=False)
            
    # 6. Clean
    vib.clean()
    print("------------------------------")
    
    return result_vec

def ase_vib(ts_atoms, name, constraint=None, CUTOFF=50, calc=None):
    # Keep existing implementation for compatibility
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
