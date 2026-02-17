import os
import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt

def get_clean_irc_path(ts_atoms, irc_log_prefix="irc", fmax=0.05, steps=1000, dx=0.1, eta=1e-4, ninner_iter=100):
    from sella import IRC
    forward_traj_file = f"{irc_log_prefix}_forward.traj"
    reverse_traj_file = f"{irc_log_prefix}_reverse.traj"
    output_dir = os.path.dirname(irc_log_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    TS = ts_atoms.copy()
    TS.calc = ts_atoms.calc
    irc_forward = IRC(TS, trajectory=forward_traj_file, dx=dx, eta=eta, ninner_iter=ninner_iter, logfile=f"{irc_log_prefix}_forward.log")
    irc_forward.run(fmax=fmax, steps=steps, direction='forward')
    TS = ts_atoms.copy()
    TS.calc = ts_atoms.calc
    irc_reverse = IRC(TS, trajectory=reverse_traj_file, dx=dx, eta=eta, ninner_iter=ninner_iter, logfile=f"{irc_log_prefix}_reverse.log")
    irc_reverse.run(fmax=fmax, steps=steps, direction='reverse')
    forward_path = read(forward_traj_file, index=":")
    reverse_path = read(reverse_traj_file, index=":")
    if not forward_path or not reverse_path:
        return None
    reverse_path.reverse()
    full_path = reverse_path + forward_path[1:]
    return full_path

def mass_weighted_path(traj):
    masses = traj[0].get_masses()
    s = [0.0]
    for i in range(1, len(traj)):
        dr = traj[i].positions - traj[i-1].positions
        step = np.sqrt((masses[:, None] * dr**2).sum())
        s.append(s[-1] + step)
    return np.array(s)

def combine_irc(reverse_traj, forward_traj):
    reverse = read(reverse_traj, ":")
    forward = read(forward_traj, ":")
    if not reverse or not forward:
        raise RuntimeError("Empty IRC trajectory")
    reverse.reverse()
    full = reverse + forward[1:]
    return full

def plot_irc(full_traj, title="IRC Path"):
    energies = np.array([atoms.get_potential_energy() for atoms in full_traj])
    s = mass_weighted_path(full_traj)
    energies -= energies.min()
    plt.figure(figsize=(6,4))
    plt.plot(s, energies, "o-", color="C0")
    plt.xlabel("Reaction coordinate (mass-weighted, Å·sqrt(u))")
    plt.ylabel("Energy (eV, relative)")
    context = f"delta_E = {energies[-1] - energies[0]:.4f} eV, barrier = {np.max(energies) - energies[0]:.4f} eV, reverse barrier = {np.max(energies) - energies[-1]:.4f} eV"
    plt.title(context)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    write(f"IRC_{title}.xyz", full_traj)
