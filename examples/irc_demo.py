import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ase.build import molecule
from ase.calculators.emt import EMT
from shared.irc import get_clean_irc_path, plot_irc

def run_irc():
    try:
        import sella
    except Exception:
        print('sella not available, skipping IRC demo')
        return
    ts = molecule('H2')
    ts.calc = EMT()
    path = get_clean_irc_path(ts, irc_log_prefix='examples/irc_demo', fmax=0.05, steps=50, dx=0.1, eta=1e-4, ninner_iter=50)
    if path is None:
        print('IRC path not generated')
        return
    plot_irc(path, title='irc_demo')
    print(f'IRC path images: {len(path)}')

if __name__ == '__main__':
    run_irc()
