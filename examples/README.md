# Examples

This directory contains small runnable demos showing how to use shared tools:

- Interpolation & IDPP: shared/interp.py
- Frequency analysis: shared/freq.py
- IRC run and plotting: shared/irc.py

Notes:
- Dependencies are imported lazily inside functions to minimize global load.
- For frequency demos, a simple calculator like EMT is used for convenience.
- For IRC demos, the sella package is required; the script exits gracefully if missing.

