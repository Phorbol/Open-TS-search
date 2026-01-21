# Examples

This directory contains small runnable demos showing how to use shared tools:

- Interpolation & IDPP: shared/interp.py
- Frequency analysis: shared/freq.py
- IRC run and plotting: shared/irc.py
- CCQN GPU optimization with MACE: examples/ccqn_gpu_demo.py
- Full Workflow (CCQN -> Freq -> IRC) with MACE: examples/full_irc_gpu_demo.py
  - Supports both 'fake' (default) and 'true' (Sella) IRC modes via `--irc-type` argument.

Notes:
- Dependencies are imported lazily inside functions to minimize global load.
- For frequency demos, a simple calculator like EMT is used for convenience.
- For IRC demos, the sella package is required; the script exits gracefully if missing.
- For CCQN/Freq/IRC GPU demos, mace-torch and a CUDA-capable GPU are required.
- The GPU Freq/IRC demos run a CCQN optimization first to find the TS from the IS structure.

