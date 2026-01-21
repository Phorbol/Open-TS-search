import os
import random
import numpy as np
import torch

def set_deterministic(seed: int = 42):
    """
    Sets seeds and configures PyTorch/CUDA for maximum reproducibility.
    
    This function should be called at the very beginning of the program,
    before any other torch/numpy operations.
    
    Args:
        seed: The random seed to use.
    """
    # 1. Set Environment Variables for cuBLAS Determinism
    # Must be set BEFORE torch is used for matrix operations
    # :4096:8 allocates a fixed workspace buffer to avoid atomic non-determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. Python & NumPy Seeds
    random.seed(seed)
    np.random.seed(seed)

    # 3. PyTorch Seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 4. Enforce Deterministic Algorithms
    # This ensures that operations like scatter_add, index_add, and matmul 
    # use deterministic implementations (potentially slower).
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older torch versions might not have this, or have it under set_deterministic
        pass
    
    # 5. Cudnn Benchmarking
    # Disable auto-tuner which might select different kernels on different runs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 6. Matmul Precision (Optional, for Ampere+ GPUs)
        # Force float32 matmul to use highest precision (disable TF32)
        # This reduces numerical noise significantly
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')

    print(f"[CCQN-GPU] Deterministic mode enabled. Seed: {seed}")
