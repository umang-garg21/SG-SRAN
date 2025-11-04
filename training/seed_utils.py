# -*-coding:utf-8 -*-
"""
File:        seed_utils.py
Created at:  2025/11/03
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Utilities for ensuring reproducibility across all experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.
    
    This ensures that:
    - Python's built-in random module is seeded
    - NumPy's random generator is seeded
    - PyTorch's CPU operations are seeded
    - PyTorch's CUDA operations are seeded (if available)
    - CuDNN backend behavior is deterministic
    
    Parameters
    ----------
    seed : int
        Random seed value (default: 42)
    """
    print(f"\n{'='*80}")
    print(f"SETTING GLOBAL SEED: {seed}")
    print(f"{'='*80}")
    
    # Python random module
    random.seed(seed)
    print(f"✓ Python random seed set to {seed}")
    
    # NumPy
    np.random.seed(seed)
    print(f"✓ NumPy random seed set to {seed}")
    
    # PyTorch CPU
    torch.manual_seed(seed)
    print(f"✓ PyTorch CPU seed set to {seed}")
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        print(f"✓ PyTorch CUDA seed set to {seed} (all devices)")
        
        # Make CuDNN deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ CuDNN deterministic mode enabled")
    
    print(f"{'='*80}\n")


def get_seed_from_config(cfg, default: int = 42) -> int:
    """
    Extract seed from configuration object. Always returns 42 for consistency.
    
    Parameters
    ----------
    cfg : ConfigNamespace or dict
        Configuration object
    default : int
        Default seed value (always 42)
        
    Returns
    -------
    int
        Seed value (always 42)
    """
    # Always use seed 42 for reproducibility
    return 42
