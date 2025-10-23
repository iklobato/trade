"""
Random seed management for reproducible results.
"""

import random
import numpy as np
import os
from typing import Optional


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set random seed for all libraries.
    
    Args:
        seed: Random seed value. If None, uses config or default 42.
        
    Returns:
        The seed value used.
    """
    if seed is None:
        seed = int(os.getenv('RANDOM_SEED', 42))
    
    # Set seeds for all libraries
    random.seed(seed)
    np.random.seed(seed)
    
    # Set environment variable for other libraries
    os.environ['RANDOM_SEED'] = str(seed)
    
    return seed


def get_seed() -> int:
    """
    Get current random seed.
    
    Returns:
        Current seed value.
    """
    return int(os.getenv('RANDOM_SEED', 42))
