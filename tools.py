import numpy as np

def denormalize(norm_val, low, high):
    """
    Denormalize a value or array that was normalized to [-1, 1].
    Formula: raw = ((norm + 1)/2) * (high - low) + low
    """
    return ((norm_val + 1) / 2) * (high - low) + low