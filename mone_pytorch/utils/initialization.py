import math
import torch

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """
    Variance scaling initializer in PyTorch, inspired by JAX's variance_scaling initializer.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to initialize in-place.
    scale : float, optional
        Scaling factor (default: 1.0).
    mode : str, optional
        One of 'fan_in', 'fan_out', or 'fan_avg' (default: 'fan_in').
    distribution : str, optional
        One of 'normal', 'uniform', or 'truncated_normal' (default: 'normal').
    
    Returns
    -------
    torch.Tensor
        The input tensor initialized in-place.
    
    Raises
    ------
    ValueError
        If an invalid mode or distribution is specified.
    """
    # Compute fan_in and fan_out
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    
    if mode == 'fan_in':
        denominator = fan_in
    elif mode == 'fan_out':
        denominator = fan_out
    elif mode == 'fan_avg':
        denominator = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'fan_in', 'fan_out', or 'fan_avg'.")
    
    variance = scale / denominator
    
    if distribution == 'normal':
        # Normal distribution with std = sqrt(variance)
        std = math.sqrt(variance)
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    elif distribution == 'uniform':
        # Uniform distribution in [-limit, limit] with limit = sqrt(3 * variance)
        limit = math.sqrt(3.0 * variance)
        return torch.nn.init.uniform_(tensor, a=-limit, b=limit)
    elif distribution == 'truncated_normal':
        # Truncated normal with std = sqrt(variance)
        # PyTorch's trunc_normal_ truncates at 2 std by default.
        std = math.sqrt(variance)
        return torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std)
    else:
        raise ValueError(f"Invalid distribution '{distribution}'. Use 'normal', 'uniform', or 'truncated_normal'.")
    
def lecun_normal_(tensor):
    """
    Lecun normal initializer in PyTorch, inspired by JAX's lecun_normal initializer.
    """
    return variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal')
