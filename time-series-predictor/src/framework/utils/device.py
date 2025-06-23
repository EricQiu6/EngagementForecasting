import torch
from typing import Optional


def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Get the best available device for computation.
    
    Args:
        preferred_device: Preferred device ('cuda', 'cpu', or None)
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if preferred_device == 'cpu':
        return 'cpu'
    
    if preferred_device == 'cuda' or preferred_device is None:
        if torch.cuda.is_available():
            return 'cuda'
        else:
            if preferred_device == 'cuda':
                print("Warning: CUDA requested but not available. Using CPU.")
            return 'cpu'
    
    return preferred_device


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': get_device(),
        'devices': []
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            info['devices'].append(device_info)
    
    return info


def print_device_info():
    """Print device information in a readable format."""
    info = get_device_info()
    
    print(f"Current device: {info['current_device']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA devices: {info['device_count']}")
        for device in info['devices']:
            print(f"  Device {device['id']}: {device['name']}")
            print(f"    Total memory: {device['memory_total'] / 1e9:.1f} GB")
            print(f"    Allocated: {device['memory_allocated'] / 1e6:.1f} MB")
            print(f"    Cached: {device['memory_cached'] / 1e6:.1f} MB")
    else:
        print("  No CUDA devices available")


def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    else:
        print("CUDA not available - no cache to clear") 