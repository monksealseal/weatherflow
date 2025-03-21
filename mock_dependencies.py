#!/usr/bin/env python3
"""
Add mock implementations of dependencies that might not be available.
This allows notebooks to run without having to install all dependencies.
"""

def install_mock_torchdiffeq():
    """
    Install a mock version of torchdiffeq that will allow notebooks to run.
    """
    import sys
    
    # Check if torchdiffeq is already installed
    try:
        import torchdiffeq
        print("torchdiffeq is already installed")
        return
    except ImportError:
        pass
    
    # Create a mock torchdiffeq module
    import types
    import torch
    
    # Create the mock module
    torchdiffeq = types.ModuleType("torchdiffeq")
    sys.modules["torchdiffeq"] = torchdiffeq
    
    # Add odeint function
    def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, **kwargs):
        """Mock implementation of torchdiffeq.odeint"""
        print("Using mock torchdiffeq.odeint (not actually solving ODE)")
        batch_size = y0.shape[0]
        t_size = t.shape[0]
        result = []
        
        # Just interpolate between initial state and some "evolved" state
        for i in range(t_size):
            t_val = t[i]
            # Apply the dynamics function at a simple level - just one step
            v = func(t_val, y0)
            state = y0 + t_val * v * 0.1  # Simplified evolution
            result.append(state)
            
        return torch.stack(result)
    
    # Add the function to the module
    torchdiffeq.odeint = odeint
    
    print("Mock torchdiffeq module installed successfully!")

def install_mock_cartopy():
    """
    Install a mock version of cartopy that will allow notebooks to run.
    """
    import sys
    
    # Check if cartopy is already installed
    try:
        import cartopy
        print("cartopy is already installed")
        return
    except ImportError:
        pass
    
    # Create mock cartopy module
    import types
    import matplotlib.pyplot as plt
    import numpy as np
    
    cartopy = types.ModuleType("cartopy")
    sys.modules["cartopy"] = cartopy
    
    # Add crs module
    crs = types.ModuleType("cartopy.crs")
    sys.modules["cartopy.crs"] = crs
    
    # Add basic projections
    class PlateCarree:
        def __init__(self, central_longitude=0.0, **kwargs):
            self.central_longitude = central_longitude
    
    class Robinson:
        def __init__(self, central_longitude=0.0, **kwargs):
            self.central_longitude = central_longitude
    
    class Orthographic:
        def __init__(self, central_longitude=0.0, central_latitude=0.0, **kwargs):
            self.central_longitude = central_longitude
            self.central_latitude = central_latitude
    
    # Add to crs module
    crs.PlateCarree = PlateCarree
    crs.Robinson = Robinson
    crs.Orthographic = Orthographic
    
    # Add feature module
    feature = types.ModuleType("cartopy.feature")
    sys.modules["cartopy.feature"] = feature
    
    class NaturalEarthFeature:
        def __init__(self, category, name, scale, **kwargs):
            self.category = category
            self.name = name
            self.scale = scale
    
    feature.NaturalEarthFeature = NaturalEarthFeature
    feature.LAND = NaturalEarthFeature('physical', 'land', '110m')
    feature.COASTLINE = NaturalEarthFeature('physical', 'coastline', '110m')
    feature.BORDERS = NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '110m')
    
    # Monkeypatch matplotlib to handle cartopy axes
    original_add_subplot = plt.Figure.add_subplot
    
    def patched_add_subplot(self, *args, projection=None, **kwargs):
        if projection is not None and not isinstance(projection, str):
            # If it's a cartopy projection, ignore it and use default
            return original_add_subplot(self, *args, **kwargs)
        return original_add_subplot(self, *args, projection=projection, **kwargs)
    
    plt.Figure.add_subplot = patched_add_subplot
    
    print("Mock cartopy module installed successfully!")

def install_all_mocks():
    """Install all mock dependencies."""
    print("Installing mock dependencies...")
    install_mock_torchdiffeq()
    install_mock_cartopy()
    print("Mock dependencies installation complete!")

if __name__ == "__main__":
    install_all_mocks()