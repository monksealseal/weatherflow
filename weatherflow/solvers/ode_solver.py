# Copyright (c) 2024 WeatherFlow
# Implementation inspired by Meta's flow matching approach

import torch
from torch import Tensor
import numpy as np
from typing import Callable, Optional, Tuple, Dict, List, Union
from torchdiffeq import odeint

class WeatherODESolver:
    """ODE solver for weather flow matching with physics constraints.
    
    This solver integrates ordinary differential equations for weather modeling
    with built-in physics constraints to ensure conservation laws are respected.
    """
    
    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = 'dopri5',
        physics_constraints: bool = True,
        constraint_types: Optional[List[str]] = None,
        constraint_weights: Optional[Dict[str, float]] = None,
        grid_spacing: Optional[Tuple[float, float]] = None
    ):
        """Initialize the ODE solver.
        
        Args:
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            method: Integration method ('dopri5', 'rk4', etc.)
            physics_constraints: Whether to apply physics constraints
            constraint_types: List of constraint types to apply
                (e.g., ['mass', 'energy', 'vorticity'])
            constraint_weights: Dictionary of weights for each constraint type
            grid_spacing: Tuple of (latitude, longitude) grid spacing in radians
        """
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.physics_constraints = physics_constraints
        
        # Default constraint types if none provided
        self.constraint_types = constraint_types or ['mass', 'energy']
        
        # Default weights if none provided
        self.constraint_weights = constraint_weights or {
            'mass': 1.0,
            'energy': 0.5,
            'vorticity': 0.2
        }
        
        # Grid spacing for finite difference approximations
        self.grid_spacing = grid_spacing
        
        # Statistics for monitoring
        self.stats = {
            'constraint_violations': [],
            'energy_conservation': []
        }
    
    def solve(
        self,
        velocity_fn: Callable,
        x0: Tensor,
        t_span: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Solve the ODE system with weather-specific handling.
        
        Args:
            velocity_fn: Function computing velocity field
            x0: Initial conditions of shape [batch_size, channels, height, width]
            t_span: Time points to solve for
            **kwargs: Additional args for velocity function
            
        Returns:
            Tuple of (solution trajectory, solver stats)
        """
        # Reset statistics
        self.stats = {
            'constraint_violations': [],
            'energy_conservation': []
        }
        
        # Store initial energy for conservation tracking
        if 'energy' in self.constraint_types:
            initial_energy = self._compute_total_energy(x0)
        
        def ode_func(t: Tensor, x: Tensor) -> Tensor:
            # Reshape if needed for the velocity function
            orig_shape = x.shape
            
            # Compute velocity field
            v = velocity_fn(x, t, **kwargs)
            
            if self.physics_constraints:
                # Apply physics constraints (conservation laws)
                v = self._apply_physics_constraints(v, x)
                
                # Track constraint violations
                if 'mass' in self.constraint_types:
                    div = self._compute_divergence(v)
                    self.stats['constraint_violations'].append(torch.mean(torch.abs(div)).item())
                
                if 'energy' in self.constraint_types:
                    current_energy = self._compute_total_energy(x)
                    energy_change = torch.abs(current_energy - initial_energy) / (initial_energy + 1e-8)
                    self.stats['energy_conservation'].append(energy_change.item())
            
            return v
        
        # Solve the ODE
        try:
            solution = odeint(
                ode_func,
                x0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method
            )
            
            success = True
            message = "Integration successful"
        except Exception as e:
            # Handle integration failures
            solution = torch.zeros((len(t_span), *x0.shape), device=x0.device)
            solution[0] = x0  # At least include initial condition
            success = False
            message = str(e)
        
        # Prepare statistics
        stats = {
            "success": success,
            "message": message,
            "constraint_violations": np.mean(self.stats['constraint_violations']) if self.stats['constraint_violations'] else 0.0,
            "energy_conservation": np.mean(self.stats['energy_conservation']) if self.stats['energy_conservation'] else 1.0
        }
        
        return solution, stats
    
    def _compute_divergence(self, v: Tensor) -> Tensor:
        """Compute divergence of velocity field.
        
        Args:
            v: Velocity field tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Divergence field
        """
        # Assume first two channels are u,v components (east-west, north-south)
        if v.dim() <= 3 or v.shape[1] < 2:
            return torch.zeros(v.shape[0], 1, device=v.device)
        
        # Compute spatial derivatives
        du_dx = torch.gradient(v[:, 0], dim=2)[0]  # Longitude gradient of u
        dv_dy = torch.gradient(v[:, 1], dim=1)[0]  # Latitude gradient of v
        
        # Apply grid spacing if available
        if self.grid_spacing:
            du_dx = du_dx / self.grid_spacing[1]
            dv_dy = dv_dy / self.grid_spacing[0]
        
        # Divergence is the sum of the derivatives
        return du_dx + dv_dy
    
    def _compute_curl(self, v: Tensor) -> Tensor:
        """Compute curl (vorticity) of velocity field.
        
        Args:
            v: Velocity field tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Vorticity field
        """
        # Assume first two channels are u,v components
        if v.dim() <= 3 or v.shape[1] < 2:
            return torch.zeros(v.shape[0], 1, device=v.device)
        
        # Compute spatial derivatives
        du_dy = torch.gradient(v[:, 0], dim=1)[0]  # Latitude gradient of u
        dv_dx = torch.gradient(v[:, 1], dim=2)[0]  # Longitude gradient of v
        
        # Apply grid spacing if available
        if self.grid_spacing:
            du_dy = du_dy / self.grid_spacing[0]
            dv_dx = dv_dx / self.grid_spacing[1]
        
        # Curl in 2D is dv/dx - du/dy
        return dv_dx - du_dy
    
    def _compute_total_energy(self, x: Tensor) -> Tensor:
        """Compute total energy of the system.
        
        Args:
            x: State tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Total energy
        """
        # Simple kinetic energy computation (sum of squared values)
        # In a real implementation, this would include potential energy terms
        return torch.sum(x**2, dim=[1, 2, 3])
    
    def _apply_physics_constraints(self, v: Tensor, x: Tensor) -> Tensor:
        """Apply physics-based constraints to velocity field.
        
        Args:
            v: Velocity field tensor
            x: Current state tensor
            
        Returns:
            Constrained velocity field
        """
        # Apply mass conservation (divergence-free constraint)
        if 'mass' in self.constraint_types:
            # Compute divergence
            div = self._compute_divergence(v)
            
            # Project out the divergent component (Helmholtz decomposition)
            # This is a simplified version - a full implementation would solve
            # a Poisson equation to get the exact projection
            if v.dim() > 3 and v.shape[1] >= 2:
                # Apply gradient of divergence (approximate projection)
                div_grad_x = torch.gradient(div, dim=2)[0]
                div_grad_y = torch.gradient(div, dim=1)[0]
                
                # Scale by weight
                weight = self.constraint_weights.get('mass', 1.0)
                
                # Remove divergent component
                v = v.clone()  # Create a copy to avoid in-place modification
                v[:, 0] = v[:, 0] - weight * div_grad_x
                v[:, 1] = v[:, 1] - weight * div_grad_y
        
        # Apply energy conservation
        if 'energy' in self.constraint_types:
            # Normalize velocity to maintain energy
            v_norm = torch.norm(v.reshape(v.shape[0], -1), dim=1)
            
            # Scale velocities to maintain energy (soft constraint)
            target_norm = torch.sqrt(self._compute_total_energy(x))
            scale = target_norm / (v_norm + 1e-8)
            
            # Apply scaling with weight
            weight = self.constraint_weights.get('energy', 0.5)
            for i in range(v.shape[0]):
                # Interpolate between original and scaled velocity
                v[i] = (1 - weight) * v[i] + weight * (v[i] * scale[i])
        
        # Apply vorticity conservation
        if 'vorticity' in self.constraint_types and v.dim() > 3 and v.shape[1] >= 2:
            # This is a simplified approach - a full implementation would
            # enforce conservation of potential vorticity
            curl = self._compute_curl(v)
            
            # Compute gradient of curl
            curl_grad_x = torch.gradient(curl, dim=2)[0]
            curl_grad_y = torch.gradient(curl, dim=1)[0]
            
            # Create a rotational correction (perpendicular to gradient)
            weight = self.constraint_weights.get('vorticity', 0.2)
            v = v.clone()  # Create a copy to avoid in-place modification
            v[:, 0] = v[:, 0] + weight * curl_grad_y
            v[:, 1] = v[:, 1] - weight * curl_grad_x
        
        return v
