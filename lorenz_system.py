import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8/3

class LorenzSystem:
    """
    Implementation of the Lorenz system with configurable numerical integration methods.
    
    Attributes:
        params (LorenzParams): Parameters for the Lorenz system
        integration_method (str): Choice of numerical integration method ('euler' or 'rk4')
    """
    
    def __init__(self, params: Optional[LorenzParams] = None, integration_method: str = 'rk4'):
        self.params = params if params is not None else LorenzParams()
        self.integration_method = integration_method.lower()
        if self.integration_method not in ['euler', 'rk4']:
            raise ValueError("Integration method must be either 'euler' or 'rk4'")
    
    def _validate_state(self, state: np.ndarray) -> None:
        """Validate the input state dimensions."""
        if not isinstance(state, np.ndarray):
            raise TypeError("State must be a numpy array")
        if state.shape != (3,):
            raise ValueError("State must be a 3D vector")
    
    def _compute_derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute the derivatives of the Lorenz system."""
        x, y, z = state
        
        dx = self.params.sigma * (y - x)
        dy = x * (self.params.rho - z) - y
        dz = x * y - self.params.beta * z
        
        return np.array([dx, dy, dz])
    
    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Perform one step of RK4 integration."""
        k1 = self._compute_derivatives(state)
        k2 = self._compute_derivatives(state + dt * k1/2)
        k3 = self._compute_derivatives(state + dt * k2/2)
        k4 = self._compute_derivatives(state + dt * k3)
        
        return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def dynamics(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Compute the next state using the chosen integration method.
        
        Args:
            state: Current state vector [x, y, z]
            dt: Time step for integration
            
        Returns:
            np.ndarray: Next state vector
        """
        self._validate_state(state)
        
        if dt <= 0:
            raise ValueError("Time step must be positive")
            
        if self.integration_method == 'euler':
            derivatives = self._compute_derivatives(state)
            next_state = state + dt * derivatives
        else:  # rk4
            next_state = self._rk4_step(state, dt)
            
        return next_state
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian of the Lorenz system.
        
        Args:
            state: Current state vector [x, y, z]
            
        Returns:
            np.ndarray: 3x3 Jacobian matrix
        """
        self._validate_state(state)
        x, y, z = state
        
        J = np.array([
            [-self.params.sigma, self.params.sigma, 0],
            [self.params.rho - z, -1, -x],
            [y, x, -self.params.beta]
        ])
        return J


class ResidualNetwork(nn.Module):
    """
    Enhanced Residual Network with dropout and layer normalization.
    
    Args:
        state_dim: Dimension of input/output state
        hidden_dim: Dimension of hidden layers
        dropout_rate: Dropout probability
    """
    
    def __init__(self, state_dim: int = 3, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Initialize weights using He initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection and proper reshaping.
        
        Args:
            x: Input tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Tensor of shape (batch_size, state_dim) or (state_dim,)
        """
        # Handle both single sample and batch inputs
        input_is_batch = len(x.shape) == 2
        
        if not input_is_batch:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Apply network
        out = x + self.net(x)  # Residual connection
        
        if not input_is_batch:
            out = out.squeeze(0)  # Remove batch dimension for single sample
            
        return out