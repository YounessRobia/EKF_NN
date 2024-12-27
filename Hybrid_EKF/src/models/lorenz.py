"""Lorenz system model with time-varying parameters."""

import numpy as np
from typing import Tuple

class TimeVaryingParameters:
    """Time-varying parameter generator for Lorenz system."""
    
    def __init__(self, 
                 sigma_mean: float = 10.0,
                 rho_mean: float = 28.0,
                 beta_mean: float = 8/3,
                 variation_amplitude: float = 0.2):
        self.sigma_mean = sigma_mean
        self.rho_mean = rho_mean
        self.beta_mean = beta_mean
        self.amplitude = variation_amplitude
        
    def get_parameters(self, t: float) -> Tuple[float, float, float]:
        """Generate time-varying parameters."""
        sigma = self.sigma_mean * (1 + self.amplitude * np.sin(0.1 * t))
        rho = self.rho_mean * (1 + self.amplitude * (
            0.7 * np.sin(0.3 * t) + 
            0.3 * np.sin(0.7 * t)
        ))
        beta_variation = 0.5 * self.amplitude * (
            np.sin(0.2 * t) + 
            np.sin(0.4 * t + np.pi/4)
        )
        beta = self.beta_mean * (1 + beta_variation)
        
        return sigma, rho, beta

class LorenzSystem:
    """Lorenz system with time-varying parameters."""
    
    def __init__(self, 
                 sigma: float = 10.0,
                 rho: float = 28.0,
                 beta: float = 8/3,
                 time_varying: bool = False,
                 variation_amplitude: float = 0.2):
        self.base_sigma = sigma
        self.base_rho = rho
        self.base_beta = beta
        self.time_varying = time_varying
        
        if time_varying:
            self.param_generator = TimeVaryingParameters(
                sigma_mean=sigma,
                rho_mean=rho,
                beta_mean=beta,
                variation_amplitude=variation_amplitude
            )
    
    def get_current_parameters(self, t: float) -> Tuple[float, float, float]:
        """Get current parameter values."""
        if self.time_varying:
            return self.param_generator.get_parameters(t)
        return self.base_sigma, self.base_rho, self.base_beta
        
    def derivatives(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Compute derivatives with current parameters."""
        x, y, z = state
        sigma, rho, beta = self.get_current_parameters(t)
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        return np.array([dx, dy, dz])
    
    def dynamics(self, state: np.ndarray, dt: float = 0.01, t: float = 0.0) -> np.ndarray:
        """RK4 integration with time-dependent dynamics."""
        def rk4_step(s, t):
            k1 = self.derivatives(s, t)
            k2 = self.derivatives(s + dt * k1/2, t + dt/2)
            k3 = self.derivatives(s + dt * k2/2, t + dt/2)
            k4 = self.derivatives(s + dt * k3, t + dt)
            return (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return state + rk4_step(state, t)
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix for EKF."""
        x, y, z = state
        
        return np.array([
            [-self.base_sigma, self.base_sigma, 0],
            [self.base_rho - z, -1, -x],
            [y, x, -self.base_beta]
        ]) 