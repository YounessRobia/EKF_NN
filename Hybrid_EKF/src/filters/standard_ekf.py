"""Standard Extended Kalman Filter implementation."""

import numpy as np
from typing import Optional
import logging
from .base import BaseFilter
from ..models.lorenz import LorenzSystem

class StandardEKF(BaseFilter):
    """Traditional Extended Kalman Filter implementation."""
    
    def __init__(self, system: LorenzSystem, state_dim: int = 3):
        super().__init__(state_dim)
        self.system = system
    
    def predict(self, dt: float) -> None:
        """Prediction step using system dynamics."""
        try:
            # State prediction
            self.x_hat = self.system.dynamics(self.x_hat, dt)
            
            # Covariance prediction
            F = self.system.jacobian(self.x_hat)
            Q = np.eye(self.state_dim) * 0.1  # Process noise
            self.P = F @ self.P @ F.T + Q
            
            self._validate_matrices()
            
        except Exception as e:
            logging.error(f"Error in StandardEKF prediction: {str(e)}")
            raise
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with measurement."""
        try:
            # Measurement model
            H = np.zeros((2, self.state_dim))
            H[0,0] = 1.0  # Observe x
            H[1,2] = 1.0  # Observe z
            
            R = np.eye(2) * 0.1  # Measurement noise
            
            # Kalman gain computation
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # State update
            innovation = np.array([measurement[0], measurement[2]]) - H @ self.x_hat
            self.x_hat = self.x_hat + K @ innovation
            
            # Covariance update
            self.P = (np.eye(self.state_dim) - K @ H) @ self.P
            
            self._validate_matrices()
            
        except Exception as e:
            logging.error(f"Error in StandardEKF update: {str(e)}")
            raise 