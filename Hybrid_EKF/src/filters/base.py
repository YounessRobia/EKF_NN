"""Base class for Kalman filter implementations."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from ..config import StateValidator

class BaseFilter(ABC):
    """Abstract base class for filter implementations."""
    
    def __init__(self, state_dim: int = 3):
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.validator = StateValidator()
    
    @abstractmethod
    def predict(self, dt: float) -> None:
        """Prediction step to be implemented by concrete classes."""
        pass
    
    @abstractmethod
    def update(self, measurement: np.ndarray) -> None:
        """Update step to be implemented by concrete classes."""
        pass
    
    def _validate_matrices(self) -> None:
        """Validate state and covariance matrices."""
        self.x_hat = self.validator.validate_state(self.x_hat)
        self.P = self.validator.validate_covariance(self.P)
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x_hat.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.P.copy()
    
    def reset(self) -> None:
        """Reset filter state."""
        self.x_hat = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) 