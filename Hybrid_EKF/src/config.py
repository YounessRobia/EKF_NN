"""Configuration classes and constants for the Hybrid EKF system."""

from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class HybridEKFConfig:
    """Configuration for the enhanced Hybrid EKF system."""
    state_dim: int = 3
    measurement_dim: int = 2  # Partial observation (x, z)
    hidden_dim: int = 64
    sequence_length: int = 20
    learning_rate: float = 0.001
    dt: float = 0.01
    min_covariance: float = 1e-10
    process_noise_init: float = 0.1
    measurement_noise_init: float = 0.1

class CovarianceUpdateMethod(Enum):
    """Methods for covariance update in Kalman filter."""
    STANDARD = "standard"
    JOSEPH = "joseph"
    SQRT = "sqrt"

class StateValidator:
    """State validation and bounds checking."""
    
    def __init__(self, 
                 min_bound: float = -1e3,
                 max_bound: float = 1e3,
                 min_covariance: float = 1e-10,
                 max_covariance: float = 1e3):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.min_covariance = min_covariance
        self.max_covariance = max_covariance
    
    def validate_state(self, state: np.ndarray) -> np.ndarray:
        """Validate and clip state values."""
        return np.clip(state, self.min_bound, self.max_bound)
    
    def validate_covariance(self, P: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix stability."""
        P = (P + P.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        eigenvalues = np.clip(eigenvalues, self.min_covariance, self.max_covariance)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T 