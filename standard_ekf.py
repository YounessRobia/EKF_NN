import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import logging
from lorenz_system import LorenzSystem

@dataclass
class StandardEKFConfig:
    """Configuration for standard EKF."""
    state_dim: int = 3
    measurement_dim: int = 3
    dt: float = 0.01
    process_noise_std: float = 0.1
    measurement_noise_std: float = 0.1
    min_covariance: float = 1e-10

class StandardEKF:
    """Standard Extended Kalman Filter implementation for comparison."""
    
    def __init__(self, system: LorenzSystem, config: StandardEKFConfig):
        self.system = system
        self.config = config
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize state estimate and covariance."""
        self.x_hat = np.zeros(self.config.state_dim)
        self.P = np.eye(self.config.state_dim)
    
    def _ensure_symmetric(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix symmetry."""
        return (matrix + matrix.T) / 2
    
    def predict(self, dt: float = 0.01) -> None:
        """Prediction step."""
        # State prediction
        self.x_hat = self.system.dynamics(self.x_hat, dt)
        
        # Compute Jacobian
        F = self.system.jacobian(self.x_hat)
        
        # Process noise
        Q = np.eye(self.config.state_dim) * (self.config.process_noise_std ** 2)
        
        # Covariance prediction
        self.P = self._ensure_symmetric(F @ self.P @ F.T + Q)
        self.P = np.maximum(self.P, self.config.min_covariance * np.eye(self.config.state_dim))
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step."""
        H = np.eye(self.config.state_dim)  # Full state measurement
        R = np.eye(self.config.state_dim) * (self.config.measurement_noise_std ** 2)
        
        # Innovation
        y = measurement - self.x_hat
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x_hat = self.x_hat + K @ y
        
        # Covariance update (Joseph form)
        I = np.eye(self.config.state_dim)
        temp = (I - K @ H)
        self.P = self._ensure_symmetric(temp @ self.P @ temp.T + K @ R @ K.T)

def simulate_standard_ekf(
    system: LorenzSystem,
    config: StandardEKFConfig,
    time_steps: int = 1000,
    initial_state: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run simulation with standard EKF."""
    
    estimator = StandardEKF(system, config)
    true_state = initial_state if initial_state is not None else np.array([1.0, 1.0, 1.0])
    
    # Pre-allocate arrays
    estimated_states = np.zeros((time_steps, config.state_dim))
    true_states = np.zeros_like(estimated_states)
    estimation_errors = np.zeros_like(estimated_states)
    
    for t in range(time_steps):
        # System simulation
        true_state = system.dynamics(true_state, config.dt)
        measurement = true_state + np.random.normal(
            0, config.measurement_noise_std, 
            size=config.state_dim
        )
        
        # Estimation
        estimator.predict(config.dt)
        estimator.update(measurement)
        
        # Store results
        estimated_states[t] = estimator.x_hat
        true_states[t] = true_state
        estimation_errors[t] = true_state - estimator.x_hat
    
    if save_path:
        np.savez(
            save_path,
            true_states=true_states,
            estimated_states=estimated_states,
            estimation_errors=estimation_errors,
            config=vars(config)
        )
    
    return true_states, estimated_states, estimation_errors 