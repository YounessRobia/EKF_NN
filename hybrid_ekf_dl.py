import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import scipy.linalg
from typing import Tuple, Optional, Union, Dict
import logging
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from lorenz_system import LorenzSystem, ResidualNetwork

class CovarianceUpdateMethod(Enum):
    """Methods for covariance update in Kalman filter."""
    STANDARD = "standard"
    JOSEPH = "joseph"
    SQRT = "sqrt"

@dataclass
class EKFConfig:
    """Configuration for the Hybrid EKF-DL system.
    
    Attributes:
        state_dim: Dimension of the state vector
        measurement_dim: Dimension of the measurement vector
        learning_rate: Learning rate for neural networks
        dt: Time step for prediction
        measurement_noise_std: Initial measurement noise standard deviation
        covariance_update_method: Method for covariance update
        min_covariance: Minimum allowed covariance for numerical stability
    """
    state_dim: int = 3
    measurement_dim: int = 3
    learning_rate: float = 0.001
    dt: float = 0.01
    measurement_noise_std: float = 0.1
    covariance_update_method: CovarianceUpdateMethod = CovarianceUpdateMethod.JOSEPH
    min_covariance: float = 1e-10


@dataclass
class SimulationResult:
    """Container for simulation results.
    
    Attributes:
        true_states: Array of true system states
        estimated_states: Array of estimated states
        estimation_errors: Array of estimation errors
    """
    true_states: np.ndarray
    estimated_states: np.ndarray
    estimation_errors: np.ndarray

class UncertaintyNetwork(nn.Module):
    """Enhanced network for uncertainty quantification using probabilistic techniques."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # [mean_Q, std_Q, mean_R, std_R]
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing means and standard deviations for Q and R
        """
        outputs = self.net(x)
        return {
            'Q_mean': torch.exp(outputs[:, 0]),
            'Q_std': torch.exp(outputs[:, 1]),
            'R_mean': torch.exp(outputs[:, 2]),
            'R_std': torch.exp(outputs[:, 3])
        }

class HybridEKFDL:
    """Hybrid Extended Kalman Filter with Deep Learning corrections and uncertainty quantification.
    
    This implementation combines traditional EKF with deep learning for:
    1. State prediction correction
    2. Adaptive noise covariance estimation
    3. Uncertainty quantification
    
    The filter uses various numerical techniques for stability and accuracy:
    - Joseph form for covariance updates
    - Square root formulation option
    - Robust matrix operations
    """
    
    def __init__(self, system: LorenzSystem, config: EKFConfig):
        """Initialize the hybrid filter.
        
        Args:
            system: Dynamical system model
            config: Filter configuration parameters
        """
        self.system = system
        self.config = config
        self._initialize_networks()
        self._initialize_state()
        
    def _initialize_networks(self) -> None:
        """Initialize neural networks with enhanced architectures."""
        self.residual_net = ResidualNetwork(
            state_dim=self.config.state_dim,
            hidden_dim=64,
            dropout_rate=0.1
        )
        self.uncertainty_net = UncertaintyNetwork(self.config.measurement_dim)
        
        # Use AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            list(self.residual_net.parameters()) + 
            list(self.uncertainty_net.parameters()),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def _initialize_state(self) -> None:
        """Initialize state estimate and covariance with validation."""
        self.x_hat = np.zeros(self.config.state_dim)
        self.P = np.eye(self.config.state_dim)
        self._validate_state()
    
    def _validate_state(self) -> None:
        """Validate state and covariance dimensions."""
        if self.x_hat.shape != (self.config.state_dim,):
            raise ValueError(f"Invalid state dimension: {self.x_hat.shape}")
        if self.P.shape != (self.config.state_dim, self.config.state_dim):
            raise ValueError(f"Invalid covariance dimension: {self.P.shape}")
    
    def _ensure_symmetric(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix symmetry for numerical stability."""
        return (matrix + matrix.T) / 2
    
    def _robust_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Compute robust matrix inverse with condition number checking."""
        try:
            # Check condition number
            cond = np.linalg.cond(matrix)
            if cond > 1e12:
                logging.warning(f"Poor conditioning detected: {cond}")
                # Add small regularization
                matrix += np.eye(matrix.shape[0]) * self.config.min_covariance
            
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            logging.error("Matrix inversion failed, using pseudo-inverse")
            return np.linalg.pinv(matrix)
    
    def predict(self, dt: float = 0.01) -> None:
        """Prediction step with DL correction and uncertainty quantification.
        
        Args:
            dt: Time step for prediction
        """
        # Convert state to tensor for DL
        x_tensor = torch.FloatTensor(self.x_hat)
        
        # Get model correction with uncertainty
        with torch.no_grad():
            correction = self.residual_net(x_tensor).numpy()
        
        # Augmented prediction
        f_nominal = self.system.dynamics(self.x_hat, dt)
        self.x_hat = f_nominal + correction
        
        # Compute augmented Jacobian
        F = self.system.jacobian(self.x_hat)
        
        # Estimate process noise covariance with uncertainty
        Q = self._estimate_process_noise_with_uncertainty()
        
        # Covariance prediction with numerical stability
        self.P = self._ensure_symmetric(F @ self.P @ F.T + Q)
        self.P = np.maximum(self.P, self.config.min_covariance * np.eye(self.config.state_dim))
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with adaptive measurement noise and robust computations.
        
        Args:
            measurement: Observation vector
        """
        self._validate_measurement(measurement)
        H = np.eye(self.config.state_dim)  # Assuming full state measurement
        
        # Compute residual
        residual = measurement - self.x_hat
        residual_tensor = torch.FloatTensor(residual)
        
        # Estimate measurement noise covariance with uncertainty
        R = self._estimate_measurement_noise_with_uncertainty(residual_tensor)
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        S = self._ensure_symmetric(S)
        
        # Kalman gain with robust inverse
        K = self.P @ H.T @ self._robust_inverse(S)
        
        # State update
        self.x_hat = self.x_hat + K @ residual
        
        # Covariance update based on selected method
        if self.config.covariance_update_method == CovarianceUpdateMethod.JOSEPH:
            self._joseph_covariance_update(K, H, R)
        elif self.config.covariance_update_method == CovarianceUpdateMethod.SQRT:
            self._sqrt_covariance_update(K, H, R)
        else:
            self._standard_covariance_update(K, H, R)
        
        # Online learning step
        self.online_learning_step(residual_tensor, S)
    
    def _validate_measurement(self, measurement: np.ndarray) -> None:
        """Validate measurement dimensions and values."""
        if measurement.shape != (self.config.measurement_dim,):
            raise ValueError(f"Invalid measurement dimension: {measurement.shape}")
    
    def _joseph_covariance_update(self, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Joseph form covariance update for guaranteed positive definiteness.
        
        Args:
            K: Kalman gain matrix
            H: Measurement matrix
            R: Measurement noise covariance matrix
        """
        I = np.eye(self.config.state_dim)
        temp = (I - K @ H)
        self.P = temp @ self.P @ temp.T + K @ R @ K.T
        self.P = self._ensure_symmetric(self.P)

    def _sqrt_covariance_update(self, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Square root formulation for enhanced numerical stability.
        
        Args:
            K: Kalman gain matrix
            H: Measurement matrix
            R: Measurement noise covariance matrix
        """
        I = np.eye(self.config.state_dim)
        U = scipy.linalg.cholesky(self.P)
        temp = U @ (I - K @ H)
        self.P = temp @ temp.T + K @ R @ K.T
        self.P = self._ensure_symmetric(self.P)

    def _standard_covariance_update(self, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Standard covariance update with symmetry enforcement.
        
        Args:
            K: Kalman gain matrix
            H: Measurement matrix
            R: Measurement noise covariance matrix
        """
        I = np.eye(self.config.state_dim)
        self.P = (I - K @ H) @ self.P
        self.P = self._ensure_symmetric(self.P)
    
    def _estimate_process_noise_with_uncertainty(self) -> np.ndarray:
        """Estimate process noise covariance with uncertainty bounds."""
        with torch.no_grad():
            uncertainty = self.uncertainty_net(torch.zeros(1, self.config.measurement_dim))
            Q_mean = uncertainty['Q_mean'].item()
            Q_std = uncertainty['Q_std'].item()
            Q = np.eye(self.config.state_dim) * max(Q_mean + Q_std, self.config.min_covariance)
            return Q
    
    def _estimate_measurement_noise_with_uncertainty(self, residual: torch.Tensor) -> np.ndarray:
        """Estimate measurement noise covariance with uncertainty bounds."""
        with torch.no_grad():
            uncertainty = self.uncertainty_net(residual.reshape(1, -1))
            R_mean = uncertainty['R_mean'].item()
            R_std = uncertainty['R_std'].item()
            R = np.eye(self.config.state_dim) * max(R_mean + R_std, self.config.min_covariance)
            return R
    
    def online_learning_step(self, residual: torch.Tensor, S: np.ndarray) -> None:
        """Perform online learning with physics-informed loss.
        
        Args:
            residual: Innovation sequence
            S: Innovation covariance
        """
        self.optimizer.zero_grad()
        
        # Prediction loss with physics-informed regularization
        pred_correction = self.residual_net(torch.FloatTensor(self.x_hat))
        physics_consistency = torch.mean((pred_correction[1:] - pred_correction[:-1])**2)
        pred_loss = torch.mean(pred_correction**2) + 0.1 * physics_consistency
        
        # Uncertainty estimation loss
        uncertainty = self.uncertainty_net(residual.reshape(1, -1))
        neg_log_likelihood = 0.5 * (torch.log(uncertainty['R_mean']) + 
                                  residual**2 / uncertainty['R_mean'])
        uncertainty_loss = torch.mean(neg_log_likelihood)
        
        # Combined loss with adaptive weighting
        total_loss = pred_loss + uncertainty_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.uncertainty_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(total_loss)

def simulate_and_estimate(
    system: LorenzSystem,
    config: EKFConfig,
    time_steps: int = 1000,
    initial_state: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> SimulationResult:
    """
    Simulate system and perform estimation with comprehensive diagnostics.
    
    Args:
        system: The dynamical system to simulate
        config: Configuration parameters
        time_steps: Number of simulation steps
        initial_state: Optional initial state
        save_path: Optional path to save results
        
    Returns:
        SimulationResult containing true states, estimates, and errors
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    logging.info("Starting simulation with configuration: %s", config)
    
    if time_steps <= 0:
        raise ValueError(f"Invalid time steps: {time_steps}")
    
    estimator = HybridEKFDL(system, config)
    true_state = initial_state if initial_state is not None else np.array([1.0, 1.0, 1.0])
    
    # Pre-allocate arrays
    estimated_states = np.zeros((time_steps, config.state_dim))
    true_states = np.zeros_like(estimated_states)
    estimation_errors = np.zeros_like(estimated_states)
    
    try:
        for t in range(time_steps):
            if t % 100 == 0:
                logging.info(f"Processing step {t}/{time_steps}")
            
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
            
    except Exception as e:
        logging.error(f"Simulation failed at step {t}: {str(e)}")
        raise
    
    if save_path:
        try:
            np.savez(
                save_path,
                true_states=true_states,
                estimated_states=estimated_states,
                estimation_errors=estimation_errors,
                config=vars(config)
            )
            logging.info(f"Results saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
    
    # Compute performance metrics
    rmse = np.sqrt(np.mean(estimation_errors**2))
    mae = np.mean(np.abs(estimation_errors))
    logging.info(f"Simulation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return SimulationResult(true_states, estimated_states, estimation_errors)

def main():
    """Main entry point with enhanced error handling and diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Configure simulation
        config = EKFConfig(
            covariance_update_method=CovarianceUpdateMethod.JOSEPH,
            min_covariance=1e-10
        )
        system = LorenzSystem()
        time_steps = 1000
        save_path = Path("simulation_results.npz")
        
        # Run simulation with progress tracking
        results = simulate_and_estimate(
            system=system,
            config=config,
            time_steps=time_steps,
            save_path=save_path
        )
        
        # Report final statistics
        final_rmse = np.sqrt(np.mean(results.estimation_errors[-100:]**2))
        logging.info(f"Final RMSE (last 100 steps): {final_rmse:.4f}")
        
        return results
        
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()