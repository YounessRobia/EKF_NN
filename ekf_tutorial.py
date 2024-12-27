import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###################
# 1. System Setup #
###################

class LorenzSystem:
    """Lorenz system for demonstration."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        """
        Initialize Lorenz system parameters.
        
        Args:
            sigma: First parameter
            rho: Second parameter (relates to convection)
            beta: Third parameter
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def dynamics(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Compute Lorenz system dynamics using RK4 integration.
        
        Args:
            state: Current state [x, y, z]
            dt: Time step
            
        Returns:
            Next state after dt
        """
        def derivatives(s):
            x, y, z = s
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return np.array([dx, dy, dz])
        
        # RK4 integration
        k1 = derivatives(state)
        k2 = derivatives(state + dt * k1/2)
        k3 = derivatives(state + dt * k2/2)
        k4 = derivatives(state + dt * k3)
        
        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Add bounds checking
        if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 1e6):
            logging.warning(f"Numerical instability detected at state: {state}")
            return state  # Return previous state if instability detected
        
        return next_state
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for EKF.
        
        Args:
            state: Current state [x, y, z]
            
        Returns:
            3x3 Jacobian matrix
        """
        x, y, z = state
        
        J = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])
        return J

############################
# 2. Network Architecture  #
############################

class PhysicsInformedNetwork(nn.Module):
    """Neural network with physics-based regularization."""
    
    def __init__(self, state_dim: int = 3, hidden_dim: int = 64):
        """
        Initialize network architecture.
        
        Args:
            state_dim: Dimension of state vector
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.physics_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()  # Gate for physics influence
        )
        
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, x: torch.Tensor, physics_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics guidance.
        
        Args:
            x: Input state
            physics_prior: Physics-based prediction
            
        Returns:
            Tuple of (corrected_state, physics_loss)
        """
        # Encode state
        encoded = self.encoder(x)
        
        # Compute physics attention
        physics_weight = self.physics_attention(encoded)
        
        # Generate correction
        correction = self.correction(encoded)
        
        # Combine with physics
        output = physics_prior + physics_weight * correction
        
        # Physics-based regularization
        physics_loss = torch.mean((correction[1:] - correction[:-1])**2)
        
        return output, physics_loss

#############################
# 3. Hybrid EKF Implementation
#############################

class HybridEKF:
    """Hybrid EKF combining traditional filtering with deep learning."""
    
    def __init__(self, system: LorenzSystem, state_dim: int = 3):
        """
        Initialize hybrid filter.
        
        Args:
            system: Dynamical system model
            state_dim: Dimension of state vector
        """
        self.system = system
        self.state_dim = state_dim
        
        # Initialize state estimate and covariance
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # Initialize neural components
        self.physics_net = PhysicsInformedNetwork(state_dim)
        self.optimizer = torch.optim.AdamW(
            self.physics_net.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
    def predict(self, dt: float) -> None:
        """
        Prediction step with neural correction and stability checks.
        """
        try:
            # Physics-based prediction
            physics_pred = self.system.dynamics(self.x_hat, dt)
            
            # Check for invalid values
            if np.any(np.isnan(physics_pred)) or np.any(np.abs(physics_pred) > 1e6):
                logging.error(f"Invalid prediction detected: {physics_pred}")
                return
            
            # Neural correction
            with torch.no_grad():
                x_tensor = torch.FloatTensor(self.x_hat)
                physics_tensor = torch.FloatTensor(physics_pred)
                corrected_state, _ = self.physics_net(x_tensor, physics_tensor)
                
                # Check neural network output
                if torch.any(torch.isnan(corrected_state)):
                    logging.error("Neural network produced NaN values")
                    return
                    
                self.x_hat = corrected_state.numpy()
            
            # Covariance update
            F = self.system.jacobian(self.x_hat)
            Q = np.eye(self.state_dim) * 0.1  # Process noise
            self.P = F @ self.P @ F.T + Q
            
        except Exception as e:
            logging.error(f"Error in prediction step: {str(e)}")
        
    def update(self, measurement: np.ndarray) -> None:
        """
        Update step with measurement.
        
        Args:
            measurement: Observed state
        """
        try:
            # Measurement matrix for partial state observation (only x and z components)
            H = np.zeros((2, self.state_dim))  # Observe first and last components
            H[0,0] = 1.0  # Observe x
            H[1,2] = 1.0  # Observe z
            
            # Measurement noise - adjust dimensions to match measurement space (2x2)
            R = np.eye(2) * 0.1  # Changed from 3x3 to 2x2
            
            # Compute Kalman gain
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state estimate using x and z measurements
            innovation = np.array([measurement[0], measurement[2]]) - H @ self.x_hat  # Use x,z components
            self.x_hat = self.x_hat + K @ innovation
            
            # Update covariance
            self.P = (np.eye(self.state_dim) - K @ H) @ self.P
            
            # Online learning step
            self._online_learning(measurement)
            
        except Exception as e:
            logging.error(f"Error in update step: {str(e)}")
    
    def _online_learning(self, measurement: np.ndarray) -> None:
        """
        Perform online learning with physics-informed loss.
        
        Args:
            measurement: Current measurement for training
        """
        self.optimizer.zero_grad()
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(self.x_hat)
        physics_pred = torch.FloatTensor(self.system.dynamics(self.x_hat))
        measurement_tensor = torch.FloatTensor(measurement)
        
        # Forward pass
        corrected_state, physics_loss = self.physics_net(x_tensor, physics_pred)
        
        # Compute losses
        prediction_loss = torch.mean((corrected_state - measurement_tensor)**2)
        total_loss = prediction_loss + 0.1 * physics_loss
        
        # Backprop and optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(), 1.0)
        self.optimizer.step()

##############################
# 4. Simulation and Evaluation
##############################

def simulate_system(
    time_steps: int = 10000,
    dt: float = 0.01,
    noise_std: float = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Lorenz system with both filters."""
    # Initialize
    system = LorenzSystem()
    hybrid_estimator = HybridEKF(system)
    standard_estimator = StandardEKF(system)
    true_state = np.array([1.0, 1.0, 1.0])
    
    # Storage
    true_states = np.zeros((time_steps, 3))
    measurements = np.zeros_like(true_states)
    hybrid_estimated_states = np.zeros_like(true_states)
    standard_estimated_states = np.zeros_like(true_states)
    
    for t in range(time_steps):
        # True system evolution
        true_state = system.dynamics(true_state, dt)
        true_states[t] = true_state
        
        # Generate noisy measurement
        measurement = true_state + np.random.normal(0, noise_std, size=3)
        measurements[t] = measurement
        
        # Hybrid estimation
        hybrid_estimator.predict(dt)
        hybrid_estimator.update(measurement)
        hybrid_estimated_states[t] = hybrid_estimator.x_hat
        
        # Standard EKF estimation
        standard_estimator.predict(dt)
        standard_estimator.update(measurement)
        standard_estimated_states[t] = standard_estimator.x_hat
        
        if t % 100 == 0:
            logging.info(f"Step {t}/{time_steps}")
    
    return true_states, measurements, hybrid_estimated_states, standard_estimated_states

def plot_results(
    true_states: np.ndarray,
    measurements: np.ndarray,
    hybrid_estimated_states: np.ndarray,
    standard_estimated_states: np.ndarray,
    save_path: Optional[Path] = None
) -> None:
    """Plot results comparing both filters."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, label in enumerate(['x', 'y', 'z']):
        axes[i].plot(true_states[:, i], 'k-', label='True', linewidth=2)
        axes[i].plot(measurements[:, i], 'r.', label='Measured', alpha=0.3)
        axes[i].plot(hybrid_estimated_states[:, i], 'b--', label='Hybrid EKF')
        axes[i].plot(standard_estimated_states[:, i], 'g:', label='Standard EKF')
        axes[i].set_ylabel(f'State {label}')
        axes[i].legend()
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle('Comparison of Hybrid and Standard EKF Estimation')
    
    if save_path:
        plt.savefig(save_path / 'estimation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_metrics(
    true_states: np.ndarray,
    estimated_states: np.ndarray
) -> Dict[str, float]:
    """
    Compute performance metrics.
    
    Args:
        true_states: Array of true states
        estimated_states: Array of estimated states
        
    Returns:
        Dictionary of metrics
    """
    errors = true_states - estimated_states
    
    metrics = {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(np.abs(errors)),
        'max_error': np.max(np.abs(errors))
    }
    
    return metrics

def plot_phase_space(
    true_states: np.ndarray,
    hybrid_estimated_states: np.ndarray,
    standard_estimated_states: np.ndarray,
    save_path: Optional[Path] = None
) -> None:
    """Plot phase space trajectories for both filters."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot3D(
        true_states[:, 0], 
        true_states[:, 1], 
        true_states[:, 2], 
        'k-', 
        label='True', 
        linewidth=1
    )
    
    ax.plot3D(
        hybrid_estimated_states[:, 0], 
        hybrid_estimated_states[:, 1], 
        hybrid_estimated_states[:, 2], 
        'b--', 
        label='Hybrid EKF', 
        linewidth=1
    )
    
    ax.plot3D(
        standard_estimated_states[:, 0], 
        standard_estimated_states[:, 1], 
        standard_estimated_states[:, 2], 
        'g:', 
        label='Standard EKF', 
        linewidth=1
    )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.title('Phase Space Trajectory Comparison')
    
    if save_path:
        plt.savefig(save_path / 'phase_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


class StandardEKF:
    """Traditional Extended Kalman Filter implementation."""
    
    def __init__(self, system: LorenzSystem, state_dim: int = 3):
        self.system = system
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)
    
    def predict(self, dt: float) -> None:
        """Prediction step using system dynamics."""
        try:
            # State prediction
            self.x_hat = self.system.dynamics(self.x_hat, dt)
            
            # Covariance prediction
            F = self.system.jacobian(self.x_hat)
            Q = np.eye(self.state_dim) * 0.1  # Process noise
            self.P = F @ self.P @ F.T + Q
            
        except Exception as e:
            logging.error(f"Error in StandardEKF prediction: {str(e)}")
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with measurement."""
        try:
            # Same measurement model as hybrid version
            H = np.zeros((2, self.state_dim))
            H[0,0] = 1.0  # Observe x
            H[1,2] = 1.0  # Observe z
            
            R = np.eye(2) * 0.1
            
            # Kalman gain computation
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # State update
            innovation = np.array([measurement[0], measurement[2]]) - H @ self.x_hat
            self.x_hat = self.x_hat + K @ innovation
            
            # Covariance update
            self.P = (np.eye(self.state_dim) - K @ H) @ self.P
            
        except Exception as e:
            logging.error(f"Error in StandardEKF update: {str(e)}")

def main():
    """Main function to run comparison demonstration."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    output_dir = Path('hybrid_ekf_results')
    output_dir.mkdir(exist_ok=True)
    
    # Run simulation
    logging.info("Starting simulation...")
    true_states, measurements, hybrid_states, standard_states = simulate_system()
    
    # Plot results
    logging.info("Plotting comparison results...")
    plot_results(true_states, measurements, hybrid_states, standard_states, output_dir)
    plot_phase_space(true_states, hybrid_states, standard_states, output_dir)
    
    # Compute and display metrics
    hybrid_metrics = compute_metrics(true_states, hybrid_states)
    standard_metrics = compute_metrics(true_states, standard_states)
    
    logging.info("\nPerformance Metrics:")
    logging.info("\nHybrid EKF:")
    for name, value in hybrid_metrics.items():
        logging.info(f"{name}: {value:.4f}")
        
    logging.info("\nStandard EKF:")
    for name, value in standard_metrics.items():
        logging.info(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()