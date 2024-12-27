"""
Advanced Hybrid Extended Kalman Filter Tutorial with Deep Learning and Uncertainty Quantification

This tutorial implements a sophisticated hybrid state estimation system that combines:
1. Traditional Extended Kalman Filtering
2. Physics-informed neural networks
3. Uncertainty quantification
4. Deep learning-based corrections

The system is demonstrated on the Lorenz attractor system.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional, Tuple, Deque
import logging
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#################################
# Part 1: System Configuration  #
#################################

@dataclass
class HybridEKFConfig:
    """Configuration for the enhanced Hybrid EKF system."""
    state_dim: int = 3
    measurement_dim: int = 2  # Partial observation (x, z)
    hidden_dim: int = 64
    sequence_length: int = 100
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
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            logging.error(f"Invalid state detected: {state}")
            return np.zeros_like(state)  # Reset to safe values
            
        return np.clip(state, self.min_bound, self.max_bound)
    
    def validate_covariance(self, P: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix stability."""
        # Ensure symmetry
        P = (P + P.T) / 2
        
        # Clip eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        eigenvalues = np.clip(eigenvalues, self.min_covariance, self.max_covariance)
        P = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return P

#################################
# Part 2: Lorenz System Model   #
#################################

class LorenzSystem:
    """Implementation of the Lorenz system with configurable parameters."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def dynamics(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Compute system dynamics with stability safeguards."""
        def derivatives(s):
            x, y, z = s
            # Add stability bounds
            x = np.clip(x, -50.0, 50.0)
            y = np.clip(y, -50.0, 50.0)
            z = np.clip(z, -50.0, 50.0)
            
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return np.array([dx, dy, dz])
        
        # RK4 with stability checks
        k1 = derivatives(state)
        k2 = derivatives(state + dt * k1/2)
        k3 = derivatives(state + dt * k2/2)
        k4 = derivatives(state + dt * k3)
        
        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Validate result
        if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 50.0):
            logging.warning(f"State instability detected, reverting to previous state")
            return state
            
        return next_state
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix for EKF."""
        x, y, z = state
        
        J = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])
        return J

#################################
# Part 3: Neural Networks       #
#################################

class PhysicsInformedNetwork(nn.Module):
    """Neural network with physics-based regularization."""
    
    def __init__(self, state_dim: int = 3, hidden_dim: int = 64):
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
            nn.Sigmoid()
        )
        
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, physics_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with physics guidance."""
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

class EnhancedUncertaintyNetwork(nn.Module):
    """Advanced uncertainty network with attention mechanisms and multi-head estimation."""
    
    def __init__(self, 
                 state_dim: int = 3,
                 measurement_dim: int = 2,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 sequence_length: int = 5):
        super().__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.sequence_length = sequence_length
        
        # State encoder with attention
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1
        )
        
        # Process noise estimation
        self.q_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        
        # Measurement noise estimation
        self.r_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, measurement_dim * measurement_dim)
        )
        
        # Uncertainty confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                state_sequence: torch.Tensor,
                innovation_sequence: torch.Tensor,
                dt: float = 0.01) -> dict:
        """Forward pass with enhanced uncertainty estimation."""
        batch_size = state_sequence.size(1) if len(state_sequence.shape) > 2 else 1
        
        # Encode state sequence
        encoded_states = self.state_encoder(state_sequence)
        
        # Apply self-attention
        attended_states, _ = self.attention(
            encoded_states,
            encoded_states,
            encoded_states
        )
        
        # Process through LSTM
        lstm_out, _ = self.lstm(attended_states)
        hidden = lstm_out[-1]
        
        # Estimate process noise covariance
        q_params = self.q_estimator(hidden)
        Q_mean = q_params.view(batch_size, self.state_dim, self.state_dim)
        Q_mean = 0.5 * (Q_mean + Q_mean.transpose(-2, -1))
        Q_mean = F.softplus(Q_mean) * dt
        
        # Estimate measurement noise covariance
        r_params = self.r_estimator(hidden)
        R_mean = r_params.view(batch_size, self.measurement_dim, self.measurement_dim)
        R_mean = 0.5 * (R_mean + R_mean.transpose(-2, -1))
        R_mean = F.softplus(R_mean)
        
        # Estimate uncertainties
        confidence = self.confidence_estimator(hidden)
        q_confidence, r_confidence = confidence[:, 0], confidence[:, 1]
        
        Q_std = (1 - q_confidence).unsqueeze(-1).unsqueeze(-1) * Q_mean
        R_std = (1 - r_confidence).unsqueeze(-1).unsqueeze(-1) * R_mean
        
        return {
            'Q_mean': Q_mean,
            'Q_std': Q_std,
            'R_mean': R_mean,
            'R_std': R_std,
            'q_confidence': q_confidence,
            'r_confidence': r_confidence
        }
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss."""
        all_params = []
        for param in self.parameters():
            if param.requires_grad:
                all_params.append(param.view(-1))
        
        l2_loss = torch.norm(torch.cat(all_params))
        return 0.01 * l2_loss

#################################
# Part 4: Hybrid EKF Class      #
#################################

class IntegratedHybridEKF:
    """Enhanced Hybrid EKF with uncertainty quantification and physics-informed learning."""
    
    def __init__(self, system: LorenzSystem, config: HybridEKFConfig):
        self.system = system
        self.config = config
        
        # Initialize state estimation
        self.x_hat = np.zeros(config.state_dim)
        self.P = np.eye(config.state_dim)
        
        # Initialize neural networks
        self.physics_net = PhysicsInformedNetwork(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.uncertainty_net = EnhancedUncertaintyNetwork(
            state_dim=config.state_dim,
            measurement_dim=config.measurement_dim,
            hidden_dim=config.hidden_dim,
            sequence_length=config.sequence_length
        )
        
        self.state_validator = StateValidator(
            min_bound=-50.0,
            max_bound=50.0,
            min_covariance=config.min_covariance,
            max_covariance=1e3
        )
        
        # Initialize optimizers
        self.physics_optimizer = torch.optim.AdamW(
            self.physics_net.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        self.uncertainty_optimizer = torch.optim.AdamW(
            self.uncertainty_net.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Initialize buffers and metrics
        self.state_buffer = deque(maxlen=config.sequence_length)
        self.innovation_buffer = deque(maxlen=config.sequence_length)
        self.metrics = {
            'prediction_loss': [],
            'uncertainty_loss': [],
            'q_confidence': [],
            'r_confidence': []
        }
    
    def predict(self, dt: float) -> None:
        """Prediction step with uncertainty quantification."""
        try:
            # Physics-based prediction
            physics_pred = self.system.dynamics(self.x_hat, dt)
            
            # Neural correction
            x_tensor = torch.FloatTensor(self.x_hat)
            physics_tensor = torch.FloatTensor(physics_pred)
            
            with torch.no_grad():
                corrected_state, physics_loss = self.physics_net(x_tensor, physics_tensor)
                self.x_hat = corrected_state.numpy()
            
            # Get uncertainty estimates with proper reshaping
            uncertainty = self._get_uncertainty_estimates()
            Q_raw = uncertainty['Q_mean'].numpy() + uncertainty['Q_std'].numpy()
            Q = Q_raw.squeeze()  # Remove batch dimensions
            if Q.ndim > 2:
                Q = Q[0]  # Take first slice if still has extra dimensions
            Q = Q.reshape(self.config.state_dim, self.config.state_dim)
            
            # Update state covariance
            F = self.system.jacobian(self.x_hat)
            self.P = F @ self.P @ F.T + Q
            
            # Ensure P is 2D and properly shaped
            self.P = self.P.squeeze()
            if self.P.ndim > 2:
                self.P = self.P.reshape(self.config.state_dim, self.config.state_dim)
            
            # Ensure numerical stability
            self.P = np.maximum(
                (self.P + self.P.T) / 2,
                self.config.min_covariance * np.eye(self.config.state_dim)
            )
            
            self._update_buffers(self.x_hat)
            
        except Exception as e:
            logging.error(f"Error in prediction step: {str(e)}")
            logging.error(f"Matrix shapes:")
            logging.error(f"Q: {Q.shape if 'Q' in locals() else 'not created'}")
            logging.error(f"P: {self.P.shape}")
            logging.error(f"F: {F.shape if 'F' in locals() else 'not created'}")
            raise
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with robust dimension handling and stability checks."""
        try:
            # Validate inputs and ensure 1D arrays
            measurement = self.state_validator.validate_state(measurement).squeeze()
            self.x_hat = self.state_validator.validate_state(self.x_hat).squeeze()
            self.P = self.state_validator.validate_covariance(self.P).squeeze()
            
            # Debug dimensions
            logging.debug(f"Initial x_hat shape: {self.x_hat.shape}")
            logging.debug(f"Initial P shape: {self.P.shape}")
            
            # Construct measurement matrix (2x3)
            H = np.zeros((self.config.measurement_dim, self.config.state_dim))
            H[0, 0] = 1.0  # Observe x
            H[1, 2] = 1.0  # Observe z
            
            # Extract measured components (2,)
            measured_components = np.array([measurement[0], measurement[2]], dtype=np.float64)
            
            # Get uncertainty estimates with bounds
            uncertainty = self._get_uncertainty_estimates()
            R_raw = uncertainty['R_mean'].numpy() + uncertainty['R_std'].numpy()
            
            # Ensure R has correct shape (2x2)
            R = R_raw.squeeze()
            if R.ndim > 2:
                R = R[0]  # Take first slice if still has extra dimensions
            R = R.reshape(self.config.measurement_dim, self.config.measurement_dim)
            R = np.maximum(R, self.config.min_covariance * np.eye(self.config.measurement_dim))
            
            # Ensure P is 2D (3x3)
            if self.P.ndim > 2:
                self.P = self.P.reshape(self.config.state_dim, self.config.state_dim)
            
            # Compute PHt (3x2)
            PHt = self.P @ H.T
            
            # Compute innovation covariance S (2x2)
            S = (H @ PHt + R).squeeze()
            if S.ndim > 2:
                S = S.reshape(self.config.measurement_dim, self.config.measurement_dim)
            S = (S + S.T) / 2  # Ensure symmetry
            
            # Compute Kalman gain (3x2)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
                
            K = PHt @ S_inv
            if K.ndim > 2:
                K = K.reshape(self.config.state_dim, self.config.measurement_dim)
            
            # Compute innovation (2,)
            innovation = measured_components - (H @ self.x_hat)
            
            # Update state (3,)
            self.x_hat = self.x_hat + (K @ innovation).squeeze()
            
            # Update covariance
            I = np.eye(self.config.state_dim)
            KH = K @ H  # (3x3)
            self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
            
            # Ensure final matrices have correct shapes
            self.x_hat = self.x_hat.reshape(self.config.state_dim)
            self.P = self.P.reshape(self.config.state_dim, self.config.state_dim)
            
            # Final validation
            self.x_hat = self.state_validator.validate_state(self.x_hat)
            self.P = self.state_validator.validate_covariance(self.P)
            
            # Log shapes for debugging
            logging.debug(f"Final shapes:")
            logging.debug(f"x_hat: {self.x_hat.shape}")
            logging.debug(f"P: {self.P.shape}")
            logging.debug(f"K: {K.shape}")
            logging.debug(f"H: {H.shape}")
            logging.debug(f"R: {R.shape}")
            
            # Update buffers and learning only if state is valid
            if not np.any(np.isnan(self.x_hat)):
                self._update_buffers(self.x_hat, innovation)
                self._online_learning(measurement, innovation, uncertainty)
                
        except Exception as e:
            logging.error(f"Error in update step: {str(e)}")
            logging.error(f"Matrix shapes:")
            logging.error(f"P: {self.P.shape}")
            logging.error(f"H: {H.shape if 'H' in locals() else 'not created'}")
            logging.error(f"K: {K.shape if 'K' in locals() else 'not created'}")
            logging.error(f"R: {R.shape if 'R' in locals() else 'not created'}")
            logging.error(f"S: {S.shape if 'S' in locals() else 'not created'}")
            logging.error(f"innovation: {innovation.shape if 'innovation' in locals() else 'not created'}")
            raise
        
    def _get_uncertainty_estimates(self) -> Dict[str, torch.Tensor]:
        """Get uncertainty estimates from the network."""
        if len(self.state_buffer) < self.config.sequence_length:
            return {
                'Q_mean': torch.eye(self.config.state_dim) * self.config.process_noise_init,
                'Q_std': torch.zeros(self.config.state_dim, self.config.state_dim),
                'R_mean': torch.eye(self.config.measurement_dim) * self.config.measurement_noise_init,
                'R_std': torch.zeros(self.config.measurement_dim, self.config.measurement_dim),
                'q_confidence': torch.tensor([0.5]),
                'r_confidence': torch.tensor([0.5])
            }
        
        # Convert list to numpy array first
        state_sequence = np.array(list(self.state_buffer))
        innovation_sequence = np.array(list(self.innovation_buffer))
        
        # Then convert to tensor and add batch dimension
        state_sequence = torch.from_numpy(state_sequence).float().unsqueeze(1)
        innovation_sequence = torch.from_numpy(innovation_sequence).float().unsqueeze(1)
        
        with torch.no_grad():
            return self.uncertainty_net(state_sequence, innovation_sequence, self.config.dt)
    
    def _update_buffers(self, state: np.ndarray, innovation: Optional[np.ndarray] = None):
        """Update state and innovation buffers."""
        self.state_buffer.append(state)
        if innovation is not None:
            self.innovation_buffer.append(innovation)
    
    def _online_learning(self, 
                        measurement: np.ndarray, 
                        innovation: np.ndarray,
                        uncertainty: Dict[str, torch.Tensor]) -> None:
        """Perform online learning for both networks."""
        # Physics network learning
        self.physics_optimizer.zero_grad()
        
        x_tensor = torch.FloatTensor(self.x_hat)
        physics_pred = torch.FloatTensor(self.system.dynamics(self.x_hat))
        measurement_tensor = torch.FloatTensor(measurement)
        
        corrected_state, physics_loss = self.physics_net(x_tensor, physics_pred)
        prediction_loss = torch.mean((corrected_state - measurement_tensor)**2)
        
        physics_total_loss = prediction_loss + 0.1 * physics_loss
        physics_total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(), 1.0)
        self.physics_optimizer.step()
        
        # Uncertainty network learning
        if len(self.state_buffer) >= self.config.sequence_length:
            self.uncertainty_optimizer.zero_grad()
            
            state_sequence = torch.FloatTensor(list(self.state_buffer)).unsqueeze(1)
            innovation_sequence = torch.FloatTensor(list(self.innovation_buffer)).unsqueeze(1)
            
            current_uncertainty = self.uncertainty_net(
                state_sequence, 
                innovation_sequence,
                self.config.dt
            )
            
            nll_loss = self._compute_uncertainty_loss(
                innovation,
                current_uncertainty['R_mean'],
                current_uncertainty['R_std']
            )
            
            uncertainty_loss = nll_loss + self.uncertainty_net.get_regularization_loss()
            uncertainty_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.uncertainty_net.parameters(), 1.0)
            self.uncertainty_optimizer.step()
            
            # Update metrics
            self.metrics['prediction_loss'].append(prediction_loss.item())
            self.metrics['uncertainty_loss'].append(uncertainty_loss.item())
            self.metrics['q_confidence'].append(uncertainty['q_confidence'].item())
            self.metrics['r_confidence'].append(uncertainty['r_confidence'].item())
    
    def _compute_uncertainty_loss(self,
                                innovation: np.ndarray,
                                R_mean: torch.Tensor,
                                R_std: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss for uncertainty estimation."""
        innovation_tensor = torch.FloatTensor(innovation)
        R_total = R_mean + R_std
        R_total = torch.clamp(R_total, min=self.config.min_covariance)
        
        log_likelihood = -0.5 * (
            torch.log(2 * torch.tensor(np.pi)) +
            torch.log(torch.det(R_total)) +
            innovation_tensor @ torch.inverse(R_total) @ innovation_tensor
        )
        
        return -log_likelihood

#################################
# Part 5: Simulation Functions  #
#################################

def simulate_system(
    config: HybridEKFConfig,
    time_steps: int = 5000,
    save_path: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Run complete system simulation."""
    # Initialize system and estimator
    system = LorenzSystem()
    estimator = IntegratedHybridEKF(system, config)
    
    # Storage arrays
    true_states = np.zeros((time_steps, config.state_dim))
    estimated_states = np.zeros_like(true_states)
    uncertainties = np.zeros((time_steps, 2))  # Q and R confidences
    
    # Initial state
    true_state = np.array([1.0, 1.0, 1.0])
    
    for t in range(time_steps):
        if t % 100 == 0:
            logging.info(f"Simulation step {t}/{time_steps}")
        
        # System evolution
        true_state = system.dynamics(true_state, config.dt)
        true_states[t] = true_state
        
        # Generate noisy measurement
        full_measurement = true_state + np.random.normal(
            0, config.measurement_noise_init, 
            size=config.state_dim
        )
        
        # Estimation
        estimator.predict(config.dt)
        estimator.update(full_measurement)
        
        # Store results
        estimated_states[t] = estimator.x_hat
        metrics = estimator.metrics
        if metrics['q_confidence'] and metrics['r_confidence']:
            uncertainties[t] = [
                metrics['q_confidence'][-1],
                metrics['r_confidence'][-1]
            ]
    
    results = {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'uncertainties': uncertainties,
        'metrics': metrics
    }
    
    if save_path:
        np.savez(save_path, **results)
    
    return results

#################################
# Part 6: Visualization Tools   #
#################################

class ResultVisualizer:
    """Visualization tools for analyzing estimation results."""
    
    def __init__(self, results: Dict[str, np.ndarray]):
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
    
    def plot_state_trajectories(self, save_path: Optional[Path] = None):
        """Plot state trajectories with uncertainty bounds."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        for i, label in enumerate(['x', 'y', 'z']):
            axes[i].plot(self.results['true_states'][:, i], 
                        'k-', label='True', linewidth=2)
            axes[i].plot(self.results['estimated_states'][:, i],
                        'r--', label='Estimated')
            
            # Add uncertainty bounds
            std = np.sqrt(self.results['uncertainties'][:, 0])  # Q confidence
            axes[i].fill_between(
                range(len(self.results['estimated_states'])),
                self.results['estimated_states'][:, i] - 2*std,
                self.results['estimated_states'][:, i] + 2*std,
                alpha=0.2, color='r'
            )
            
            axes[i].set_ylabel(f'State {label}')
            axes[i].legend()
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('State Trajectories with Uncertainty Bounds')
        
        if save_path:
            plt.savefig(save_path / 'state_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_uncertainty_evolution(self, save_path: Optional[Path] = None):
        """Plot evolution of uncertainty estimates."""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.results['uncertainties'][:, 0],
                label='Process Noise Confidence')
        plt.plot(self.results['uncertainties'][:, 1],
                label='Measurement Noise Confidence')
        
        plt.xlabel('Time Step')
        plt.ylabel('Confidence')
        plt.title('Evolution of Uncertainty Estimates')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path / 'uncertainty_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_metrics(self, save_path: Optional[Path] = None):
        """Plot learning metrics over time."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        axes[0].plot(self.results['metrics']['prediction_loss'],
                    label='Prediction Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Prediction Network Learning')
        axes[0].legend()
        
        axes[1].plot(self.results['metrics']['uncertainty_loss'],
                    label='Uncertainty Loss')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Uncertainty Network Learning')
        axes[1].legend()
        
        if save_path:
            plt.savefig(save_path / 'learning_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

#################################
# Part 7: Main Execution        #
#################################

def main():
    """Main execution function."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configure the system
    config = HybridEKFConfig(
        state_dim=3,
        measurement_dim=2,
        sequence_length=5,
        dt=0.01,
        learning_rate=0.001
    )
    
    # Create output directory
    output_dir = Path('hybrid_ekf_results')
    output_dir.mkdir(exist_ok=True)
    
    # Run simulation
    logging.info("Starting simulation...")
    results = simulate_system(
        config=config,
        time_steps=5000,
        save_path=output_dir / 'simulation_results.npz'
    )
    
    # Visualize results
    logging.info("Generating visualizations...")
    visualizer = ResultVisualizer(results)
    visualizer.plot_state_trajectories(output_dir)
    visualizer.plot_uncertainty_evolution(output_dir)
    visualizer.plot_learning_metrics(output_dir)
    
    # Compute final metrics
    rmse = np.sqrt(np.mean((results['true_states'] - results['estimated_states'])**2))
    final_uncertainty = np.mean(results['uncertainties'][-100:], axis=0)
    
    logging.info(f"\nFinal Results:")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"Final Process Noise Confidence: {final_uncertainty[0]:.4f}")
    logging.info(f"Final Measurement Noise Confidence: {final_uncertainty[1]:.4f}")

if __name__ == "__main__":
    main()