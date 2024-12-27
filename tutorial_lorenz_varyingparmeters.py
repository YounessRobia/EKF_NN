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
        
    def get_parameters(self, t: float) -> tuple:
        """
        Generate time-varying parameters.
        
        Args:
            t: Current time
            
        Returns:
            Tuple of (sigma, rho, beta) at time t
        """
        # Slow variation in sigma (periodic)
        sigma = self.sigma_mean * (1 + self.amplitude * np.sin(0.1 * t))
        
        # Faster variation in rho with two frequencies
        rho = self.rho_mean * (1 + self.amplitude * (
            0.7 * np.sin(0.3 * t) + 
            0.3 * np.sin(0.7 * t)
        ))
        
        # Random walk variation in beta (bounded)
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
        """
        Initialize Lorenz system.
        
        Args:
            sigma, rho, beta: Initial parameters
            time_varying: Whether to use time-varying parameters
            variation_amplitude: Amplitude of parameter variations
        """
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
    
    def get_current_parameters(self, t: float) -> tuple:
        """Get current parameter values."""
        if self.time_varying:
            return self.param_generator.get_parameters(t)
        return self.base_sigma, self.base_rho, self.base_beta
        
    def derivatives(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Compute derivatives with current parameters."""
        x, y, z = state
        
        # Get current parameters
        sigma, rho, beta = self.get_current_parameters(t)
        
        # Standard Lorenz terms with current parameters
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
            
        next_state = state + rk4_step(state, t)
        
        if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 1e6):
            logging.warning(f"Numerical instability detected at state: {state}")
            return state
            
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
            [-self.base_sigma, self.base_sigma, 0],
            [self.base_rho - z, -1, -x],
            [y, x, -self.base_beta]
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
            min_bound=-1e4,
            max_bound=1000.0,
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
            logging.debug(f"S: {S.shape}")
            
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


class UnscentedKF:
    """Unscented Kalman Filter implementation."""
    
    def __init__(self, system: LorenzSystem, state_dim: int = 3):
        self.system = system
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # UKF parameters
        self.alpha = 0.3  # Primary scaling parameter
        self.beta = 2.0   # Secondary scaling parameter
        self.kappa = 0.0  # Tertiary scaling parameter
        
        # Derived parameters
        self.n = state_dim
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Calculate weights
        self.weights_m = np.zeros(2 * self.n + 1)
        self.weights_c = np.zeros(2 * self.n + 1)
        
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.weights_m[i] = 1.0 / (2 * (self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]
        
        # Add numerical stability parameters
        self.min_cov_eigenval = 1e-10
        self.regularization_noise = 1e-8
        
    def ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is symmetric and positive definite."""
        # Force symmetry
        matrix = (matrix + matrix.T) / 2
        
        # Add small regularization to diagonal
        matrix += np.eye(matrix.shape[0]) * self.regularization_noise
        
        # Check eigenvalues and adjust if necessary
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        if np.any(eigenvals < self.min_cov_eigenval):
            eigenvals = np.maximum(eigenvals, self.min_cov_eigenval)
            matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        return matrix
    
    def safe_cholesky(self, matrix: np.ndarray) -> np.ndarray:
        """Compute Cholesky decomposition with added stability."""
        try:
            matrix = self.ensure_positive_definite(matrix)
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            # If still failing, add more regularization
            matrix += np.eye(matrix.shape[0]) * self.regularization_noise * 10
            return np.linalg.cholesky(matrix)
    
    def generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points using current state estimate."""
        n = self.state_dim
        sigma_points = np.zeros((2 * n + 1, n))
        
        # Mean point
        sigma_points[0] = self.x_hat
        
        try:
            # Stabilized matrix square root calculation
            scaled_cov = (self.n + self.lambda_) * self.P
            S = self.safe_cholesky(scaled_cov)
            
            # Generate remaining points
            for i in range(n):
                sigma_points[i + 1] = self.x_hat + S[i]
                sigma_points[n + i + 1] = self.x_hat - S[i]
                
        except Exception as e:
            logging.error(f"Error in sigma point generation: {str(e)}")
            # Fall back to current state estimate
            sigma_points = np.tile(self.x_hat, (2 * n + 1, 1))
            
        return sigma_points
    
    def predict(self, dt: float) -> None:
        """Prediction step using sigma points with stability checks."""
        try:
            # Generate sigma points
            sigma_points = self.generate_sigma_points()
            
            # Propagate sigma points through dynamics
            propagated_points = np.array([
                self.system.dynamics(point, dt) for point in sigma_points
            ])
            
            # Calculate predicted mean
            self.x_hat = np.sum(self.weights_m[:, None] * propagated_points, axis=0)
            
            # Calculate predicted covariance with stability enforcement
            self.P = np.zeros((self.n, self.n))
            for i in range(len(sigma_points)):
                diff = propagated_points[i] - self.x_hat
                self.P += self.weights_c[i] * np.outer(diff, diff)
            
            # Ensure covariance stays positive definite
            self.P = self.ensure_positive_definite(self.P)
            
            # Add process noise
            Q = np.eye(self.state_dim) * 0.1
            self.P += Q
            
        except Exception as e:
            logging.error(f"Error in UKF prediction: {str(e)}")
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with measurement and stability checks."""
        try:
            H = np.zeros((2, self.state_dim))
            H[0,0] = 1.0
            H[1,2] = 1.0
            
            sigma_points = self.generate_sigma_points()
            projected_points = np.array([H @ point for point in sigma_points])
            
            # Calculate measurement mean
            y_mean = np.sum(self.weights_m[:, None] * projected_points, axis=0)
            
            # Calculate covariances with stability enforcement
            Pyy = np.zeros((2, 2))
            Pxy = np.zeros((self.n, 2))
            
            for i in range(len(sigma_points)):
                diff_y = projected_points[i] - y_mean
                diff_x = sigma_points[i] - self.x_hat
                Pyy += self.weights_c[i] * np.outer(diff_y, diff_y)
                Pxy += self.weights_c[i] * np.outer(diff_x, diff_y)
            
            # Ensure measurement covariance is positive definite
            Pyy = self.ensure_positive_definite(Pyy)
            
            # Add measurement noise
            R = np.eye(2) * 0.1
            Pyy += R
            
            # Compute Kalman gain with stable inverse
            try:
                K = Pxy @ np.linalg.inv(Pyy)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if regular inverse fails
                K = Pxy @ np.linalg.pinv(Pyy)
            
            # Update state and covariance
            innovation = np.array([measurement[0], measurement[2]]) - y_mean
            self.x_hat = self.x_hat + K @ innovation
            
            # Joseph form for covariance update (more stable)
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
            
            # Final stability enforcement
            self.P = self.ensure_positive_definite(self.P)
            
        except Exception as e:
            logging.error(f"Error in UKF update: {str(e)}")
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

def simulate_comparative_system(
    time_steps: int = 5000,
    dt: float = 0.01,
    noise_type: str = 'mixture',
    config: Optional[HybridEKFConfig] = None
) -> Dict[str, np.ndarray]:
    """Run simulation with all three filters."""
    
    # Initialize systems with parameter variation
    true_system = LorenzSystem(time_varying=True, variation_amplitude=0.1)
    filter_system = LorenzSystem(time_varying=False)  # Filters assume constant parameters
    
    # Create default config if none provided
    if config is None:
        config = HybridEKFConfig()
    
    # Initialize all three filters
    hybrid_filter = IntegratedHybridEKF(filter_system, config)
    standard_filter = StandardEKF(true_system)
    ukf_filter = UnscentedKF(filter_system)  # Add UKF
    
    # Initial state and storage
    true_state = np.array([1.0, 1.0, 1.0])
    t = 0.0
    
    results = {
        'true_states': np.zeros((time_steps, 3)),
        'hybrid_states': np.zeros((time_steps, 3)),
        'standard_states': np.zeros((time_steps, 3)),
        'ukf_states': np.zeros((time_steps, 3)),  # Add UKF storage
        'hybrid_std': np.zeros((time_steps, 3)),
        'ukf_std': np.zeros((time_steps, 3)),     # Add UKF uncertainty
        'measurements': np.zeros((time_steps, 3))
    }
    
    for step in range(time_steps):
        if step % 100 == 0:
            logging.info(f"Simulation step {step}/{time_steps}")
        # True system evolution
        true_state = true_system.dynamics(true_state, dt, t)
        results['true_states'][step] = true_state
        
        # Generate non-Gaussian measurement noise
        noise = generate_non_gaussian_noise(3, noise_type)
        measurement = true_state + noise
        results['measurements'][step] = measurement
        
        # Update both filters
        hybrid_filter.predict(dt)
        hybrid_filter.update(measurement)
        
        standard_filter.predict(dt)
        standard_filter.update(measurement)
        
        ukf_filter.predict(dt)
        ukf_filter.update(measurement)
        
        # Store results
        results['hybrid_states'][step] = hybrid_filter.x_hat
        results['standard_states'][step] = standard_filter.x_hat
        results['ukf_states'][step] = ukf_filter.x_hat
        results['hybrid_std'][step] = np.sqrt(np.diag(hybrid_filter.P))
        results['ukf_std'][step] = np.sqrt(np.diag(ukf_filter.P))
        
        t += dt
        
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

class ComparisonVisualizer:
    """Enhanced visualization for filter comparison."""
    
    def __init__(self, results: Dict[str, np.ndarray]):
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
    
    def plot_comparative_trajectories(self, save_path: Optional[Path] = None):
        """Plot state trajectories for all three filters."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        for i, label in enumerate(['x', 'y', 'z']):
            axes[i].plot(self.results['true_states'][:, i], 
                        'k-', label='True', linewidth=2)
            axes[i].plot(self.results['hybrid_states'][:, i],
                        'b--', label='Hybrid EKF')
            axes[i].plot(self.results['standard_states'][:, i],
                        'r:', label='Standard EKF')
            axes[i].plot(self.results['ukf_states'][:, i],
                        'g-.', label='UKF')  # Add UKF plot
            
            # Add uncertainty bounds for both Hybrid EKF and UKF
            for filter_name, color in [('hybrid', 'b'), ('ukf', 'g')]:
                axes[i].fill_between(
                    range(len(self.results[f'{filter_name}_states'])),
                    self.results[f'{filter_name}_states'][:, i] - 2*self.results[f'{filter_name}_std'][:, i],
                    self.results[f'{filter_name}_states'][:, i] + 2*self.results[f'{filter_name}_std'][:, i],
                    alpha=0.1, color=color
                )
            
            axes[i].set_ylabel(f'State {label}')
            axes[i].legend()
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('State Trajectories Comparison')
        
        if save_path:
            plt.savefig(save_path / 'comparative_trajectories.png', dpi=300)
        plt.show()

def plot_comparative_phase_space(results: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """Plot 3D phase space comparison of both filters."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # True trajectory
    ax.plot3D(
        results['true_states'][:, 0],
        results['true_states'][:, 1],
        results['true_states'][:, 2],
        'k-', label='True', linewidth=1
    )
    
    # Hybrid EKF trajectory
    ax.plot3D(
        results['hybrid_states'][:, 0],
        results['hybrid_states'][:, 1],
        results['hybrid_states'][:, 2],
        'b--', label='Hybrid EKF', linewidth=1
    )
    
    # Standard EKF trajectory
    ax.plot3D(
        results['standard_states'][:, 0],
        results['standard_states'][:, 1],
        results['standard_states'][:, 2],
        'r:', label='Standard EKF', linewidth=1
    )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path / 'comparative_phase_space.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_evolution(metrics: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """
    Plot error evolution over time for both filters.
    
    Args:
        metrics: Dictionary containing error metrics
        save_path: Optional path to save the plots
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot RMSE evolution
    axes[0,0].plot(metrics['hybrid_rmse'], 'b-', label='Hybrid EKF')
    axes[0,0].plot(metrics['standard_rmse'], 'r--', label='Standard EKF')
    axes[0,0].set_title('RMSE Evolution')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot MAE evolution
    axes[0,1].plot(metrics['hybrid_mae'], 'b-', label='Hybrid EKF')
    axes[0,1].plot(metrics['standard_mae'], 'r--', label='Standard EKF')
    axes[0,1].set_title('MAE Evolution')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Plot dimensional RMSE comparison
    dim_labels = ['x', 'y', 'z']
    x = np.arange(len(dim_labels))
    width = 0.35
    
    axes[1,0].bar(x - width/2, metrics['hybrid_dim_rmse'], width, 
                 label='Hybrid EKF', color='blue', alpha=0.7)
    axes[1,0].bar(x + width/2, metrics['standard_dim_rmse'], width,
                 label='Standard EKF', color='red', alpha=0.7)
    axes[1,0].set_title('RMSE by Dimension')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(dim_labels)
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Plot error distribution
    axes[1,1].hist(metrics['hybrid_errors'].flatten(), bins=50, alpha=0.5,
                  label='Hybrid EKF', color='blue', density=True)
    axes[1,1].hist(metrics['standard_errors'].flatten(), bins=50, alpha=0.5,
                  label='Standard EKF', color='red', density=True)
    axes[1,1].set_title('Error Distribution')
    axes[1,1].set_xlabel('Error Magnitude')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'error_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparative_phase_portraits(results: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """Plot side-by-side phase portraits for all filters."""
    
    # Create figure with 3 subplots side by side
    fig = plt.figure(figsize=(20, 6))
    
    # Color map for time evolution
    n_points = len(results['true_states'])
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    # Subplot titles and data keys
    plots = [
        ('Hybrid EKF', 'hybrid_states'),
        ('Standard EKF', 'standard_states'),
        ('UKF', 'ukf_states')
    ]
    
    for idx, (title, key) in enumerate(plots, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        # Plot true trajectory with time-based coloring
        for i in range(n_points-1):
            ax.plot3D(results['true_states'][i:i+2, 0],
                     results['true_states'][i:i+2, 1],
                     results['true_states'][i:i+2, 2],
                     color=colors[i], alpha=0.5, linewidth=1)
        
        # Plot estimated trajectory
        ax.plot3D(results[key][:, 0],
                 results[key][:, 1],
                 results[key][:, 2],
                 'r--', label='Estimated', linewidth=1)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'{title} Phase Portrait')
        
        # Set consistent view angle for all subplots
        ax.view_init(elev=20, azim=45)
        
        # Add legend
        ax.legend(['True (time-colored)', 'Estimated'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'comparative_phase_portraits.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_variation(results: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """Plot the time variation of Lorenz system parameters."""
    
    # Generate time points
    t = np.arange(len(results['true_states'])) * 0.01
    
    # Create parameter generator
    param_gen = TimeVaryingParameters()
    
    # Get parameter values over time
    params = np.array([param_gen.get_parameters(ti) for ti in t])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    param_names = ['σ (sigma)', 'ρ (rho)', 'β (beta)']
    base_values = [10.0, 28.0, 8/3]
    
    for i, (name, base) in enumerate(zip(param_names, base_values)):
        axes[i].plot(t, params[:, i], 'b-', label='Time-varying')
        axes[i].axhline(y=base, color='r', linestyle='--', label='Base value')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(name)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'parameter_variation.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()

#################################
# Part 7: Main Execution        #
#################################

def main(mode: str = 'default'):
    """
    Main execution function with selectable mode.
    
    Args:
        mode: 'default' or 'comparative'
    """
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    output_dir = Path('hybrid_ekf_results_new_architecture')
    output_dir.mkdir(exist_ok=True)
    
    if mode == 'comparative':
        logging.info("Running comparative simulation mode...")
        # Create config for comparative simulation
        config = HybridEKFConfig(
            state_dim=3,
            measurement_dim=2,
            sequence_length=100,
            dt=0.01,
            learning_rate=0.001
        )
        
        # Comparative simulation with config
        results = simulate_comparative_system(
            time_steps=5000,
            dt=0.01,
            noise_type='mixture',
            config=config
        )
        
        # Comparative visualization
        visualizer = ComparisonVisualizer(results)
        visualizer.plot_comparative_trajectories(output_dir)
        
        # Compute and display comparative metrics
        metrics = compute_comparative_metrics(
            results['true_states'],
            results['hybrid_states'],
            results['standard_states'],
            results['ukf_states']
        )
        
        # Plot error evolution
        plot_error_evolution(metrics, output_dir)
        
        # Calculate RMSE values
        hybrid_rmse = np.sqrt(np.mean((results['true_states'] - results['hybrid_states'])**2))
        standard_rmse = np.sqrt(np.mean((results['true_states'] - results['standard_states'])**2))
        ukf_rmse = np.sqrt(np.mean((results['true_states'] - results['ukf_states'])**2))
        
        # Create dictionary for easier comparison
        rmse_dict = {
            'Hybrid EKF': hybrid_rmse,
            'Standard EKF': standard_rmse,
            'UKF': ukf_rmse
        }
        
        # Find best performer
        best_performer = min(rmse_dict.items(), key=lambda x: x[1])[0]
        
        # Log results
        logging.info("\nComparative Results:")
        logging.info(f"Hybrid EKF RMSE: {hybrid_rmse:.4f}")
        logging.info(f"Standard EKF RMSE: {standard_rmse:.4f}")
        logging.info(f"UKF RMSE: {ukf_rmse:.4f}")
        logging.info(f"Best performer: {best_performer}")
        
        # Add new visualizations
        plot_comparative_phase_portraits(results, output_dir)
        plot_parameter_variation(results, output_dir)
        
    else:
        logging.info("Running default simulation mode...")
        # Default simulation code...
        config = HybridEKFConfig(
            state_dim=3,
            measurement_dim=2,
            sequence_length=100,
            dt=0.01,
            learning_rate=0.001
        )
        
        results = simulate_system(
            config=config,
            time_steps=5000,
            save_path=output_dir / 'simulation_results.npz'
        )
        
        # Default visualization
        visualizer = ResultVisualizer(results)
        visualizer.plot_state_trajectories(output_dir)
        visualizer.plot_uncertainty_evolution(output_dir)
        visualizer.plot_learning_metrics(output_dir)
        
        # Display final metrics
        rmse = np.sqrt(np.mean((results['true_states'] - results['estimated_states'])**2))
        final_uncertainty = np.mean(results['uncertainties'][-100:], axis=0)
        
        logging.info(f"\nFinal Results:")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"Final Process Noise Confidence: {final_uncertainty[0]:.4f}")
        logging.info(f"Final Measurement Noise Confidence: {final_uncertainty[1]:.4f}")

def generate_non_gaussian_noise(size: int, noise_type: str = 'mixture') -> np.ndarray:
    """
    Generate non-Gaussian noise.
    
    Args:
        size: Size of noise vector
        noise_type: Type of non-Gaussian noise ('mixture', 'student_t', 'skewed')
        
    Returns:
        Non-Gaussian noise samples
    """
    if noise_type == 'mixture':
        # Gaussian mixture model
        noise = np.where(
            np.random.rand(size) > 0.7,
            np.random.normal(0, 2.0, size),
            np.random.normal(0, 0.5, size)
        )
    elif noise_type == 'student_t':
        # Student's t-distribution (heavy-tailed)
        noise = np.random.standard_t(df=3, size=size)
    elif noise_type == 'skewed':
        # Skewed distribution
        noise = np.random.normal(0, 10, size) + np.abs(np.random.normal(0, 3, size))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
        
    return noise

def compute_comparative_metrics(
    true_states: np.ndarray,
    hybrid_states: np.ndarray,
    standard_states: np.ndarray,
    ukf_states: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute comparative metrics for all three filters."""
    
    # Compute errors for all filters
    hybrid_errors = np.abs(true_states - hybrid_states)
    standard_errors = np.abs(true_states - standard_states)
    ukf_errors = np.abs(true_states - ukf_states)
    
    # Compute RMSE over time
    hybrid_rmse = np.sqrt(np.mean(hybrid_errors**2, axis=1))
    standard_rmse = np.sqrt(np.mean(standard_errors**2, axis=1))
    ukf_rmse = np.sqrt(np.mean(ukf_errors**2, axis=1))
    
    # Compute MAE over time
    hybrid_mae = np.mean(hybrid_errors, axis=1)
    standard_mae = np.mean(standard_errors, axis=1)
    ukf_mae = np.mean(ukf_errors, axis=1)
    
    # Compute error statistics per dimension
    hybrid_dim_rmse = np.sqrt(np.mean(hybrid_errors**2, axis=0))
    standard_dim_rmse = np.sqrt(np.mean(standard_errors**2, axis=0))
    ukf_dim_rmse = np.sqrt(np.mean(ukf_errors**2, axis=0))
    
    return {
        'hybrid_rmse': hybrid_rmse,
        'standard_rmse': standard_rmse,
        'ukf_rmse': ukf_rmse,
        'hybrid_mae': hybrid_mae,
        'standard_mae': standard_mae,
        'ukf_mae': ukf_mae,
        'hybrid_dim_rmse': hybrid_dim_rmse,
        'standard_dim_rmse': standard_dim_rmse,
        'ukf_dim_rmse': ukf_dim_rmse,
        'hybrid_errors': hybrid_errors,
        'standard_errors': standard_errors,
        'ukf_errors': ukf_errors
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid EKF Tutorial')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'comparative'],
        default='default',
        help='Simulation mode: default or comparative'
    )
    parser.add_argument(
        '--noise-type',
        type=str,
        choices=['mixture', 'student_t', 'skewed'],
        default='mixture',
        help='Type of non-Gaussian noise to use'
    )
    parser.add_argument(
        '--time-steps',
        type=int,
        default=5000,
        help='Number of simulation time steps'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        main(mode=args.mode)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise