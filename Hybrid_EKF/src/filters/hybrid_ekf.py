"""Hybrid Extended Kalman Filter implementation with neural network enhancements."""

import numpy as np
import torch
from typing import Dict, Optional, Deque
from collections import deque
import logging

from .base import BaseFilter
from ..models.lorenz import LorenzSystem
from ..models.neural_networks import PhysicsInformedNetwork, EnhancedUncertaintyNetwork
from ..config import HybridEKFConfig

class HybridEKF(BaseFilter):
    """Enhanced Hybrid EKF with uncertainty quantification and physics-informed learning."""
    
    def __init__(self, 
                 system: LorenzSystem,
                 config: HybridEKFConfig):
        super().__init__(config.state_dim)
        self.system = system
        self.config = config
        
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
        
        # Initialize buffers
        self.state_buffer: Deque = deque(maxlen=config.sequence_length)
        self.innovation_buffer: Deque = deque(maxlen=config.sequence_length)
        
        # Initialize metrics
        self.metrics = {
            'prediction_loss': [],
            'uncertainty_loss': [],
            'q_confidence': [],
            'r_confidence': []
        }
    
    def _get_uncertainty_estimates(self) -> Dict[str, torch.Tensor]:
        """Get uncertainty estimates from the network."""
        if len(self.state_buffer) < self.config.sequence_length:
            return {
                'Q_mean': torch.eye(self.state_dim) * self.config.process_noise_init,
                'Q_std': torch.zeros(self.state_dim, self.state_dim),
                'R_mean': torch.eye(self.config.measurement_dim) * self.config.measurement_noise_init,
                'R_std': torch.zeros(self.config.measurement_dim, self.config.measurement_dim),
                'q_confidence': torch.tensor([0.5]),
                'r_confidence': torch.tensor([0.5])
            }
        
        state_sequence = np.array(list(self.state_buffer))
        innovation_sequence = np.array(list(self.innovation_buffer))
        
        state_sequence = torch.from_numpy(state_sequence).float().unsqueeze(1)
        innovation_sequence = torch.from_numpy(innovation_sequence).float().unsqueeze(1)
        
        with torch.no_grad():
            return self.uncertainty_net(state_sequence, innovation_sequence, self.config.dt)
    
    def predict(self, dt: float) -> None:
        """Prediction step with neural network enhancement."""
        try:
            # Physics-based prediction
            physics_pred = self.system.dynamics(self.x_hat, dt)
            
            # Neural correction
            x_tensor = torch.FloatTensor(self.x_hat)
            physics_tensor = torch.FloatTensor(physics_pred)
            
            with torch.no_grad():
                corrected_state, _ = self.physics_net(x_tensor, physics_tensor)
                self.x_hat = corrected_state.numpy()
            
            # Get uncertainty estimates
            uncertainty = self._get_uncertainty_estimates()
            Q = uncertainty['Q_mean'].numpy() + uncertainty['Q_std'].numpy()
            Q = Q.squeeze()
            
            # Update state covariance
            F = self.system.jacobian(self.x_hat)
            self.P = F @ self.P @ F.T + Q
            
            self._validate_matrices()
            self._update_buffers(self.x_hat)
            
        except Exception as e:
            logging.error(f"Error in HybridEKF prediction: {str(e)}")
            raise
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with uncertainty quantification."""
        try:
            # Measurement model
            H = np.zeros((self.config.measurement_dim, self.state_dim))
            H[0,0] = 1.0  # Observe x
            H[1,2] = 1.0  # Observe z
            
            # Extract measured components
            measured_components = np.array([measurement[0], measurement[2]])
            
            # Get uncertainty estimates
            uncertainty = self._get_uncertainty_estimates()
            R = uncertainty['R_mean'].numpy() + uncertainty['R_std'].numpy()
            R = R.squeeze()
            
            # Kalman gain computation
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state
            innovation = measured_components - H @ self.x_hat
            self.x_hat = self.x_hat + K @ innovation
            
            # Update covariance (Joseph form)
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
            
            self._validate_matrices()
            self._update_buffers(self.x_hat, innovation)
            self._online_learning(measurement, innovation, uncertainty)
            
        except Exception as e:
            logging.error(f"Error in HybridEKF update: {str(e)}")
            raise
    
    def _update_buffers(self, state: np.ndarray, innovation: Optional[np.ndarray] = None) -> None:
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
            
            uncertainty_loss = self._compute_uncertainty_loss(
                innovation,
                current_uncertainty['R_mean'],
                current_uncertainty['R_std']
            )
            
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