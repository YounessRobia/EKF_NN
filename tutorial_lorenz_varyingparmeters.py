"""
Hybrid EKF Tutorial: State + Parameter Inference for the Lorenz system.
Author: Your Name
Date: 2025-01-11
"""

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from enum import Enum
from typing import Dict, Optional, Tuple, Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
#                         LOGGING & DEVICE CONFIG
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


@dataclass
class HybridEKFConfig:
    """
    Configuration for the enhanced Hybrid EKF system.
    """
    state_dim: int = 3
    measurement_dim: int = 2  # We observe only x and z
    hidden_dim: int = 64
    sequence_length: int = 20
    learning_rate: float = 0.001
    dt: float = 0.01
    min_covariance: float = 1e-10
    process_noise_init: float = 0.1
    measurement_noise_init: float = 0.1


class CovarianceUpdateMethod(Enum):
    """
    Methods for covariance update in a Kalman filter.
    """
    STANDARD = "standard"
    JOSEPH = "joseph"
    SQRT = "sqrt"


class StateValidator:
    """
    State validation and bounding utility.
    Ensures numeric stability and clips large values.
    """

    def __init__(self,
                 min_bound: float = -1e3,
                 max_bound: float = 1e3,
                 min_covariance: float = 1e-10,
                 max_covariance: float = 1e3) -> None:
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.min_covariance = min_covariance
        self.max_covariance = max_covariance

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            logging.error(f"Invalid state detected: {state}")
            return np.zeros_like(state)
        return np.clip(state, self.min_bound, self.max_bound)

    def validate_covariance(self, P: np.ndarray) -> np.ndarray:
        """Enhanced covariance validation with error handling."""
        try:
            P = 0.5 * (P + P.T)  # force symmetry
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(P)
            except np.linalg.LinAlgError:
                # Fallback to more stable but slower SVD
                U, s, Vh = np.linalg.svd(P)
                eigenvalues, eigenvectors = s, U
                
            eigenvalues = np.clip(eigenvalues, self.min_covariance, self.max_covariance)
            P_clamped = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Additional numerical stability check
            if not np.all(np.isfinite(P_clamped)):
                logging.warning("Non-finite values in covariance. Resetting.")
                return self.min_covariance * np.eye(P.shape[0])
                
            return P_clamped
            
        except Exception as e:
            logging.error(f"Error in covariance validation: {e}")
            return self.min_covariance * np.eye(P.shape[0])


class TimeVaryingParameters:
    """
    Time-varying parameter generator for the Lorenz system.
    Demonstrates mild fluctuations in sigma, rho, beta over time.
    """
    def __init__(self,
                 sigma_mean: float = 10.0,
                 rho_mean: float = 28.0,
                 beta_mean: float = 8/3,
                 variation_amplitude: float = 0.2) -> None:
        self.sigma_mean = sigma_mean
        self.rho_mean = rho_mean
        self.beta_mean = beta_mean
        self.amplitude = variation_amplitude

    def get_parameters(self, t: float) -> Tuple[float, float, float]:
        """
        Generate time-varying parameters (sigma, rho, beta).
        """
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
    """
    Lorenz system dynamics with optional time-varying parameters.
    Allows state propagation via RK4 integration.
    """
    def __init__(self,
                 sigma: float = 10.0,
                 rho: float = 28.0,
                 beta: float = 8/3,
                 time_varying: bool = False,
                 variation_amplitude: float = 0.2) -> None:
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

    def get_current_parameters(self, t: float = 0.0) -> Tuple[float, float, float]:
        if self.time_varying:
            return self.param_generator.get_parameters(t)
        return self.base_sigma, self.base_rho, self.base_beta

    def derivatives(self, 
                    state: np.ndarray, 
                    t: float = 0.0,
                    sigma: Optional[float] = None,
                    rho:   Optional[float] = None,
                    beta:  Optional[float] = None) -> np.ndarray:
        """
        If sigma, rho, beta are provided externally, use them. Otherwise, use get_current_parameters().
        """
        x, y, z = state
        if sigma is None or rho is None or beta is None:
            sigma, rho, beta = self.get_current_parameters(t)

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    def dynamics(self, 
                 state: np.ndarray, 
                 dt: float = 0.01, 
                 t: float = 0.0,
                 sigma: Optional[float] = None,
                 rho:   Optional[float] = None,
                 beta:  Optional[float] = None) -> np.ndarray:
        """
        RK4 integration with external parameters if given.
        """
        def rk4_step(s: np.ndarray, t_curr: float) -> np.ndarray:
            k1 = self.derivatives(s, t_curr, sigma, rho, beta)
            k2 = self.derivatives(s + 0.5*dt*k1, t_curr + 0.5*dt, sigma, rho, beta)
            k3 = self.derivatives(s + 0.5*dt*k2, t_curr + 0.5*dt, sigma, rho, beta)
            k4 = self.derivatives(s + dt*k3,     t_curr + dt,     sigma, rho, beta)
            return (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        next_state = state + rk4_step(state, t)
        if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 1e6):
            logging.warning(f"Numerical instability detected at state: {state}")
            return state
        return next_state

    def jacobian(self, 
                 state: np.ndarray,
                 sigma: Optional[float] = None,
                 rho:   Optional[float] = None,
                 beta:  Optional[float] = None) -> np.ndarray:
        """
        Optionally compute the Jacobian using external sigma, rho, beta.
        """
        x, y, z = state
        if sigma is None:
            sigma = self.base_sigma
        if rho is None:
            rho = self.base_rho
        if beta is None:
            beta = self.base_beta
        return np.array([
            [-sigma,    sigma,      0.0],
            [ rho - z,  -1.0,       -x  ],
            [ y,        x,    -beta ]
        ])


class PhysicsSequenceNetwork(nn.Module):
    """
    A sequential (LSTM-based) physics-informed neural network.
    It processes a window of recent states to generate a correction
    on top of a physics-based prior.
    """
    def __init__(self, 
                 state_dim: int = 3, 
                 hidden_dim: int = 64, 
                 sequence_length: int = 20) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # LSTM to encode sequential dynamics
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Correction head
        self.correction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, 
                past_states: torch.Tensor, 
                physics_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            past_states: shape (B, T, state_dim)
            physics_prior: shape (B, state_dim)

        Returns:
            corrected_state: shape (B, state_dim)
            physics_loss: scalar (L2 on corrections or other)
        """
        lstm_out, _ = self.lstm(past_states)  # (B, T, hidden_dim)
        final_hidden = lstm_out[:, -1, :]     # shape (B, hidden_dim)

        correction = self.correction_head(final_hidden)  # (B, state_dim)
        corrected_state = physics_prior + correction

        # example: L2 penalty on the correction
        physics_loss = torch.mean(correction**2)

        return corrected_state, physics_loss
class EnhancedUncertaintyNetwork(nn.Module):
    """
    Advanced uncertainty network with attention and multi-head estimation.
    It uses an LSTM (bidirectional) to output Q/R estimates from the last
    `sequence_length` states/innovations.
    """
    def __init__(self, 
                 state_dim: int = 3,
                 measurement_dim: int = 2,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 sequence_length: int = 5) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.sequence_length = sequence_length

        # State encoder
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
            dropout=0.1,
            batch_first=True
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
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

        # Confidence network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, 
                state_sequence: torch.Tensor,
                innovation_sequence: torch.Tensor,
                dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Forward pass to estimate Q/R mean and std, plus confidence.
        """
        batch_size = state_sequence.size(0)
        encoded = self.state_encoder(state_sequence)  # (B, T, hidden_dim)

        # Self-attention
        attended, _ = self.attention(encoded, encoded, encoded)  # (B, T, hidden_dim)

        # Bi-LSTM
        lstm_out, _ = self.lstm(attended)       # (B, T, hidden_dim)
        hidden = lstm_out[:, -1, :]             # last time-step (B, hidden_dim)

        # Q
        q_params = self.q_estimator(hidden)  # (B, state_dim^2)
        Q_mean = q_params.view(batch_size, self.state_dim, self.state_dim)
        Q_mean = 0.5 * (Q_mean + Q_mean.transpose(-2, -1))  # symmetrize
        Q_mean = F.softplus(Q_mean) * dt

        # R
        r_params = self.r_estimator(hidden)  # (B, measurement_dim^2)
        R_mean = r_params.view(batch_size, self.measurement_dim, self.measurement_dim)
        R_mean = 0.5 * (R_mean + R_mean.transpose(-2, -1))
        R_mean = F.softplus(R_mean)

        # Confidence
        confidence = self.confidence_estimator(hidden)  # (B, 2)
        q_confidence = confidence[:, 0]
        r_confidence = confidence[:, 1]

        Q_std = (1.0 - q_confidence).unsqueeze(-1).unsqueeze(-1) * Q_mean
        R_std = (1.0 - r_confidence).unsqueeze(-1).unsqueeze(-1) * R_mean

        return {
            'Q_mean': Q_mean,
            'Q_std': Q_std,
            'R_mean': R_mean,
            'R_std': R_std,
            'q_confidence': q_confidence,
            'r_confidence': r_confidence
        }

    def get_regularization_loss(self) -> torch.Tensor:
        all_params = []
        for param in self.parameters():
            if param.requires_grad:
                all_params.append(param.view(-1))
        l2_loss = torch.norm(torch.cat(all_params))
        return 0.01 * l2_loss


class ParameterNetwork(nn.Module):
    """
    Network to estimate the Lorenz system parameters (sigma, rho, beta).
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # outputs [sigma, rho, beta]
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, hidden_rep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns [sigma, rho, beta].
        hidden_rep: shape (B, hidden_dim)
        """
        return self.net(hidden_rep)
class IntegratedHybridEKF:
    """
    Hybrid EKF that merges:
      - Physics-based prediction (with on-the-fly parameter inference)
      - Physics-based prior corrections via PhysicsSequenceNetwork
      - Uncertainty estimation from EnhancedUncertaintyNetwork
      - Extended Kalman Filter update
      - Online learning with backprop
    """

    def __init__(self, system: LorenzSystem, config: HybridEKFConfig) -> None:
        self.system = system
        self.config = config

        # EKF state
        self.x_hat = np.zeros(config.state_dim)
        self.P = np.eye(config.state_dim)

        # Neural networks
        self.physics_net = PhysicsSequenceNetwork(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            sequence_length=config.sequence_length
        ).to(DEVICE)

        self.uncertainty_net = EnhancedUncertaintyNetwork(
            state_dim=config.state_dim,
            measurement_dim=config.measurement_dim,
            hidden_dim=config.hidden_dim,
            sequence_length=config.sequence_length
        ).to(DEVICE)

        # New: Parameter Network
        self.parameter_net = ParameterNetwork(hidden_dim=config.hidden_dim).to(DEVICE)

        self.state_validator = StateValidator(
            min_bound=-1e4,
            max_bound=1e4,
            min_covariance=config.min_covariance,
            max_covariance=1e3
        )

        # Optimizers
        # Combine physics_net and parameter_net parameters
        self.physics_optimizer = torch.optim.AdamW(
            list(self.physics_net.parameters()) + list(self.parameter_net.parameters()),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.uncertainty_optimizer = torch.optim.AdamW(
            self.uncertainty_net.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )

        # Buffers to store recent states and innovations
        self.state_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)
        self.innovation_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)

        # Metrics
        self.metrics = {
            'prediction_loss': [],
            'uncertainty_loss': [],
            'q_confidence': [],
            'r_confidence': [],
            'sigma_est': [],
            'rho_est': [],
            'beta_est': []
        }

    def predict(self, dt: float, t: float = 0.0) -> None:
        """
        EKF prediction step with on-the-fly parameter inference.
        """
        try:
            # Validate inputs
            if not (0 < dt < 1.0):
                raise ValueError(f"Invalid dt: {dt}. Must be between 0 and 1.")
            
            # 1) Physics-based prior (using the system's current or base parameters)
            physics_pred = self.system.dynamics(self.x_hat, dt, t)

            # 2) Neural correction + parameter estimation
            states_array = np.array(list(self.state_buffer))
            if len(states_array) < self.config.sequence_length:
                corrected_state = physics_pred
                # If not enough data, fall back to system's current parameters
                sigma_est, rho_est, beta_est = self.system.get_current_parameters(t)
            else:
                states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)
                physics_prior = torch.FloatTensor(physics_pred).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Forward pass through LSTM
                    lstm_out, _ = self.physics_net.lstm(states_tensor)
                    final_hidden = lstm_out[:, -1, :]  # (1, hidden_dim)

                    # Param estimation
                    param_out = self.parameter_net(final_hidden)  # (1, 3)
                    sigma_est, rho_est, beta_est = param_out.squeeze(0).cpu().numpy()

                    # Correction head
                    correction = self.physics_net.correction_head(final_hidden)
                    corrected_tensor = physics_prior + correction
                    corrected_state = corrected_tensor.squeeze(0).cpu().numpy()

            self.x_hat = corrected_state

            # 3) Uncertainty estimation
            uncertainty = self._get_uncertainty_estimates()
            Q_raw = uncertainty['Q_mean'] + uncertainty['Q_std']
            Q_np = Q_raw.squeeze(0).cpu().numpy()
            Q_np = 0.5 * (Q_np + Q_np.T)  # symmetrize

            # 4) Covariance update via Jacobian (with estimated parameters)
            F = self.system.jacobian(self.x_hat, sigma_est, rho_est, beta_est)
            self.P = F @ self.P @ F.T + Q_np
            self.P = 0.5 * (self.P + self.P.T)

            # 5) Buffer and logs
            self._update_buffers(self.x_hat)
            self.metrics['sigma_est'].append(sigma_est)
            self.metrics['rho_est'].append(rho_est)
            self.metrics['beta_est'].append(beta_est)

        except torch.cuda.OutOfMemoryError:
            logging.error("GPU OOM in predict step. Falling back to CPU")
            self.physics_net.to('cpu')
            self.parameter_net.to('cpu')
            self.uncertainty_net.to('cpu')
            
        except np.linalg.LinAlgError as e:
            logging.error(f"Linear algebra error in predict: {e}")
            self.P = self.state_validator.validate_covariance(self.P)
            
        except Exception as e:
            logging.error(f"Error in predict step: {e}")
            # Fallback to simple physics prediction
            self.x_hat = self.system.dynamics(self.x_hat, dt, t)

    def update(self, measurement: np.ndarray) -> None:
        """
        EKF update step.
        """
        try:
            # 1) Validate
            measurement = self.state_validator.validate_state(measurement).squeeze()
            self.x_hat = self.state_validator.validate_state(self.x_hat).squeeze()
            self.P = self.state_validator.validate_covariance(self.P).squeeze()

            # 2) Partial measurement matrix H
            H = np.zeros((self.config.measurement_dim, self.config.state_dim))
            H[0, 0] = 1.0  # observe x
            H[1, 2] = 1.0  # observe z

            # 3) Measurement noise R
            uncertainty = self._get_uncertainty_estimates()
            R_raw = uncertainty['R_mean'] + uncertainty['R_std']
            R_np = R_raw.squeeze(0).cpu().numpy()
            R_np = 0.5 * (R_np + R_np.T)
            R_np = np.maximum(R_np, self.config.min_covariance * np.eye(self.config.measurement_dim))

            PHt = self.P @ H.T
            S = H @ PHt + R_np
            S = 0.5 * (S + S.T)

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            K = PHt @ S_inv
            innovation = measurement - (H @ self.x_hat)
            self.x_hat = self.x_hat + (K @ innovation)

            I = np.eye(self.config.state_dim)
            KH = K @ H
            self.P = (I - KH) @ self.P @ (I - KH).T + K @ R_np @ K.T

            self.x_hat = self.state_validator.validate_state(self.x_hat)
            self.P = self.state_validator.validate_covariance(self.P)

            # 4) Online learning
            if not np.any(np.isnan(self.x_hat)):
                self._update_buffers(self.x_hat, innovation)
                self._online_learning(measurement, innovation, uncertainty)

        except Exception as e:
            logging.error(f"Error in update step: {e}")
            raise

    def _get_uncertainty_estimates(self) -> Dict[str, torch.Tensor]:
        """
        Returns Q/R from the EnhancedUncertaintyNetwork.
        If insufficient data, return default values.
        """
        if len(self.state_buffer) < self.config.sequence_length:
            eye_Q = torch.eye(self.config.state_dim, device=DEVICE) * self.config.process_noise_init
            eye_R = torch.eye(self.config.measurement_dim, device=DEVICE) * self.config.measurement_noise_init
            return {
                'Q_mean': eye_Q.unsqueeze(0),
                'Q_std': torch.zeros_like(eye_Q).unsqueeze(0),
                'R_mean': eye_R.unsqueeze(0),
                'R_std': torch.zeros_like(eye_R).unsqueeze(0),
                'q_confidence': torch.tensor([0.5], device=DEVICE),
                'r_confidence': torch.tensor([0.5], device=DEVICE)
            }

        state_seq = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(DEVICE)
        innov_seq = torch.FloatTensor(list(self.innovation_buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.uncertainty_net(state_seq, innov_seq, dt=self.config.dt)
        return outputs

    def _update_buffers(self, state: np.ndarray, innovation: Optional[np.ndarray] = None) -> None:
        self.state_buffer.append(state)
        if innovation is not None:
            self.innovation_buffer.append(innovation)

    def _online_learning(self,
                         measurement: np.ndarray,
                         innovation: np.ndarray,
                         uncertainty: Dict[str, torch.Tensor]) -> None:
        """
        Train the physics_net + parameter_net, and the uncertainty_net using the most recent data.
        """
        try:
            # Validate inputs
            if not isinstance(measurement, np.ndarray) or not isinstance(innovation, np.ndarray):
                raise TypeError("Measurement and innovation must be numpy arrays")
            
            if np.any(np.isnan(measurement)) or np.any(np.isnan(innovation)):
                raise ValueError("NaN values in measurement or innovation")
            
            # ------ Train Physics + Parameter Network ------
            self.physics_optimizer.zero_grad()
            if len(self.state_buffer) == self.config.sequence_length:
                states_array = np.array(list(self.state_buffer))
                states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)

                last_state = states_array[-1]
                # Quick physics prior with base or time-varying parameters
                physics_pred_np = self.system.dynamics(last_state, self.config.dt)
                physics_pred_tensor = torch.FloatTensor(physics_pred_np).unsqueeze(0).to(DEVICE)

                measurement_tensor = torch.FloatTensor(measurement).unsqueeze(0).to(DEVICE)

                # Forward pass through LSTM
                lstm_out, _ = self.physics_net.lstm(states_tensor)
                final_hidden = lstm_out[:, -1, :]

                # Parameter inference
                param_out = self.parameter_net(final_hidden)  # shape (1, 3)
                # You can impose a regularization if desired:
                param_reg = torch.mean(param_out**2) * 1e-4

                # Correction
                correction = self.physics_net.correction_head(final_hidden)
                corrected = physics_pred_tensor + correction  # shape (1, 3)

                # Compare corrected x,z to measurement
                partial_corrected = torch.stack([corrected[:, 0], corrected[:, 2]], dim=-1)
                partial_loss = F.mse_loss(partial_corrected, measurement_tensor)

                # L2 penalty on correction
                physics_loss = torch.mean(correction**2)

                physics_total_loss = partial_loss + 0.1 * physics_loss + param_reg
                physics_total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.physics_net.parameters()) + list(self.parameter_net.parameters()), 
                    1.0
                )
                self.physics_optimizer.step()
                self.metrics['prediction_loss'].append(partial_loss.item())

            # ------ Train Uncertainty Network ------
            if len(self.state_buffer) == self.config.sequence_length:
                self.uncertainty_optimizer.zero_grad()
                state_seq = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(DEVICE)
                innov_seq = torch.FloatTensor(list(self.innovation_buffer)).unsqueeze(0).to(DEVICE)

                current_uncert = self.uncertainty_net(state_seq, innov_seq, dt=self.config.dt)
                nll_loss = self._compute_uncertainty_loss(innovation, 
                                                          current_uncert['R_mean'], 
                                                          current_uncert['R_std'])
                reg_loss = self.uncertainty_net.get_regularization_loss()
                uncertainty_loss = nll_loss + reg_loss
                uncertainty_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.uncertainty_net.parameters(), 1.0)
                self.uncertainty_optimizer.step()

                # Record metrics
                self.metrics['uncertainty_loss'].append(uncertainty_loss.item())
                self.metrics['q_confidence'].append(uncertainty['q_confidence'].item())
                self.metrics['r_confidence'].append(uncertainty['r_confidence'].item())

        except torch.cuda.OutOfMemoryError:
            logging.error("GPU OOM in learning step. Skipping update.")
            return
        
        except Exception as e:
            logging.error(f"Error in online learning: {e}")
            # Skip this learning step
            return

    def _compute_uncertainty_loss(self,
                                  innovation: np.ndarray,
                                  R_mean: torch.Tensor,
                                  R_std: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood for measurement innovation ~ N(0, R).
        """
        innovation_tensor = torch.FloatTensor(innovation).to(DEVICE)
        R_total = R_mean + R_std
        R_total = torch.clamp(R_total, min=self.config.min_covariance)
        R_2x2 = R_total.squeeze(0)
        logdetR = torch.logdet(R_2x2)

        inn_2x1 = innovation_tensor.unsqueeze(-1)
        mahalanobis = torch.matmul(
            torch.matmul(inn_2x1.transpose(0, 1), torch.inverse(R_2x2)),
            inn_2x1
        )
        # Gaussian log-likelihood
        ll = -0.5 * (2 * np.log(2.0 * np.pi) + logdetR + mahalanobis)
        return -ll.squeeze()

    def _check_system_health(self) -> bool:
        """
        Validates overall system state health.
        Returns True if healthy, False otherwise.
        """
        try:
            # Check state vector
            if np.any(np.isnan(self.x_hat)) or np.any(np.abs(self.x_hat) > 1e6):
                logging.error("State vector corrupted")
                return False
            
            # Check covariance
            if not np.all(np.linalg.eigvals(self.P) > 0):
                logging.error("Covariance matrix not positive definite")
                return False
            
            # Check neural network states
            if torch.isnan(next(self.physics_net.parameters())[0]):
                logging.error("Physics network contains NaN weights")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False
def simulate_parameter_estimation(
    config: HybridEKFConfig,
    time_steps: int = 1000,
    save_path: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Demonstration of on-the-fly parameter inference with the Hybrid EKF.
    """
    # Create Lorenz system with time-varying parameters
    system = LorenzSystem(time_varying=True, variation_amplitude=0.5)
    ekf = IntegratedHybridEKF(system, config)

    # Allocate storage
    true_states = np.zeros((time_steps, config.state_dim))
    estimated_states = np.zeros_like(true_states)
    sigma_true = np.zeros(time_steps)
    rho_true   = np.zeros(time_steps)
    beta_true  = np.zeros(time_steps)

    dt = config.dt
    x_true = np.array([1.0, 1.0, 1.0])

    for t in range(time_steps):
        # Retrieve true (time-varying) parameters
        sigma_t, rho_t, beta_t = system.get_current_parameters(t * dt)
        sigma_true[t] = sigma_t
        rho_true[t]   = rho_t
        beta_true[t]  = beta_t

        # Evolve the true state
        x_true = system.dynamics(x_true, dt, t * dt)
        true_states[t] = x_true

        # Noisy measurement of (x, z)
        measured_xz = x_true[[0, 2]] + np.random.normal(0, config.measurement_noise_init, size=2)

        # Hybrid EKF steps
        ekf.predict(dt, t * dt)
        ekf.update(measured_xz)

        estimated_states[t] = ekf.x_hat

    results = {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'sigma_true': sigma_true,
        'rho_true': rho_true,
        'beta_true': beta_true,
        'sigma_est': np.array(ekf.metrics['sigma_est']),
        'rho_est':   np.array(ekf.metrics['rho_est']),
        'beta_est':  np.array(ekf.metrics['beta_est']),
        'metrics': ekf.metrics
    }

    if save_path:
        np.savez(save_path, **results)

    return results
class ResultVisualizer:
    def __init__(self, results: Dict[str, np.ndarray]):
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]

    def plot_state_trajectories(self, save_path: Optional[Path] = None):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        true_states = self.results['true_states']
        estimated_states = self.results['estimated_states']

        for i, label in enumerate(['x', 'y', 'z']):
            axes[i].plot(true_states[:, i], 'k-', label='True', linewidth=2)
            axes[i].plot(estimated_states[:, i], 'r--', label='Estimated')
            axes[i].set_ylabel(label)
            axes[i].legend()

        axes[-1].set_xlabel('Time step')
        plt.suptitle('State Trajectories')
        if save_path:
            plt.savefig(save_path / 'state_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_parameter_estimates(results: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    sns.set_style("whitegrid")
    t = np.arange(len(results['sigma_true']))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Sigma
    axes[0].plot(t, results['sigma_true'], 'k-', label='True Sigma')
    axes[0].plot(t, results['sigma_est'],  'r--', label='Estimated Sigma')
    axes[0].set_ylabel('sigma')
    axes[0].legend()

    # Rho
    axes[1].plot(t, results['rho_true'], 'k-', label='True Rho')
    axes[1].plot(t, results['rho_est'],  'r--', label='Estimated Rho')
    axes[1].set_ylabel('rho')
    axes[1].legend()

    # Beta
    axes[2].plot(t, results['beta_true'], 'k-', label='True Beta')
    axes[2].plot(t, results['beta_est'],  'r--', label='Estimated Beta')
    axes[2].set_ylabel('beta')
    axes[2].set_xlabel('Time step')
    axes[2].legend()

    plt.suptitle('Parameter Estimation')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'parameter_estimates.png', dpi=300, bbox_inches='tight')
    plt.show()
def main_parameter_estimation():
    """
    Main execution function for parameter inference demonstration.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = Path('hybrid_ekf_param_est')
    output_dir.mkdir(exist_ok=True)

    config = HybridEKFConfig(
        state_dim=3,
        measurement_dim=2,
        hidden_dim=32,
        sequence_length=20,
        dt=0.01,
        learning_rate=0.001,
        process_noise_init=0.1,
        measurement_noise_init=0.1
    )

    logging.info("Running simulation with on-the-fly parameter estimation...")

    results = simulate_parameter_estimation(
        config=config,
        time_steps=2000,
        save_path=output_dir / 'param_est_results.npz'
    )

    # Plot states
    viz = ResultVisualizer(results)
    viz.plot_state_trajectories(output_dir)

    # Plot parameters
    plot_parameter_estimates(results, output_dir)

    # Compute final RMSE over all states
    rmse = np.sqrt(np.mean((results['true_states'] - results['estimated_states'])**2))
    logging.info(f"Final RMSE over all states: {rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid EKF Tutorial with Parameter Inference')
    parser.add_argument(
        '--mode',
        type=str,
        default='param_estimation',
        help='Execution mode: param_estimation or any other custom mode.'
    )
    args = parser.parse_args()

    if args.mode == 'param_estimation':
        main_parameter_estimation()
    else:
        logging.info(f"Unknown mode: {args.mode}")
