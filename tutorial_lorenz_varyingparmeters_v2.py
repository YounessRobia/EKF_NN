#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Hybrid Extended Kalman Filter Tutorial with Sequential Physics Network and Best Practices

This version includes a 'comparative' mode to compare:
1. IntegratedHybridEKF
2. StandardEKF
3. UnscentedKF

on the Lorenz system.

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


# -----------------------------------------------------------------------------
#                      PART 1: System Configuration
# -----------------------------------------------------------------------------
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
        """
        Validate and clip state values.
        Resets to zero if state has NaN or inf.
        """
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            logging.error(f"Invalid state detected: {state}")
            return np.zeros_like(state)  # Reset to safe values

        return np.clip(state, self.min_bound, self.max_bound)

    def validate_covariance(self, P: np.ndarray) -> np.ndarray:
        """
        Ensure the covariance matrix is symmetric and
        has eigenvalues within [min_covariance, max_covariance].
        """
        # Force symmetry
        P = 0.5 * (P + P.T)
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        eigenvalues = np.clip(eigenvalues, self.min_covariance, self.max_covariance)
        P_clamped = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return P_clamped


# -----------------------------------------------------------------------------
#                      PART 2: Lorenz System Model
# -----------------------------------------------------------------------------
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
        # Slow variation in sigma
        sigma = self.sigma_mean * (1 + self.amplitude * np.sin(0.1 * t))
        # Faster variation in rho with two sine frequencies
        rho = self.rho_mean * (1 + self.amplitude * (
            0.7 * np.sin(0.3 * t) +
            0.3 * np.sin(0.7 * t)
        ))
        # Quasi-random variation in beta
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

    def get_current_parameters(self, t: float) -> Tuple[float, float, float]:
        """
        Return current (sigma, rho, beta).
        If time_varying=False, return base values.
        """
        if self.time_varying:
            return self.param_generator.get_parameters(t)
        return self.base_sigma, self.base_rho, self.base_beta

    def derivatives(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute the Lorenz derivatives given the current state (x, y, z).
        """
        x, y, z = state
        sigma, rho, beta = self.get_current_parameters(t)

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return np.array([dx, dy, dz])

    def dynamics(self, state: np.ndarray, dt: float = 0.01, t: float = 0.0) -> np.ndarray:
        """
        Numerical integration via RK4 step.
        """
        def rk4_step(s: np.ndarray, t_curr: float) -> np.ndarray:
            k1 = self.derivatives(s, t_curr)
            k2 = self.derivatives(s + 0.5 * dt * k1, t_curr + 0.5 * dt)
            k3 = self.derivatives(s + 0.5 * dt * k2, t_curr + 0.5 * dt)
            k4 = self.derivatives(s + dt * k3, t_curr + dt)
            return (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        next_state = state + rk4_step(state, t)

        # Basic numerical stability check
        if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 1e6):
            logging.warning(f"Numerical instability detected at state: {state}")
            return state

        return next_state

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Jacobian used by the Extended Kalman Filter (constant base parameters).
        """
        x, y, z = state
        return np.array([
            [-self.base_sigma,    self.base_sigma,       0.0],
            [ self.base_rho - z, -1.0,               -x  ],
            [        y,          x,            -self.base_beta]
        ])


# -----------------------------------------------------------------------------
#       PART 3: Neural Networks (Sequential Physics + Uncertainty)
# -----------------------------------------------------------------------------
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
        """
        Orthogonal initialization for linear and LSTM layers.
        """
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
            past_states: shape (batch_size, sequence_length, state_dim)
            physics_prior: shape (batch_size, state_dim)

        Returns:
            corrected_state: shape (batch_size, state_dim)
            physics_loss: scalar regularization term (e.g., L2 on corrections)
        """
        # LSTM expects shape (B, T, D). We assume batch_first=True
        lstm_out, _ = self.lstm(past_states)  # (B, T, hidden_dim)

        # Use the final time-step's hidden state
        final_hidden = lstm_out[:, -1, :]  # shape (B, hidden_dim)

        # Generate correction
        correction = self.correction_head(final_hidden)  # (B, state_dim)

        # Combine with physics prior
        corrected_state = physics_prior + correction

        # Example: simple L2 penalty on corrections
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
        """
        Orthogonal initialization for linear layers.
        """
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
        # Encode states
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
        """
        Optional L2 regularization on all parameters.
        """
        all_params = []
        for param in self.parameters():
            if param.requires_grad:
                all_params.append(param.view(-1))
        l2_loss = torch.norm(torch.cat(all_params))
        return 0.01 * l2_loss


# -----------------------------------------------------------------------------
#                PART 4: Integrated Hybrid EKF + Reference Filters
# -----------------------------------------------------------------------------
class IntegratedHybridEKF:
    """
    Hybrid EKF that merges:
      - Physics-based prediction (LorenzSystem + sequential PhysicsNetwork)
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

        self.state_validator = StateValidator(
            min_bound=-1e4,
            max_bound=1e4,
            min_covariance=config.min_covariance,
            max_covariance=1e3
        )

        # Optimizers
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

        # Buffers to store recent states and innovations
        self.state_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)
        self.innovation_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)

        # Metric logging
        self.metrics = {
            'prediction_loss': [],
            'uncertainty_loss': [],
            'q_confidence': [],
            'r_confidence': []
        }

    def predict(self, dt: float) -> None:
        """
        EKF prediction step:
          1) Use system.dynamics() for physics-based next-state guess
          2) Use physics_net (LSTM) to refine that guess
          3) Update covariance with predicted process noise
        """
        try:
            # 1) Physics-based prediction
            physics_pred = self.system.dynamics(self.x_hat, dt)

            # 2) Neural correction
            states_array = np.array(list(self.state_buffer))
            if len(states_array) < self.config.sequence_length:
                # Not enough historical data, skip correction
                corrected_state = physics_pred
            else:
                states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)
                physics_prior = torch.FloatTensor(physics_pred).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    corrected_tensor, _ = self.physics_net(states_tensor, physics_prior)
                corrected_state = corrected_tensor.squeeze(0).cpu().numpy()

            self.x_hat = corrected_state

            # 3) Uncertainty estimation
            uncertainty = self._get_uncertainty_estimates()
            Q_raw = uncertainty['Q_mean'] + uncertainty['Q_std']
            Q_np = Q_raw.squeeze(0).cpu().numpy()
            Q_np = 0.5 * (Q_np + Q_np.T)

            F = self.system.jacobian(self.x_hat)
            self.P = F @ self.P @ F.T + Q_np
            self.P = 0.5 * (self.P + self.P.T)

            # Buffer update
            self._update_buffers(self.x_hat)

        except Exception as e:
            logging.error(f"Error in predict step: {e}")
            raise

    def update(self, measurement: np.ndarray) -> None:
        """
        EKF update step:
          1) Validate states/covariance
          2) Construct measurement model
          3) Compute Kalman gain & correct state
          4) Update covariance
          5) Online learning
        """
        try:
            # 1) Validate
            measurement = self.state_validator.validate_state(measurement).squeeze()
            self.x_hat = self.state_validator.validate_state(self.x_hat).squeeze()
            self.P = self.state_validator.validate_covariance(self.P).squeeze()

            # 2) Partial measurement model (H)
            H = np.zeros((self.config.measurement_dim, self.config.state_dim))
            H[0, 0] = 1.0  # observe x
            H[1, 2] = 1.0  # observe z

            measured_components = measurement.astype(np.float64)

            # 3) Uncertainty: R
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
            innovation = measured_components - (H @ self.x_hat)
            self.x_hat = self.x_hat + (K @ innovation)

            I = np.eye(self.config.state_dim)
            KH = K @ H
            self.P = (I - KH) @ self.P @ (I - KH).T + K @ R_np @ K.T

            self.x_hat = self.state_validator.validate_state(self.x_hat)
            self.P = self.state_validator.validate_covariance(self.P)

            # 5) Online learning
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
            # Default
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

        # Convert buffers to tensors
        state_seq = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(DEVICE)
        innov_seq = torch.FloatTensor(list(self.innovation_buffer)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.uncertainty_net(state_seq, innov_seq, dt=self.config.dt)
        return outputs

    def _update_buffers(self, state: np.ndarray, innovation: Optional[np.ndarray] = None) -> None:
        """
        Store new state and innovation in ring buffers.
        """
        self.state_buffer.append(state)
        if innovation is not None:
            self.innovation_buffer.append(innovation)

    def _online_learning(self,
                         measurement: np.ndarray,
                         innovation: np.ndarray,
                         uncertainty: Dict[str, torch.Tensor]) -> None:
        """
        Train the physics_net and uncertainty_net using the most recent data.
        """
        # ----------------- Physics Net Training -----------------
        self.physics_optimizer.zero_grad()

        if len(self.state_buffer) == self.config.sequence_length:
            states_array = np.array(list(self.state_buffer))
            states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)

            last_state = states_array[-1]
            physics_pred_np = self.system.dynamics(last_state, self.config.dt)
            physics_pred_tensor = torch.FloatTensor(physics_pred_np).unsqueeze(0).to(DEVICE)

            measurement_tensor = torch.FloatTensor(measurement).unsqueeze(0).to(DEVICE)

            corrected, physics_loss = self.physics_net(states_tensor, physics_pred_tensor)
            # corrected is shape (B, 3); measurement_tensor is shape (B, 2)
            partial_corrected = torch.stack([corrected[:, 0], corrected[:, 2]], dim=-1)
            partial_loss = F.mse_loss(partial_corrected, measurement_tensor)

            physics_total_loss = partial_loss + 0.1 * physics_loss


            physics_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(), 1.0)
            self.physics_optimizer.step()

            self.metrics['prediction_loss'].append(partial_loss.item())

        # ----------------- Uncertainty Net Training -----------------
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
        # Gaussian log-likelihood:
        ll = -0.5 * (2 * np.log(2.0 * np.pi) + logdetR + mahalanobis)
        return -ll.squeeze()


class StandardEKF:
    """
    A standard Extended Kalman Filter for comparison.
    """

    def __init__(self, system: LorenzSystem, state_dim: int = 3) -> None:
        self.system = system
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def predict(self, dt: float) -> None:
        """
        Classic EKF prediction with fixed Q.
        """
        self.x_hat = self.system.dynamics(self.x_hat, dt)
        F = self.system.jacobian(self.x_hat)
        Q = np.eye(self.state_dim) * 0.1
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: np.ndarray) -> None:
        """
        Classic EKF update with partial measurement of x and z.
        """
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        R = np.eye(2) * 0.1

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        innovation = measurement - (H @ self.x_hat)
        self.x_hat = self.x_hat + K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P


class UnscentedKF:
    """
    A simple Unscented Kalman Filter for comparison (constant Q/R).
    """

    def __init__(self, system: LorenzSystem, state_dim: int = 3) -> None:
        self.system = system
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)

        # UKF parameters
        self.alpha = 0.3
        self.beta = 2.0
        self.kappa = 0.0
        self.n = state_dim
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights
        self.weights_m = np.zeros(2 * self.n + 1)
        self.weights_c = np.zeros(2 * self.n + 1)
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2*self.n + 1):
            self.weights_m[i] = 1.0 / (2.0 * (self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def predict(self, dt: float) -> None:
        """
        UKF prediction using sigma points.
        """
        sigma_points = self._generate_sigma_points()
        propagated = np.array([self.system.dynamics(sp, dt) for sp in sigma_points])

        # Predicted mean
        self.x_hat = np.sum(self.weights_m[:, None] * propagated, axis=0)

        # Predicted covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = (propagated[i] - self.x_hat).reshape(-1, 1)
            P_pred += self.weights_c[i] * (diff @ diff.T)
        Q = np.eye(self.state_dim) * 0.1
        self.P = P_pred + Q

    def update(self, measurement: np.ndarray) -> None:
        """
        UKF update step with partial measurement of x and z.
        """
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        R = np.eye(2) * 0.1

        sigma_points = self._generate_sigma_points()
        projected = np.array([H @ sp for sp in sigma_points])

        y_mean = np.sum(self.weights_m[:, None] * projected, axis=0)

        # Covariances
        Pyy = np.zeros((2, 2))
        Pxy = np.zeros((self.n, 2))
        for i in range(2*self.n + 1):
            dy = projected[i] - y_mean
            dx = sigma_points[i] - self.x_hat
            Pyy += self.weights_c[i] * np.outer(dy, dy)
            Pxy += self.weights_c[i] * np.outer(dx, dy)

        Pyy += R
        K = Pxy @ np.linalg.inv(Pyy)

        innovation = measurement - y_mean
        self.x_hat = self.x_hat + K @ innovation
        self.P = self.P - K @ Pyy @ K.T

    def _generate_sigma_points(self) -> np.ndarray:
        """
        Generate sigma points from x_hat and P with enhanced numerical stability.
        """
        sigma_points = np.zeros((2*self.n + 1, self.n))
        sigma_points[0] = self.x_hat

        scaled_cov = (self.n + self.lambda_) * self.P
        
        # Ensure symmetry
        scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)
        
        # Check eigenvalues
        eigvals = np.linalg.eigvals(scaled_cov)
        min_eig = np.min(np.real(eigvals))
        
        # Add stronger regularization if needed
        reg_factor = 1e-3
        if min_eig < reg_factor:
            reg_matrix = reg_factor * np.eye(self.n)
            scaled_cov += reg_matrix
        
        try:
            # Try standard Cholesky first
            sqrt_cov = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            try:
                # If that fails, try with additional regularization
                scaled_cov += 1e-2 * np.eye(self.n)
                sqrt_cov = np.linalg.cholesky(scaled_cov)
            except np.linalg.LinAlgError:
                # If still fails, use eigendecomposition as fallback
                eigvals, eigvecs = np.linalg.eigh(scaled_cov)
                eigvals = np.maximum(eigvals, 1e-3)
                sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals))

        for i in range(self.n):
            sigma_points[i+1] = self.x_hat + sqrt_cov[i]
            sigma_points[self.n + i + 1] = self.x_hat - sqrt_cov[i]

        return sigma_points


# -----------------------------------------------------------------------------
#                      PART 5: Simulation & Visualization
# -----------------------------------------------------------------------------
def simulate_system(
    config: HybridEKFConfig,
    time_steps: int = 5000,
    save_path: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Run a simulation of the Lorenz system with the IntegratedHybridEKF.
    """
    system = LorenzSystem()
    ekf = IntegratedHybridEKF(system, config)

    # Storage
    true_states = np.zeros((time_steps, config.state_dim))
    estimated_states = np.zeros_like(true_states)
    uncertainties = np.zeros((time_steps, 2))  # store [q_conf, r_conf]

    # Initial state
    x_true = np.array([1.0, 1.0, 1.0])

    for t in range(time_steps):
        if t % 100 == 0:
            logging.info(f"Simulation step {t}/{time_steps}")

        # System evolves
        x_true = system.dynamics(x_true, config.dt)
        true_states[t] = x_true

        # Noisy measurement
        # New: measure only x,z
        measured_xz = x_true[[0, 2]] + np.random.normal(
            0, config.measurement_noise_init, size=2
        )

        # EKF steps
        ekf.predict(config.dt)
        ekf.update(measured_xz)

        estimated_states[t] = ekf.x_hat

        # Attempt to record Q/R confidence if available
        if ekf.metrics['q_confidence'] and ekf.metrics['r_confidence']:
            uncertainties[t, 0] = ekf.metrics['q_confidence'][-1]
            uncertainties[t, 1] = ekf.metrics['r_confidence'][-1]

    results = {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'uncertainties': uncertainties,
        'metrics': ekf.metrics
    }

    if save_path:
        np.savez(save_path, **results)

    return results


def simulate_comparative_system(
    time_steps: int = 5000,
    dt: float = 0.01,
    config: Optional[HybridEKFConfig] = None
) -> Dict[str, np.ndarray]:
    """
    Run a comparative simulation with:
      - IntegratedHybridEKF
      - StandardEKF
      - UnscentedKF

    Returns a dictionary with states from each filter plus the true states.
    """
    if config is None:
        config = HybridEKFConfig(dt=dt)

    # Create a single system for generating true states
    true_system = LorenzSystem(time_varying=True, variation_amplitude=0.9)
    filter_system = LorenzSystem(time_varying=False)  # Filters assume constant parameters    
    # Initialize the three filters with the same system
    hybrid_ekf = IntegratedHybridEKF(filter_system, config)
    standard_ekf = StandardEKF(filter_system, state_dim=config.state_dim)
    ukf = UnscentedKF(filter_system, state_dim=config.state_dim)

    # Allocate storage
    true_states = np.zeros((time_steps, config.state_dim))
    hybrid_states = np.zeros_like(true_states)
    standard_states = np.zeros_like(true_states)
    ukf_states = np.zeros_like(true_states)

    x_true = np.array([1.0, 1.0, 1.0])  # initial condition

    for t in range(time_steps):
        if t % 100 == 0:
            logging.info(f"Comparative simulation step {t}/{time_steps}")

        # Evolve the true system
        x_true = true_system.dynamics(x_true, dt, t)
        true_states[t] = x_true

        # Noisy measurement
        measured_xz = x_true[[0, 2]] + np.random.normal(
            0, config.measurement_noise_init, size=2
        )

        # --- Hybrid EKF ---
        hybrid_ekf.predict(dt)
        hybrid_ekf.update(measured_xz)
        hybrid_states[t] = hybrid_ekf.x_hat

        # --- Standard EKF ---
        standard_ekf.predict(dt)
        standard_ekf.update(measured_xz)
        standard_states[t] = standard_ekf.x_hat

        # --- Unscented KF ---
        ukf.predict(dt)
        ukf.update(measured_xz)
        ukf_states[t] = ukf.x_hat

    return {
        'true_states': true_states,
        'hybrid_states': hybrid_states,
        'standard_states': standard_states,
        'ukf_states': ukf_states
    }


class ResultVisualizer:
    """
    Visualization for analyzing estimation results (single filter).
    """
    def __init__(self, results: Dict[str, np.ndarray]):
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]

    def plot_state_trajectories(self, save_path: Optional[Path] = None):
        """
        Plot state trajectories with optional 2-sigma bounds 
        (if a single measure of uncertainty is available).
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        true_states = self.results['true_states']
        estimated_states = self.results['estimated_states']
        uncertainties = self.results['uncertainties']  # shape (T, 2) => not direct std dev

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

    def plot_uncertainty_evolution(self, save_path: Optional[Path] = None):
        """
        Plot evolution of Q/R confidence estimates (if recorded).
        """
        uncertainties = self.results['uncertainties']
        plt.figure(figsize=(12, 5))
        plt.plot(uncertainties[:, 0], label='Q confidence')
        plt.plot(uncertainties[:, 1], label='R confidence')
        plt.xlabel('Time step')
        plt.ylabel('Confidence')
        plt.title('Uncertainty Confidence Evolution')
        plt.legend()
        if save_path:
            plt.savefig(save_path / 'uncertainty_confidence.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_learning_metrics(self, save_path: Optional[Path] = None):
        """
        Plot recorded losses for debugging/tracking.
        """
        metrics = self.results['metrics']
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(metrics['prediction_loss'], label='Prediction Loss')
        axes[0].set_title('Physics Network Learning')
        axes[0].legend()

        axes[1].plot(metrics['uncertainty_loss'], label='Uncertainty Loss')
        axes[1].set_title('Uncertainty Network Learning')
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / 'learning_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


class ComparisonVisualizer:
    """
    Visualization for analyzing multiple filters side by side.
    """
    def __init__(self, results: Dict[str, np.ndarray]):
        """
        results keys:
            'true_states', 'hybrid_states', 'standard_states', 'ukf_states'
        """
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]

    def plot_comparative_trajectories(self, save_path: Optional[Path] = None):
        """
        Plots x, y, z trajectories for all filters + true states.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        true_states = self.results['true_states']
        hybrid_states = self.results['hybrid_states']
        standard_states = self.results['standard_states']
        ukf_states = self.results['ukf_states']

        for i, label in enumerate(['x', 'y', 'z']):
            axes[i].plot(true_states[:, i], 'k-', label='True', linewidth=2)
            axes[i].plot(hybrid_states[:, i], 'b--', label='Hybrid EKF')
            axes[i].plot(standard_states[:, i], 'r-.', label='Standard EKF')
            axes[i].plot(ukf_states[:, i], 'g:', label='Unscented KF')
            axes[i].set_ylabel(label)
            axes[i].legend()

        axes[-1].set_xlabel('Time step')
        plt.suptitle('Comparative State Trajectories')
        if save_path:
            plt.savefig(save_path / 'comparative_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()


# -----------------------------------------------------------------------------
#                          MAIN EXECUTION
# -----------------------------------------------------------------------------
def main(mode: str = 'default'):
    """
    Main execution function with selectable mode:
      - 'default': run a single integrated simulation
      - 'comparative': compare filters side by side
    """
    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = Path('hybrid_ekf_results_seq')
    output_dir.mkdir(exist_ok=True)

    if mode == 'default':
        logging.info("Running default simulation mode with IntegratedHybridEKF...")

        config = HybridEKFConfig(
            state_dim=3,
            measurement_dim=2,
            hidden_dim=32,
            sequence_length=20,
            dt=0.01,
            learning_rate=0.001
        )

        results = simulate_system(
            config=config,
            time_steps=5000,
            save_path=output_dir / 'simulation_results.npz'
        )

        # Visualization
        viz = ResultVisualizer(results)
        viz.plot_state_trajectories(output_dir)
        viz.plot_uncertainty_evolution(output_dir)
        viz.plot_learning_metrics(output_dir)

        rmse = np.sqrt(np.mean((results['true_states'] - results['estimated_states'])**2))
        logging.info(f"Final RMSE: {rmse:.4f}")

    else:
        logging.info("Running comparative simulation mode...")

        # You can adjust time_steps and dt as desired
        comparative_results = simulate_comparative_system(
            time_steps=5000,
            dt=0.01
        )

        # Visualization
        comp_viz = ComparisonVisualizer(comparative_results)
        comp_viz.plot_comparative_trajectories(output_dir)

        # Compute and log final RMSE for each filter
        true_states = comparative_results['true_states']
        hybrid_states = comparative_results['hybrid_states']
        standard_states = comparative_results['standard_states']
        ukf_states = comparative_results['ukf_states']

        hybrid_rmse = np.sqrt(np.mean((true_states - hybrid_states) ** 2))
        standard_rmse = np.sqrt(np.mean((true_states - standard_states) ** 2))
        ukf_rmse = np.sqrt(np.mean((true_states - ukf_states) ** 2))

        logging.info(f"Hybrid EKF RMSE:   {hybrid_rmse:.4f}")
        logging.info(f"Standard EKF RMSE:{standard_rmse:.4f}")
        logging.info(f"Unscented KF RMSE:{ukf_rmse:.4f}")

        # You might also add other plots (e.g., error evolution, etc.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid EKF Tutorial (Sequential Physics Net)')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'comparative'],
        default='default',
        help='Simulation mode: default or comparative'
    )
    args = parser.parse_args()

    try:
        main(mode=args.mode)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
