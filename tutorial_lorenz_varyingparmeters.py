#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Hybrid Extended Kalman Filter Tutorial with Sequential Physics Network,
Warmup/Shadow Mode, and Best Practices

This version includes:
1. A "shadow" or "warmup" mode for the first N steps (collect data but do not
   train the neural networks or apply their corrections).
2. After warmup, the filter transitions to the usual Hybrid EKF with online 
   learning and neural corrections.
3. A 'comparative' mode to compare:
   - IntegratedHybridEKF
   - StandardEKF
   - UnscentedKF

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
from typing import Dict, Optional, Tuple, Deque, List

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
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
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
    learning_rate: float = 1e-3
    dt: float = 0.01
    min_covariance: float = 1e-10
    process_noise_init: float = 0.1
    measurement_noise_init: float = 0.1

    # NEW: Number of steps to run in 'shadow' mode before full online learning
    warmup_steps: int = 3000


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
    """Lorenz system for demonstration."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3,
                 unmodeled_dynamics: bool = False):
        """
        Initialize Lorenz system with optional unmodeled dynamics.
        
        Args:
            sigma: First parameter
            rho: Second parameter
            beta: Third parameter
            unmodeled_dynamics: Whether to include unmodeled terms
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.unmodeled_dynamics = unmodeled_dynamics
        
        # Parameters for unmodeled dynamics
        if unmodeled_dynamics:
            self.epsilon = 5.1  # Strength of unmodeled terms
            self.omega = 2.0    # Frequency of oscillation
            
    def derivatives(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Compute derivatives with optional unmodeled dynamics."""
        x, y, z = state
        
        # Standard Lorenz terms
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        if self.unmodeled_dynamics:
            # Add time-varying perturbations
            dx += self.epsilon * np.sin(self.omega * t) * y
            dy += self.epsilon * np.cos(self.omega * t) * x
            dz += self.epsilon * np.sin(self.omega * t + np.pi/4) * z
            
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
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])
        return J


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
            dropout=0.2
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


class SimplifiedUncertaintyNetwork(nn.Module):
    """
    A streamlined uncertainty network that estimates process (Q) and measurement (R) 
    noise covariances for Kalman filtering.
    """
    def __init__(self,
                state_dim: int = 3,
                measurement_dim: int = 2,
                hidden_dim: int = 64,
                sequence_length: int = 5) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.sequence_length = sequence_length

        # Single state encoder with normalization
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Single-layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Simplified process noise (Q) estimator
        self.q_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )

        # Simplified measurement noise (R) estimator
        self.r_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, measurement_dim * measurement_dim)
        )

        # Single confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Simple orthogonal initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, 
                state_sequence: torch.Tensor,
                innovation_sequence: torch.Tensor,
                dt: float = 0.01) -> dict:
        """
        Forward pass to estimate Q and R matrices.
        
        Args:
            state_sequence: Shape (batch_size, sequence_length, state_dim)
            innovation_sequence: Shape (batch_size, sequence_length, measurement_dim)
            dt: Time step for scaling Q

        Returns:
            Dictionary containing Q and R estimates with confidences
        """
        batch_size = state_sequence.size(0)
        
        # Encode states
        encoded = self.state_encoder(state_sequence)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(encoded)
        hidden = lstm_out[:, -1, :]  # Use final timestep
        
        # Estimate Q
        q_raw = self.q_estimator(hidden)
        Q = q_raw.view(batch_size, self.state_dim, self.state_dim)
        Q = 0.5 * (Q + Q.transpose(-2, -1))  # Ensure symmetry
        Q = F.softplus(Q) * dt  # Ensure positive definiteness and scale
        
        # Estimate R
        r_raw = self.r_estimator(hidden)
        R = r_raw.view(batch_size, self.measurement_dim, self.measurement_dim)
        R = 0.5 * (R + R.transpose(-2, -1))  # Ensure symmetry
        R = F.softplus(R)  # Ensure positive definiteness
        
        # Estimate confidences
        confidence = self.confidence_net(hidden)
        q_conf, r_conf = confidence[:, 0], confidence[:, 1]
        
        return {
            'Q': Q,
            'R': R,
            'q_confidence': q_conf,
            'r_confidence': r_conf
        }

    def get_regularization_loss(self) -> torch.Tensor:
        """Simple L2 regularization on all parameters."""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return 0.01 * l2_loss


# -----------------------------------------------------------------------------
#                PART 4: Integrated Hybrid EKF + Reference Filters
# -----------------------------------------------------------------------------
class ExperienceBuffer:
    """Buffer for experience replay in continual learning."""
    def __init__(self, buffer_size: int = 1000) -> None:
        self.states = deque(maxlen=buffer_size)
        self.measurements = deque(maxlen=buffer_size)
        self.innovations = deque(maxlen=buffer_size)
        
    def add(self, state: np.ndarray, measurement: np.ndarray, innovation: np.ndarray) -> None:
        """Add experience to buffer."""
        self.states.append(state.copy())
        self.measurements.append(measurement.copy())
        self.innovations.append(innovation.copy())
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random batch of experiences."""
        indices = np.random.choice(len(self.states), min(batch_size, len(self.states)))
        return (
            torch.FloatTensor([self.states[i] for i in indices]).to(DEVICE),
            torch.FloatTensor([self.measurements[i] for i in indices]).to(DEVICE),
            torch.FloatTensor([self.innovations[i] for i in indices]).to(DEVICE)
        )


class IntegratedHybridEKF:
    """
    Hybrid EKF that merges:
      - Physics-based prediction (LorenzSystem + sequential PhysicsNetwork)
      - Uncertainty estimation from SimplifiedUncertaintyNetwork
      - Extended Kalman Filter update
      - Online learning with backprop
      - **Shadow/Warmup mode** for first config.warmup_steps (no neural training/correction)
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

        self.uncertainty_net = SimplifiedUncertaintyNetwork(
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

        # NEW: Global counter to track steps (for warmup vs normal operation)
        self.global_step = 0

        # NEW: Add buffer for pre-training data
        self.pretraining_states: List[np.ndarray] = []
        self.pretraining_measurements: List[np.ndarray] = []
        self.pretraining_innovations: List[np.ndarray] = []

        # Add experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Store pretrained parameters after warmup
        self.pretrained_physics = None
        self.pretrained_uncertainty = None

    def predict(self, dt: float) -> None:
        """
        EKF prediction step with constant Q during warmup.
        """
        try:
            # 1) Pure physics-based prediction
            physics_pred = self.system.dynamics(self.x_hat, dt)

            # 2) Neural correction only if outside warmup
            if self.global_step >= self.config.warmup_steps:
                states_array = np.array(list(self.state_buffer))
                if len(states_array) < self.config.sequence_length:
                    corrected_state = physics_pred
                else:
                    states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)
                    physics_prior = torch.FloatTensor(physics_pred).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        corrected_tensor, _ = self.physics_net(states_tensor, physics_prior)
                    corrected_state = corrected_tensor.squeeze(0).cpu().numpy()
            else:
                # During warmup, skip neural correction
                corrected_state = physics_pred

            self.x_hat = corrected_state

            # 3) Use constant Q during warmup, uncertainty network after
            if self.global_step < self.config.warmup_steps:
                # Use constant Q like StandardEKF during warmup
                Q = np.eye(self.config.state_dim) * self.config.process_noise_init
            else:
                # Use uncertainty network estimates after warmup
                uncertainty = self._get_uncertainty_estimates()
                Q = uncertainty['Q'].squeeze(0).cpu().numpy()
                Q = 0.5 * (Q + Q.T)  # ensure symmetry

            F = self.system.jacobian(self.x_hat)
            self.P = F @ self.P @ F.T + Q
            self.P = 0.5 * (self.P + self.P.T)

            # Buffer update
            self._update_buffers(self.x_hat)

        except Exception as e:
            logging.error(f"Error in predict step: {e}")
            raise

    def update(self, measurement: np.ndarray) -> None:
        """
        EKF update step with simplified uncertainty estimates
        """
        try:
            # Validate states/covariance
            measurement = self.state_validator.validate_state(measurement).squeeze()
            self.x_hat = self.state_validator.validate_state(self.x_hat).squeeze()
            self.P = self.state_validator.validate_covariance(self.P).squeeze()

            # Partial measurement model (H)
            H = np.zeros((self.config.measurement_dim, self.config.state_dim))
            H[0, 0] = 1.0  # observe x
            H[1, 2] = 1.0  # observe z

            measured_components = measurement.astype(np.float64)

            # Use constant R during warmup, uncertainty network after
            if self.global_step < self.config.warmup_steps:
                # Use constant R like StandardEKF during warmup
                R_np = np.eye(self.config.measurement_dim) * self.config.measurement_noise_init
            else:
                # Use uncertainty network estimates after warmup
                uncertainty = self._get_uncertainty_estimates()
                R_np = uncertainty['R'].squeeze(0).cpu().numpy()
                R_np = 0.5 * (R_np + R_np.T)  # ensure symmetry
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

            # Final validations
            self.x_hat = self.state_validator.validate_state(self.x_hat)
            self.P = self.state_validator.validate_covariance(self.P)

            # Update ring buffers with the new state and the measurement innovation
            self._update_buffers(self.x_hat, innovation)

            # Store data for pre-training during shadow mode
            if self.global_step < self.config.warmup_steps:
                self.pretraining_states.append(self.x_hat.copy())
                self.pretraining_measurements.append(measurement.copy())
                self.pretraining_innovations.append(innovation.copy())
                
                # Pre-train networks when we have enough shadow mode data
                if self.global_step == self.config.warmup_steps - 1:
                    self._pretrain_networks()

            # Online learning only if outside warmup
            if self.global_step >= self.config.warmup_steps and not np.any(np.isnan(self.x_hat)):
                self._online_learning(measurement, innovation, uncertainty)

            # Increase global step after each update
            self.global_step += 1

        except Exception as e:
            logging.error(f"Error in update step: {e}")
            raise

    def _get_uncertainty_estimates(self) -> Dict[str, torch.Tensor]:
        """
        Returns Q/R from the SimplifiedUncertaintyNetwork.
        If insufficient data, return default values.
        """
        if len(self.state_buffer) < self.config.sequence_length:
            # Default
            eye_Q = torch.eye(self.config.state_dim, device=DEVICE) * self.config.process_noise_init
            eye_R = torch.eye(self.config.measurement_dim, device=DEVICE) * self.config.measurement_noise_init
            return {
                'Q': eye_Q.unsqueeze(0),
                'R': eye_R.unsqueeze(0),
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
    def _compute_uncertainty_loss(self,
                                  innovation: np.ndarray,
                                  R: torch.Tensor,
                                  _: None) -> torch.Tensor:
        """
        Negative log-likelihood for measurement innovation ~ N(0, R).
        """
        innovation_tensor = torch.FloatTensor(innovation).to(DEVICE)
        R_total = R  # no separate std in this simplified approach
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
    def _online_learning(self,
                         measurement: np.ndarray,
                         innovation: np.ndarray,
                         uncertainty: Dict[str, torch.Tensor]) -> None:
        """Enhanced online learning with experience replay and EWC."""
        # Add current experience to buffer
        self.experience_buffer.add(self.x_hat, measurement, innovation)
        
        if len(self.state_buffer) == self.config.sequence_length:
            # Current batch processing
            states_tensor = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(DEVICE)
            last_state = self.state_buffer[-1]
            physics_pred = self.system.dynamics(last_state, self.config.dt)
            physics_tensor = torch.FloatTensor(physics_pred).unsqueeze(0).to(DEVICE)
            measurement_tensor = torch.FloatTensor(measurement).unsqueeze(0).to(DEVICE)
            
            # Get replay batch
            replay_states, replay_meas, replay_inn = self.experience_buffer.sample(32)
            
            # Physics network training
            self.physics_optimizer.zero_grad()
            
            # Current loss
            corrected, physics_loss = self.physics_net(states_tensor, physics_tensor)
            partial_corrected = torch.stack([corrected[:, 0], corrected[:, 2]], dim=-1)
            current_loss = F.mse_loss(partial_corrected, measurement_tensor)
            
            # Replay loss
            replay_loss = torch.tensor(0.0).to(DEVICE)
            if len(replay_states) > 0:
                replay_pred = torch.stack([
                    torch.FloatTensor(self.system.dynamics(s.cpu().numpy(), self.config.dt))
                    for s in replay_states
                ]).to(DEVICE)
                
                # Fix: Ensure proper reshaping for sequence input
                replay_states_seq = replay_states.unsqueeze(0)  # Add sequence dimension
                replay_pred_seq = replay_pred.unsqueeze(0)      # Add sequence dimension
                
                replay_corrected, _ = self.physics_net(replay_states_seq, replay_pred_seq)
                # Fix: Properly extract the full state vector
                replay_corrected = replay_corrected.squeeze(0)  # Remove sequence dimension
                
                # Now correctly extract x and z components
                replay_partial = torch.stack([
                    replay_corrected[:, 0],  # x component
                    replay_corrected[:, 2]   # z component
                ], dim=1)  # Changed dim=-1 to dim=1 for explicit dimension handling
                
                replay_loss = F.mse_loss(replay_partial, replay_meas)
            
            # EWC loss
            ewc_physics = self.compute_ewc_loss(
                dict(self.physics_net.named_parameters()),
                self.pretrained_physics
            )
            
            # Total physics loss
            physics_total_loss = current_loss + 0.1 * physics_loss + 0.3 * replay_loss + ewc_physics
            physics_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(), 1.0)
            self.physics_optimizer.step()
            
            # Uncertainty network training
            self.uncertainty_optimizer.zero_grad()
            
            current_uncert = self.uncertainty_net(states_tensor, 
                                                torch.FloatTensor(list(self.innovation_buffer)).unsqueeze(0).to(DEVICE),
                                                dt=self.config.dt)
            
            uncertainty_loss = self._compute_uncertainty_loss(innovation, current_uncert['R'], None)
            ewc_uncertainty = self.compute_ewc_loss(
                dict(self.uncertainty_net.named_parameters()),
                self.pretrained_uncertainty
            )
            
            total_uncertainty_loss = uncertainty_loss + ewc_uncertainty
            total_uncertainty_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.uncertainty_net.parameters(), 1.0)
            self.uncertainty_optimizer.step()
            
            # Record metrics
            self.metrics['prediction_loss'].append(current_loss.item())
            self.metrics['uncertainty_loss'].append(uncertainty_loss.item())
            self.metrics['q_confidence'].append(uncertainty['q_confidence'].item())
            self.metrics['r_confidence'].append(uncertainty['r_confidence'].item())

    def store_pretrained_params(self) -> None:
        """Store network parameters after pretraining."""
        self.pretrained_physics = {
            name: param.clone().detach()
            for name, param in self.physics_net.named_parameters()
        }
        self.pretrained_uncertainty = {
            name: param.clone().detach()
            for name, param in self.uncertainty_net.named_parameters()
        }

    def compute_ewc_loss(self, current_params: Dict[str, torch.Tensor], 
                        pretrained_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Elastic Weight Consolidation loss."""
        if pretrained_params is None:
            return torch.tensor(0.0).to(DEVICE)
            
        ewc_lambda = 0.4
        loss = torch.tensor(0.0).to(DEVICE)
        for name in current_params:
            loss += ewc_lambda * torch.sum(
                (current_params[name] - pretrained_params[name])**2
            )
        return loss

    def _pretrain_networks(self) -> None:
        """Pre-train both networks using shadow mode data before online learning."""
        logging.info("Pre-training neural networks with shadow mode data...")
        
        # Convert collected data to tensors
        states_tensor = torch.FloatTensor(self.pretraining_states).to(DEVICE)
        measurements_tensor = torch.FloatTensor(self.pretraining_measurements).to(DEVICE)
        innovations_tensor = torch.FloatTensor(self.pretraining_innovations).to(DEVICE)
        
        # Pre-train physics network
        for epoch in range(50):
            for i in range(0, len(states_tensor) - self.config.sequence_length, 32):
                self.physics_optimizer.zero_grad()
                
                batch_states = states_tensor[i:i+self.config.sequence_length]
                batch_measurements = measurements_tensor[i:i+self.config.sequence_length]
                
                physics_pred = torch.FloatTensor([
                    self.system.dynamics(s.cpu().numpy(), self.config.dt) 
                    for s in batch_states
                ]).to(DEVICE)
                
                corrected, physics_loss = self.physics_net(
                    batch_states.unsqueeze(0), 
                    physics_pred.unsqueeze(0)
                )
                
                # Extract only x and z components to match measurements
                corrected_xz = torch.stack([
                    corrected[..., 0],  # x component
                    corrected[..., 2]   # z component
                ], dim=-1).squeeze(0)
                
                prediction_loss = F.mse_loss(corrected_xz, batch_measurements)
                total_loss = prediction_loss + 0.1 * physics_loss
                
                total_loss.backward()
                self.physics_optimizer.step()

        # Pre-train uncertainty network
        # Create target covariances (same as standard EKF)
        Q_target = torch.eye(self.config.state_dim, device=DEVICE) * self.config.process_noise_init
        R_target = torch.eye(self.config.measurement_dim, device=DEVICE) * self.config.measurement_noise_init

        for epoch in range(50):
            for i in range(0, len(states_tensor) - self.config.sequence_length, 32):
                self.uncertainty_optimizer.zero_grad()
                
                batch_states = states_tensor[i:i+self.config.sequence_length]
                batch_innovations = innovations_tensor[i:i+self.config.sequence_length]
                
                # Get network predictions
                uncertainty_pred = self.uncertainty_net(
                    batch_states.unsqueeze(0),
                    batch_innovations.unsqueeze(0),
                    dt=self.config.dt
                )
                
                # Losses for Q and R predictions
                q_loss = F.mse_loss(uncertainty_pred['Q'].squeeze(0), Q_target)
                r_loss = F.mse_loss(uncertainty_pred['R'].squeeze(0), R_target)
                
                # Add confidence losses (should start confident during warmup)
                confidence_target = torch.ones(2, device=DEVICE) * 0.9  # High initial confidence
                confidence_loss = F.mse_loss(
                    torch.stack([uncertainty_pred['q_confidence'], 
                               uncertainty_pred['r_confidence']]),
                    confidence_target
                )
                
                # Regularization
                reg_loss = self.uncertainty_net.get_regularization_loss()
                
                # Total loss
                total_loss = q_loss + r_loss + 0.1 * confidence_loss + 0.01 * reg_loss
                
                total_loss.backward()
                self.uncertainty_optimizer.step()
        
        logging.info("Pre-training completed. Transitioning to online learning mode.")
        
        # Clear pre-training buffers to free memory
        self.pretraining_states.clear()
        self.pretraining_measurements.clear()
        self.pretraining_innovations.clear()

        # Store pretrained parameters after pretraining
        self.store_pretrained_params()
        logging.info("Stored pretrained parameters for EWC.")

# Add new Hybrid UKF class
    """
    Hybrid Unscented Kalman Filter combining:
    - UKF's sigma point approach
    - Neural network corrections
    - Online uncertainty estimation
    - Warmup/shadow mode compatibility
    """
    
    def __init__(self, system: LorenzSystem, config: HybridEKFConfig):
        super().__init__(system, config)
        
        # UKF-specific parameters
        self.alpha = 0.3
        self.beta = 2.0
        self.kappa = 0.0
        self.n = config.state_dim
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Initialize weights
        self.weights_m = np.zeros(2*self.n + 1)
        self.weights_c = np.zeros(2*self.n + 1)
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.lambda_/(self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2*self.n + 1):
            self.weights_m[i] = 1/(2*(self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def predict(self, dt: float) -> None:
        """Hybrid UKF prediction with neural corrections"""
        try:
            # Generate sigma points
            sigma_points = self._generate_sigma_points()
            
            # Propagate through physics model
            propagated = np.array([self.system.dynamics(sp, dt) for sp in sigma_points])
            
            # Neural correction if past warmup
            if self.global_step >= self.config.warmup_steps:
                corrected_points = []
                for sp in propagated:
                    # Convert to tensor format for neural network
                    state_seq = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(DEVICE)
                    physics_prior = torch.FloatTensor(sp).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        corrected, _ = self.physics_net(
                            state_seq, 
                            physics_prior.unsqueeze(0)
                        )
                    corrected_points.append(corrected.squeeze().cpu().numpy())
                propagated = np.array(corrected_points)
            
            # Calculate predicted state and covariance
            self.x_hat = np.sum(self.weights_m[:, None] * propagated, axis=0)
            
            # Process noise handling
            if self.global_step < self.config.warmup_steps:
                Q = np.eye(self.n) * self.config.process_noise_init
            else:
                uncertainty = self._get_uncertainty_estimates()
                Q = uncertainty['Q'].squeeze(0).cpu().numpy()
                Q = 0.5 * (Q + Q.T)
            
            # Calculate predicted covariance
            P_pred = np.zeros((self.n, self.n))
            for i in range(2*self.n + 1):
                diff = (propagated[i] - self.x_hat).reshape(-1, 1)
                P_pred += self.weights_c[i] * (diff @ diff.T)
            self.P = P_pred + Q
            
            # Buffer update
            self._update_buffers(self.x_hat)

        except Exception as e:
            logging.error(f"Hybrid UKF predict error: {e}")
            raise

    def update(self, measurement: np.ndarray) -> None:
        """Hybrid UKF update with neural uncertainty estimation"""
        try:
            # Generate sigma points
            sigma_points = self._generate_sigma_points()
            
            # Measurement projection
            H = np.zeros((2, self.n))
            H[0, 0] = 1.0  # observe x
            H[1, 2] = 1.0  # observe z
            projected = np.array([H @ sp for sp in sigma_points])
            
            # Measurement noise handling
            if self.global_step < self.config.warmup_steps:
                R = np.eye(2) * self.config.measurement_noise_init
            else:
                uncertainty = self._get_uncertainty_estimates()
                R = uncertainty['R'].squeeze(0).cpu().numpy()
                R = 0.5 * (R + R.T)
            
            # UKF update steps
            y_mean = np.sum(self.weights_m[:, None] * projected, axis=0)
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
            
            # Neural innovation adjustment (post-warmup)
            if self.global_step >= self.config.warmup_steps:
                innov_tensor = torch.FloatTensor(innovation).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    uncertainty = self._get_uncertainty_estimates()
                    R_net = uncertainty['R'].squeeze(0).cpu().numpy()
                    K = K @ np.linalg.pinv(R) @ R_net  # Neural-adjusted gain

            self.x_hat += K @ innovation
            self.P -= K @ Pyy @ K.T
            
            # Joseph form covariance update for stability
            I = np.eye(self.n)
            KH = K @ H
            self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
            
            # Validate and update buffers
            self.x_hat = self.state_validator.validate_state(self.x_hat)
            self.P = self.state_validator.validate_covariance(self.P)
            self._update_buffers(self.x_hat, innovation)
            
            # Online learning (inherited from HybridEKF)
            if self.global_step >= self.config.warmup_steps and not np.any(np.isnan(self.x_hat)):
                self._online_learning(measurement, innovation, uncertainty)

            self.global_step += 1

        except Exception as e:
            logging.error(f"Hybrid UKF update error: {e}")
            raise

    def _generate_sigma_points(self) -> np.ndarray:
        """Enhanced sigma point generation with neural-aware regularization"""
        sigma_points = np.zeros((2*self.n + 1, self.n))
        sigma_points[0] = self.x_hat
        
        # Neural-adjusted covariance scaling
        scale_factor = 1.0
        if self.global_step >= self.config.warmup_steps:
            scale_factor = 1.0 / (1.0 + np.mean(self.metrics['q_confidence'][-10:]))
        
        scaled_cov = scale_factor * (self.n + self.lambda_) * self.P
        
        # Regularization for numerical stability
        scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)
        scaled_cov += 1e-3 * np.eye(self.n)  # Add small diagonal
        
        try:
            sqrt_cov = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(scaled_cov)
            eigvals = np.maximum(eigvals, 1e-3)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals))

        for i in range(self.n):
            sigma_points[i+1] = self.x_hat + sqrt_cov[i]
            sigma_points[self.n+i+1] = self.x_hat - sqrt_cov[i]

        return sigma_points
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
    Now includes separate metrics for warmup and post-warmup periods.
    """
    system = LorenzSystem()
    ekf = IntegratedHybridEKF(system, config)

    # Storage
    true_states = np.zeros((time_steps, config.state_dim))
    estimated_states = np.zeros_like(true_states)
    uncertainties = np.zeros((time_steps, 2))  # store [q_conf, r_conf]
    state_covariances = np.zeros((time_steps, config.state_dim, config.state_dim))

    # Initial state
    x_true = np.array([1.0, 1.0, 1.0])

    for t in range(time_steps):
        if t % 100 == 0:
            logging.info(f"Simulation step {t}/{time_steps}")

        # System evolves
        x_true = system.dynamics(x_true, config.dt)
        true_states[t] = x_true

        # Noisy measurement (x,z only)
        measured_xz = x_true[[0, 2]] + np.random.normal(
            0, config.measurement_noise_init, size=2
        )

        # EKF steps
        ekf.predict(config.dt)
        ekf.update(measured_xz)

        estimated_states[t] = ekf.x_hat
        state_covariances[t] = ekf.P.copy()  # Store covariance

        # Record uncertainties if available
        if ekf.metrics['q_confidence'] and ekf.metrics['r_confidence']:
            uncertainties[t, 0] = ekf.metrics['q_confidence'][-1]
            uncertainties[t, 1] = ekf.metrics['r_confidence'][-1]

    # Add period markers to results
    results = {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'uncertainties': uncertainties,
        'metrics': ekf.metrics,
        'state_covariances': state_covariances,
        'warmup_steps': config.warmup_steps  # Add this to results
    }

    # Calculate separate metrics for warmup and post-warmup
    warmup_rmse = np.sqrt(np.mean(
        (true_states[:config.warmup_steps] - estimated_states[:config.warmup_steps])**2
    ))
    post_warmup_rmse = np.sqrt(np.mean(
        (true_states[config.warmup_steps:] - estimated_states[config.warmup_steps:])**2
    ))
    
    results['performance_metrics'] = {
        'warmup_rmse': warmup_rmse,
        'post_warmup_rmse': post_warmup_rmse,
        # Add transition period metric (first 100 steps after warmup)
        'transition_rmse': np.sqrt(np.mean(
            (true_states[config.warmup_steps:config.warmup_steps+100] - 
             estimated_states[config.warmup_steps:config.warmup_steps+100])**2
        ))
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
    Run a comparative simulation with uncertainty tracking for Hybrid EKF, 
    Standard EKF, and Unscented KF.
    """
    if config is None:
        config = HybridEKFConfig(dt=dt)

    # Initialize systems
    true_system = LorenzSystem(unmodeled_dynamics=True)  # With unmodeled dynamics
    filter_system = LorenzSystem(unmodeled_dynamics=False)  # Basic model for filters   
    # Initialize the three filters
    hybrid_ekf = IntegratedHybridEKF(filter_system, config)
    
    standard_ekf = StandardEKF(filter_system, state_dim=config.state_dim)
    ukf = UnscentedKF(filter_system, state_dim=config.state_dim)
    

    # Allocate storage
    true_states = np.zeros((time_steps, config.state_dim))
    hybrid_states = np.zeros_like(true_states)
    standard_states = np.zeros_like(true_states)
    ukf_states = np.zeros_like(true_states)

    # Add storage for Hybrid EKF covariances
    hybrid_covariances = np.zeros((time_steps, config.state_dim, config.state_dim))

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

        # Store Hybrid EKF covariance
        hybrid_covariances[t] = hybrid_ekf.P.copy()

    # Calculate post-warmup RMSE for each filter
    warmup_steps = config.warmup_steps if config else 500  # Default warmup if no config
    
    def calculate_split_metrics(estimated, true):
        warmup_rmse = np.sqrt(np.mean(
            (true[:warmup_steps] - estimated[:warmup_steps])**2
        ))
        post_warmup_rmse = np.sqrt(np.mean(
            (true[warmup_steps:] - estimated[warmup_steps:])**2
        ))
        transition_rmse = np.sqrt(np.mean(
            (true[warmup_steps:warmup_steps+100] - 
             estimated[warmup_steps:warmup_steps+100])**2
        ))
        return {
            'warmup_rmse': warmup_rmse,
            'post_warmup_rmse': post_warmup_rmse,
            'transition_rmse': transition_rmse
        }
    
    return {
        'true_states': true_states,
        'hybrid_states': hybrid_states,
        'standard_states': standard_states,
        'ukf_states': ukf_states,
        'hybrid_covariances': hybrid_covariances,
        'warmup_steps': warmup_steps,
        'performance_metrics': {
            'hybrid': calculate_split_metrics(hybrid_states, true_states),
            'standard': calculate_split_metrics(standard_states, true_states),
            'ukf': calculate_split_metrics(ukf_states, true_states)
        }
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
        Plot state trajectories with 2-sigma uncertainty bounds.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        time_steps = len(self.results['true_states'])
        t = np.arange(time_steps)

        true_states = self.results['true_states']
        estimated_states = self.results['estimated_states']
        
        # Get uncertainty bounds if available
        if 'state_covariances' in self.results:
            covariances = self.results['state_covariances']
            std_devs = np.sqrt(np.array([np.diag(P) for P in covariances]))
            bounds_upper = estimated_states + 2 * std_devs  # 2-sigma bounds
            bounds_lower = estimated_states - 2 * std_devs
        
        state_labels = ['x', 'y', 'z']
        for i, label in enumerate(state_labels):
            axes[i].plot(t, true_states[:, i], 'k-', label='True', linewidth=2, alpha=0.7)
            axes[i].plot(t, estimated_states[:, i], 'r--', label='Estimated', linewidth=2)
            
            if 'state_covariances' in self.results:
                axes[i].fill_between(t, 
                                     bounds_lower[:, i], 
                                     bounds_upper[:, i],
                                     color='r', alpha=0.2, label='2σ bounds')
            
            axes[i].set_ylabel(f'State {label}')
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel('Time step')
        plt.suptitle('State Trajectories with Uncertainty Bounds')
        
        if save_path:
            plt.savefig(save_path / 'state_trajectories_with_uncertainty.png', 
                        dpi=300, bbox_inches='tight')
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
        Plots x, y, z trajectories for all filters + true states 
        with uncertainty bounds for the Hybrid EKF.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        time_steps = len(self.results['true_states'])
        t = np.arange(time_steps)

        true_states = self.results['true_states']
        hybrid_states = self.results['hybrid_states']
        standard_states = self.results['standard_states']
        ukf_states = self.results['ukf_states']

        # Get uncertainty bounds for Hybrid EKF if available
        if 'hybrid_covariances' in self.results:
            covariances = self.results['hybrid_covariances']
            std_devs = np.sqrt(np.array([np.diag(P) for P in covariances]))
            bounds_upper = hybrid_states + 2 * std_devs
            bounds_lower = hybrid_states - 2 * std_devs

        state_labels = ['x', 'y', 'z']
        for i, label in enumerate(state_labels):
            axes[i].plot(t, true_states[:, i], 'k-', label='True', linewidth=2, alpha=0.7)
            axes[i].plot(t, hybrid_states[:, i], 'b--', label='Hybrid EKF', linewidth=2)
            axes[i].plot(t, standard_states[:, i], 'r-.', label='Standard EKF', linewidth=1.5)
            axes[i].plot(t, ukf_states[:, i], 'g:', label='Unscented KF', linewidth=1.5)
            
            if 'hybrid_covariances' in self.results:
                axes[i].fill_between(t, 
                                     bounds_lower[:, i], 
                                     bounds_upper[:, i],
                                     color='b', alpha=0.2, label='Hybrid EKF 2σ')
            
            axes[i].set_ylabel(f'State {label}')
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel('Time step')
        plt.suptitle('Comparative State Trajectories with Uncertainty Bounds')
        
        if save_path:
            plt.savefig(save_path / 'comparative_trajectories_with_uncertainty.png', 
                        dpi=300, bbox_inches='tight')
        plt.show()


# -----------------------------------------------------------------------------
#                          MAIN EXECUTION
# -----------------------------------------------------------------------------
def main(mode: str = 'default'):
    """
    Main execution function with selectable mode:
      - 'default': run a single integrated simulation (with warmup).
      - 'comparative': compare filters side by side.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = Path('hybrid_ekf_results_seq')
    output_dir.mkdir(exist_ok=True)

    if mode == 'default':
        logging.info("Running default simulation mode with IntegratedHybridEKF + warmup/shadow...")

        config = HybridEKFConfig(
            state_dim=3,
            measurement_dim=2,
            hidden_dim=16,
            sequence_length=20,
            dt=0.01,
            learning_rate=0.001,
            warmup_steps=500  # adjust as needed
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

        # Update logging to show split metrics
        metrics = results['performance_metrics']
        logging.info("Performance Metrics:")
        logging.info(f"  Warmup RMSE:     {metrics['warmup_rmse']:.4f}")
        logging.info(f"  Post-warmup RMSE: {metrics['post_warmup_rmse']:.4f}")
        logging.info(f"  Transition RMSE:  {metrics['transition_rmse']:.4f}")

    else:
        logging.info("Running comparative simulation mode...")

        # You can adjust time_steps, dt, or warmup_steps as desired
        comparative_results = simulate_comparative_system(
            time_steps=5000,
            dt=0.01
        )

        # Visualization
        comp_viz = ComparisonVisualizer(comparative_results)
        comp_viz.plot_comparative_trajectories(output_dir)

        # Update logging for comparative results
        metrics = comparative_results['performance_metrics']
        logging.info("\nPost-warmup Performance Metrics:")
        for filter_name in ['hybrid', 'standard', 'ukf']:
            m = metrics[filter_name]
            logging.info(f"\n{filter_name.upper()} Filter:")
            logging.info(f"  Warmup RMSE:     {m['warmup_rmse']:.4f}")
            logging.info(f"  Post-warmup RMSE: {m['post_warmup_rmse']:.4f}")
            logging.info(f"  Transition RMSE:  {m['transition_rmse']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid EKF Tutorial (Sequential Physics Net + Shadow Mode)')
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
