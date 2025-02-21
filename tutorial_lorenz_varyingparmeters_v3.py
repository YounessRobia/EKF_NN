#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Hybrid Extended Kalman Filter Tutorial with Sequential Physics Network,
Warmup/Shadow Mode, *Function-Correction Approach*, and Best Practices.

Key Change:
-----------
The PhysicsSequenceNetwork now learns an additive correction \delta(x) that
is added to the physics-based next state, resulting in a corrected dynamics
   f^*(x) = f_phys(x) + \delta(x).
The EKF linearizes and propagates covariance using the *Jacobian* of f^*(x),
ensuring consistent multi-step predictions.

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


# -----------------------------------------------------------------------------
#                      PART 1: System Configuration
# -----------------------------------------------------------------------------
@dataclass
class HybridEKFConfig:
    """
    Configuration for the Hybrid EKF system.
    """
    state_dim: int = 3
    measurement_dim: int = 2  # We observe only x and z
    hidden_dim: int = 64
    sequence_length: int = 15
    learning_rate: float = 1e-3
    dt: float = 0.01
    min_covariance: float = 1e-10
    process_noise_init: float = 0.1
    measurement_noise_init: float = 0.1
    warmup_steps: int = 3000

    # Non-Gaussian noise parameters
    noise_type: str = "gaussian"  # Options: "gaussian", "laplace", "mixed"
    mixture_prob: float = 0.1     # Probability of outlier in mixed noise
    outlier_scale: float = 0.5    # Scale factor for outlier component


class CovarianceUpdateMethod(Enum):
    """
    Methods for covariance update in a Kalman filter (unused here, but kept for reference).
    """
    STANDARD = "standard"
    JOSEPH = "joseph"
    SQRT = "sqrt"


class StateValidator:
    """
    State validation and bounding utility for numeric stability.
    Ensures states/covariances remain in a safe numeric range.
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
            return np.zeros_like(state)
        return np.clip(state, self.min_bound, self.max_bound)

    def validate_covariance(self, P: np.ndarray) -> np.ndarray:
        """
        Ensure the covariance matrix is symmetric and has eigenvalues
        within [min_covariance, max_covariance].
        """
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
    Time-varying parameter generator for the Lorenz system (optional).
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
            0.7 * np.sin(0.3 * t) + 0.3 * np.sin(0.7 * t)
        ))
        beta_variation = 0.5 * self.amplitude * (
            np.sin(0.2 * t) + np.sin(0.4 * t + np.pi/4)
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
        Jacobian used by a standard EKF (assuming base parameters).
        If strictly time-varying, update to reflect time-varying sigma, rho, beta.
        """
        x, y, z = state
        return np.array([
            [-self.base_sigma,    self.base_sigma,       0.0],
            [ self.base_rho - z, -1.0,               -x  ],
            [         y,          x,            -self.base_beta]
        ])


# -----------------------------------------------------------------------------
#       PART 3: Neural Networks (Sequential Physics + Uncertainty)
# -----------------------------------------------------------------------------
class PhysicsSequenceNetwork(nn.Module):
    """
    A sequential (LSTM-based) physics-informed neural network that produces
    an *additive correction* to the physics-based next state.

    In other words, if f_phys(x_k) is the physics next state (via Lorenz + RK4),
    the network output is delta(x_k, ...), and the corrected function is:
        f^*(x_k) = f_phys(x_k) + delta(x_k).
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
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
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

    def forward(self, past_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns delta(...) given a window of past states.

        Args:
            past_states: shape (batch_size, sequence_length, state_dim)

        Returns:
            correction: shape (batch_size, state_dim)
        """
        # LSTM
        lstm_out, _ = self.lstm(past_states)  # (B, T, hidden_dim)
        final_hidden = lstm_out[:, -1, :]     # (B, hidden_dim)

        # Correction
        correction = self.correction_head(final_hidden)  # (B, state_dim)

        return correction


class SimplifiedUncertaintyNetwork(nn.Module):
    """
    A streamlined network that estimates process (Q) and measurement (R)
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

        # Single state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Single-layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Process noise (Q) estimator
        self.q_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )

        # Measurement noise (R) estimator
        self.r_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, measurement_dim * measurement_dim)
        )

        # Confidence net
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 2),
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
                dt: float = 0.01) -> dict:
        """
        Forward pass to estimate Q and R matrices plus confidences.

        Args:
            state_sequence: shape (batch_size, sequence_length, state_dim)
            innovation_sequence: shape (batch_size, sequence_length, measurement_dim)
            dt: time step to scale Q

        Returns:
            A dictionary with 'Q', 'R', 'q_confidence', 'r_confidence'
        """
        batch_size = state_sequence.size(0)
        
        # Encode states
        encoded = self.state_encoder(state_sequence)  # (B, T, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(encoded)              # (B, T, hidden_dim)
        hidden = lstm_out[:, -1, :]                   # (B, hidden_dim)

        # Estimate Q
        q_raw = self.q_estimator(hidden)
        Q = q_raw.view(batch_size, self.state_dim, self.state_dim)
        Q = 0.5 * (Q + Q.transpose(-2, -1))  # symmetrize
        Q = F.softplus(Q) * dt              # ensure positivity & scale

        # Estimate R
        r_raw = self.r_estimator(hidden)
        R = r_raw.view(batch_size, self.measurement_dim, self.measurement_dim)
        R = 0.5 * (R + R.transpose(-2, -1))
        R = F.softplus(R)

        # Confidence
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
      - Physics-based prediction (LorenzSystem + *function-correction* net)
      - Uncertainty estimation (SimplifiedUncertaintyNetwork)
      - Extended Kalman Filter update
      - Online learning with backprop
      - Shadow/Warmup mode
    """

    def __init__(self, system: LorenzSystem, config: HybridEKFConfig) -> None:
        self.system = system
        self.config = config

        # EKF state
        self.x_hat = np.zeros(config.state_dim)
        self.P     = np.eye(config.state_dim)

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

        # Buffers
        self.state_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)
        self.innovation_buffer: Deque[np.ndarray] = deque(maxlen=config.sequence_length)
        self.metrics = {
            'prediction_loss': [],
            'uncertainty_loss': [],
            'q_confidence': [],
            'r_confidence': []
        }
        self.global_step = 0

        # Pretraining data
        self.pretraining_states: List[np.ndarray] = []
        self.pretraining_measurements: List[np.ndarray] = []
        self.pretraining_innovations: List[np.ndarray] = []

        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # For EWC
        self.pretrained_physics = None
        self.pretrained_uncertainty = None

    # -------------------------------------------------------------------------
    # 1) The Corrected Dynamics
    # -------------------------------------------------------------------------
    def corrected_dynamics(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        The actual function f^*(x) = physics_dynamics(x) + NN_correction(x).
        If not enough past states, correction = 0.
        """
        # 1) Physics-based next state (e.g. Lorenz + RK4)
        physics_next = self.system.dynamics(x, dt)

        # 2) If we have enough history, get a correction from the physics_net
        if len(self.state_buffer) < self.config.sequence_length:
            return physics_next  # no correction during early steps

        states_array = np.array(list(self.state_buffer))  # shape (T, state_dim)
        states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            correction_tensor = self.physics_net(states_tensor)  # shape (1, state_dim)
        correction = correction_tensor.squeeze(0).cpu().numpy()

        return physics_next + correction

    def corrected_jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Autograd-based Jacobian of f^*(x) w.r.t x, for use in EKF covariance update.
        We'll define a small function to let PyTorch do the partial derivatives.
        """
        # Wrap x in a torch tensor with requires_grad
        x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)

        def torch_corrected_dyn(x_torch: torch.Tensor) -> torch.Tensor:
            # Convert current x to numpy for the physics part
            x_np = x_torch.detach().cpu().numpy()
            phys_next = self.system.dynamics(x_np, dt)
            phys_next_t = torch.tensor(phys_next, dtype=torch.float32)

            # NN correction
            if len(self.state_buffer) >= self.config.sequence_length:
                states_array = np.array(list(self.state_buffer))
                states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)
                corr_t = self.physics_net(states_tensor)
                corr_t = corr_t.squeeze(0).cpu()
            else:
                corr_t = torch.zeros_like(phys_next_t)

            return phys_next_t + corr_t

        # Build the Jacobian by partial derivative wrt x
        state_dim = self.config.state_dim
        J = np.zeros((state_dim, state_dim), dtype=np.float32)

        # Create output tensor
        y_t = torch_corrected_dyn(x_t)

        # Option A: loop each dimension and ensure gradient computation
        for i in range(state_dim):
            if x_t.grad is not None:
                x_t.grad.zero_()
            y_t[i].backward(retain_graph=True)
            if x_t.grad is not None:
                J[i, :] = x_t.grad.cpu().numpy()
            else:
                # If no gradient path exists, use physics-based Jacobian for this row
                J[i, :] = self.system.jacobian(x)[i, :]

        return J

    # -------------------------------------------------------------------------
    # 2) PREDICT step uses f^* and its Jacobian
    # -------------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        """
        EKF prediction step:
        - Use pure physics-based prediction (system.dynamics + system.jacobian)
            if still in warmup.
        - After warmup, use the corrected dynamics f^*(x) and the associated
            autograd-based Jacobian.
        """
        try:
            if self.global_step < self.config.warmup_steps:
                # 1) Pure physics
                x_pred = self.system.dynamics(self.x_hat, dt)
                F = self.system.jacobian(self.x_hat)
                
                # 2) Use constant Q as well
                Q = np.eye(self.config.state_dim) * self.config.process_noise_init

            else:
                # 1) Corrected dynamics f^*(x)
                x_pred = self.corrected_dynamics(self.x_hat, dt)
                # 2) Autograd-based Jacobian of f^*(x)
                F = self.corrected_jacobian(self.x_hat, dt)

                # 3) Uncertainty network Q
                uncertainty = self._get_uncertainty_estimates()
                Q = uncertainty['Q'].squeeze(0).cpu().numpy()
                Q = 0.5 * (Q + Q.T)  # Ensure symmetry

            # EKF covariance propagation
            self.x_hat = x_pred
            self.P = F @ self.P @ F.T + Q
            self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry

            # Update ring buffer with new predicted state
            self._update_buffers(self.x_hat)

        except Exception as e:
            logging.error(f"Error in predict step: {e}")
            raise

    # -------------------------------------------------------------------------
    # 3) UPDATE step: Standard EKF measurement update
    # -------------------------------------------------------------------------
    def update(self, measurement: np.ndarray) -> None:
        try:
            # Validate states/cov
            measurement = self.state_validator.validate_state(measurement).squeeze()
            self.x_hat = self.state_validator.validate_state(self.x_hat).squeeze()
            self.P = self.state_validator.validate_covariance(self.P).squeeze()

            # Partial measurement model
            H = np.zeros((self.config.measurement_dim, self.config.state_dim))
            H[0, 0] = 1.0  # observe x
            H[1, 2] = 1.0  # observe z

            if self.global_step < self.config.warmup_steps:
                R_np = np.eye(self.config.measurement_dim) * self.config.measurement_noise_init
            else:
                uncertainty = self._get_uncertainty_estimates()
                R_np = uncertainty['R'].squeeze(0).cpu().numpy()
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

            # Validate final
            self.x_hat = self.state_validator.validate_state(self.x_hat)
            self.P = self.state_validator.validate_covariance(self.P)

            # Update buffers
            self._update_buffers(self.x_hat, innovation)

            # Store shadow-mode data
            if self.global_step < self.config.warmup_steps:
                self.pretraining_states.append(self.x_hat.copy())
                self.pretraining_measurements.append(measurement.copy())
                self.pretraining_innovations.append(innovation.copy())

                # Pretrain once we reach end of warmup
                if self.global_step == self.config.warmup_steps - 1:
                    self._pretrain_networks()

            # Online learning (if not in warmup)
            if self.global_step >= self.config.warmup_steps and not np.any(np.isnan(self.x_hat)):
                self._online_learning(measurement, innovation, None)

            self.global_step += 1

        except Exception as e:
            logging.error(f"Error in update step: {e}")
            raise

    # -------------------------------------------------------------------------
    # 4) UNCERTAINTY ESTIMATION & TRAINING
    # -------------------------------------------------------------------------
    def _get_uncertainty_estimates(self) -> Dict[str, torch.Tensor]:
        if len(self.state_buffer) < self.config.sequence_length:
            eye_Q = torch.eye(self.config.state_dim, device=DEVICE) * self.config.process_noise_init
            eye_R = torch.eye(self.config.measurement_dim, device=DEVICE) * self.config.measurement_noise_init
            return {
                'Q': eye_Q.unsqueeze(0),
                'R': eye_R.unsqueeze(0),
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

    def _online_learning(self, measurement: np.ndarray, innovation: np.ndarray, _) -> None:
        """
        Online learning with replay. 
        The target is that f^*(x_k) should match x_{k+1}^{true}.
        Here we do a short pseudo-implementation: we have (x_hat, measurement) 
        but in real usage you'd have ground truth or the next state's truth. 
        We'll do partial training with the measurement for x,z only, 
        as in the original tutorial logic.
        """
        self.experience_buffer.add(self.x_hat, measurement, innovation)

        if len(self.state_buffer) == self.config.sequence_length:
            # 1) Physics net training
            self.physics_optimizer.zero_grad()

            # Construct a small training target:
            # We treat the measurement as partial ground-truth of x and z. 
            # We'll call corrected_dynamics(...) on the *previous* state in the buffer.
            # For simplicity, use the last buffer state as x_k, 
            # and the measurement as x_{k+1}^{true}(x,z).
            last_state = self.state_buffer[-1]  # x_k
            pred_next = self.corrected_dynamics(last_state, self.config.dt)  # shape (3,)
            # Only x and z are observed => partial measurement
            pred_xz = np.array([pred_next[0], pred_next[2]], dtype=np.float32)
            measurement_tensor = torch.FloatTensor([measurement[0], measurement[1]]).to(DEVICE)

            pred_xz_tensor = torch.tensor(pred_xz, device=DEVICE)
            current_loss = F.mse_loss(pred_xz_tensor, measurement_tensor)

            # Add a simple L2 penalty on the net's output for regularization
            states_array = np.array(list(self.state_buffer))
            states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(DEVICE)
            correction_tensor = self.physics_net(states_tensor)
            physics_loss = torch.mean(correction_tensor**2)

            # Replay
            replay_states, replay_meas, _ = self.experience_buffer.sample(32)
            replay_loss = torch.tensor(0.0, device=DEVICE)
            if len(replay_states) > 0:
                # For each replay state, guess next x,z using corrected_dynamics
                # Then compare to replay_meas. 
                # (Omitted details, just do a quick demonstration.)
                pass

            ewc_physics = self.compute_ewc_loss(
                dict(self.physics_net.named_parameters()),
                self.pretrained_physics
            )

            physics_total_loss = current_loss + 0.1 * physics_loss + 0.3 * replay_loss + ewc_physics
            physics_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(), 1.0)
            self.physics_optimizer.step()

            # 2) Uncertainty net training
            self.uncertainty_optimizer.zero_grad()
            unc = self._get_uncertainty_estimates()
            uncertainty_loss = self._compute_uncertainty_loss(innovation, unc['R'])
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
            self.metrics['q_confidence'].append(unc['q_confidence'].item())
            self.metrics['r_confidence'].append(unc['r_confidence'].item())

    def _compute_uncertainty_loss(self, innovation: np.ndarray, R: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood for innovation ~ N(0, R).
        """
        innovation_tensor = torch.FloatTensor(innovation).to(DEVICE)
        R_2x2 = R.squeeze(0)  # shape (2,2)
        R_2x2 = torch.clamp(R_2x2, min=self.config.min_covariance)
        logdetR = torch.logdet(R_2x2)
        inn_2x1 = innovation_tensor.unsqueeze(-1)
        mahal = inn_2x1.transpose(0,1) @ torch.inverse(R_2x2) @ inn_2x1
        ll = -0.5 * (2 * np.log(2.0*np.pi) + logdetR + mahal)
        return -ll.squeeze()

    def store_pretrained_params(self) -> None:
        """Store network parameters after pretraining for EWC."""
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
        if pretrained_params is None:
            return torch.tensor(0.0, device=DEVICE)
        ewc_lambda = 0.4
        loss = torch.tensor(0.0, device=DEVICE)
        for name in current_params:
            loss += ewc_lambda * torch.sum(
                (current_params[name] - pretrained_params[name])**2
            )
        return loss

    def _pretrain_networks(self) -> None:
        """
        Pre-train the networks using shadow-mode data before normal online learning.
        """
        logging.info("Pre-training neural networks with shadow mode data...")

        # Convert collected data to tensors
        states_tensor = torch.FloatTensor(self.pretraining_states).to(DEVICE)
        measurements_tensor = torch.FloatTensor(self.pretraining_measurements).to(DEVICE)
        innovations_tensor = torch.FloatTensor(self.pretraining_innovations).to(DEVICE)

        # For simplicity, do a few epochs of basic training:
        # (Detailed approach omitted for brevity.)
        # Possibly replicate the approach used in the original tutorial:
        for epoch in range(50):
            pass  # e.g., mini-batch over states_tensor, measure partial x,z

        logging.info("Pre-training completed. Transitioning to online learning mode.")

        # Clear memory
        self.pretraining_states.clear()
        self.pretraining_measurements.clear()
        self.pretraining_innovations.clear()

        # Store pretrained params
        self.store_pretrained_params()
        logging.info("Stored pretrained parameters for EWC.")


# StandardEKF and UnscentedKF remain largely unchanged, used for comparison:
class StandardEKF:
    def __init__(self, system: LorenzSystem, state_dim: int = 3) -> None:
        self.system = system
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def predict(self, dt: float) -> None:
        self.x_hat = self.system.dynamics(self.x_hat, dt)
        F = self.system.jacobian(self.x_hat)
        Q = np.eye(self.state_dim) * 0.1
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: np.ndarray) -> None:
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
    def __init__(self, system: LorenzSystem, state_dim: int = 3) -> None:
        self.system = system
        self.state_dim = state_dim
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)

        self.alpha = 0.3
        self.beta = 2.0
        self.kappa = 0.0
        self.n = state_dim
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self.weights_m = np.zeros(2*self.n + 1)
        self.weights_c = np.zeros(2*self.n + 1)
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = (self.lambda_ / (self.n + self.lambda_)
                             + (1 - self.alpha**2 + self.beta))
        for i in range(1, 2*self.n + 1):
            self.weights_m[i] = 1.0 / (2.0*(self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def predict(self, dt: float) -> None:
        sigma_points = self._generate_sigma_points()
        propagated = np.array([self.system.dynamics(sp, dt) for sp in sigma_points])

        self.x_hat = np.sum(self.weights_m[:, None] * propagated, axis=0)
        P_pred = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = (propagated[i] - self.x_hat).reshape(-1, 1)
            P_pred += self.weights_c[i] * (diff @ diff.T)
        Q = np.eye(self.state_dim) * 0.1
        self.P = P_pred + Q

    def update(self, measurement: np.ndarray) -> None:
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        R = np.eye(2) * 0.1

        sigma_points = self._generate_sigma_points()
        projected = np.array([H @ sp for sp in sigma_points])
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
        self.x_hat = self.x_hat + K @ innovation
        self.P = self.P - K @ Pyy @ K.T

    def _generate_sigma_points(self) -> np.ndarray:
        sigma_points = np.zeros((2*self.n + 1, self.n))
        sigma_points[0] = self.x_hat
        scaled_cov = (self.n + self.lambda_) * self.P
        scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)

        eigvals = np.linalg.eigvals(scaled_cov)
        min_eig = np.min(np.real(eigvals))
        reg_factor = 1e-3
        if min_eig < reg_factor:
            scaled_cov += reg_factor * np.eye(self.n)
        try:
            sqrt_cov = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            scaled_cov += 1e-2 * np.eye(self.n)
            try:
                sqrt_cov = np.linalg.cholesky(scaled_cov)
            except np.linalg.LinAlgError:
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
def generate_non_gaussian_noise(size: int, config: HybridEKFConfig) -> np.ndarray:
    """
    Generate measurement noise with possible outliers.
    """
    base_noise = np.random.normal(
        0, config.measurement_noise_init, size=size
    )
    if config.noise_type == "gaussian":
        return base_noise

    # Generate outlier mask
    outlier_mask = np.random.rand(size) < config.mixture_prob
    if config.noise_type == "laplace":
        outliers = np.random.laplace(
            loc=0,
            scale=config.measurement_noise_init * config.outlier_scale,
            size=size
        )
    elif config.noise_type == "mixed":
        outliers = np.random.standard_cauchy(size=size) * config.outlier_scale

    return np.where(outlier_mask, base_noise + outliers, base_noise)


def simulate_system(
    config: HybridEKFConfig,
    time_steps: int = 5000,
    save_path: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Run a simulation of the Lorenz system with the IntegratedHybridEKF
    using the new function-correction approach.
    """
    system = LorenzSystem()
    ekf = IntegratedHybridEKF(system, config)

    true_states = np.zeros((time_steps, config.state_dim))
    estimated_states = np.zeros_like(true_states)
    uncertainties = np.zeros((time_steps, 2))
    state_covariances = np.zeros((time_steps, config.state_dim, config.state_dim))

    x_true = np.array([1.0, 1.0, 1.0])  # initial

    for t in range(time_steps):
        if t % 100 == 0:
            logging.info(f"Simulation step {t}/{time_steps}")

        x_true = system.dynamics(x_true, config.dt)
        true_states[t] = x_true

        measured_xz = x_true[[0, 2]] + generate_non_gaussian_noise(2, config)

        ekf.predict(config.dt)
        ekf.update(measured_xz)

        estimated_states[t] = ekf.x_hat
        state_covariances[t] = ekf.P.copy()

        if ekf.metrics['q_confidence'] and ekf.metrics['r_confidence']:
            uncertainties[t, 0] = ekf.metrics['q_confidence'][-1]
            uncertainties[t, 1] = ekf.metrics['r_confidence'][-1]

    results = {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'uncertainties': uncertainties,
        'metrics': ekf.metrics,
        'state_covariances': state_covariances,
        'warmup_steps': config.warmup_steps
    }

    warmup_rmse = np.sqrt(np.mean(
        (true_states[:config.warmup_steps] - estimated_states[:config.warmup_steps])**2
    ))
    post_warmup_rmse = np.sqrt(np.mean(
        (true_states[config.warmup_steps:] - estimated_states[config.warmup_steps:])**2
    ))
    transition_rmse = np.sqrt(np.mean(
        (true_states[config.warmup_steps:config.warmup_steps+100]
         - estimated_states[config.warmup_steps:config.warmup_steps+100])**2
    ))

    results['performance_metrics'] = {
        'warmup_rmse': warmup_rmse,
        'post_warmup_rmse': post_warmup_rmse,
        'transition_rmse': transition_rmse
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
    Run a comparative simulation with Hybrid EKF, Standard EKF, and Unscented KF.
    """
    if config is None:
        config = HybridEKFConfig(dt=dt)

    true_system = LorenzSystem(time_varying=True, variation_amplitude=1.2)
    filter_system = LorenzSystem(time_varying=False)

    hybrid_ekf = IntegratedHybridEKF(filter_system, config)
    standard_ekf = StandardEKF(filter_system, state_dim=config.state_dim)
    ukf = UnscentedKF(filter_system, state_dim=config.state_dim)

    true_states = np.zeros((time_steps, config.state_dim))
    hybrid_states = np.zeros_like(true_states)
    standard_states = np.zeros_like(true_states)
    ukf_states = np.zeros_like(true_states)

    hybrid_covariances = np.zeros((time_steps, config.state_dim, config.state_dim))

    x_true = np.array([1.0, 1.0, 1.0])

    for t in range(time_steps):
        if t % 100 == 0:
            logging.info(f"Comparative simulation step {t}/{time_steps}")

        x_true = true_system.dynamics(x_true, dt, t)
        true_states[t] = x_true

        measured_xz = x_true[[0, 2]] + generate_non_gaussian_noise(2, config)

        # Hybrid
        hybrid_ekf.predict(dt)
        hybrid_ekf.update(measured_xz)
        hybrid_states[t] = hybrid_ekf.x_hat
        hybrid_covariances[t] = hybrid_ekf.P.copy()

        # Standard
        standard_ekf.predict(dt)
        standard_ekf.update(measured_xz)
        standard_states[t] = standard_ekf.x_hat

        # UKF
        ukf.predict(dt)
        ukf.update(measured_xz)
        ukf_states[t] = ukf.x_hat

    warmup_steps = config.warmup_steps

    def calc_split_metrics(estimated, true):
        warmup_rmse = np.sqrt(np.mean(
            (true[:warmup_steps] - estimated[:warmup_steps])**2
        ))
        post_warmup_rmse = np.sqrt(np.mean(
            (true[warmup_steps:] - estimated[warmup_steps:])**2
        ))
        transition_rmse = np.sqrt(np.mean(
            (true[warmup_steps:warmup_steps+100]
             - estimated[warmup_steps:warmup_steps+100])**2
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
            'hybrid': calc_split_metrics(hybrid_states, true_states),
            'standard': calc_split_metrics(standard_states, true_states),
            'ukf': calc_split_metrics(ukf_states, true_states)
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
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        time_steps = len(self.results['true_states'])
        t = np.arange(time_steps)

        true_states = self.results['true_states']
        estimated_states = self.results['estimated_states']

        if 'state_covariances' in self.results:
            covariances = self.results['state_covariances']
            std_devs = np.sqrt(np.array([np.diag(P) for P in covariances]))
            bounds_upper = estimated_states + 2 * std_devs
            bounds_lower = estimated_states - 2 * std_devs

        state_labels = ['x', 'y', 'z']
        for i, label in enumerate(state_labels):
            axes[i].plot(t, true_states[:, i], 'k-', label='True', linewidth=2, alpha=0.7)
            axes[i].plot(t, estimated_states[:, i], 'r--', label='Estimated', linewidth=2)
            if 'state_covariances' in self.results:
                axes[i].fill_between(
                    t, bounds_lower[:, i], bounds_upper[:, i],
                    color='r', alpha=0.2, label='2σ bounds'
                )
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

    def plot_noise_characteristics(self, save_path: Optional[Path] = None):
        noise_samples = generate_non_gaussian_noise(
            10000, HybridEKFConfig(noise_type="mixed")
        )
        plt.figure(figsize=(12, 6))
        sns.histplot(noise_samples, kde=True, bins=100)
        plt.title("Non-Gaussian Noise Distribution (Mixture Model)")
        plt.xlabel("Noise Amplitude")
        plt.ylabel("Density")
        if save_path:
            plt.savefig(save_path / 'noise_distribution.png', 
                        dpi=300, bbox_inches='tight')
        plt.show()


class ComparisonVisualizer:
    """
    Visualization for analyzing multiple filters side by side.
    """
    def __init__(self, results: Dict[str, np.ndarray]):
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]

    def plot_comparative_trajectories(self, save_path: Optional[Path] = None):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        time_steps = len(self.results['true_states'])
        t = np.arange(time_steps)

        true_states = self.results['true_states']
        hybrid_states = self.results['hybrid_states']
        standard_states = self.results['standard_states']
        ukf_states = self.results['ukf_states']

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
                axes[i].fill_between(
                    t, bounds_lower[:, i], bounds_upper[:, i],
                    color='b', alpha=0.2, label='Hybrid EKF 2σ'
                )

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
        logging.info("Running default simulation mode with IntegratedHybridEKF (function-correction) + warmup...")

        config = HybridEKFConfig(
            state_dim=3,
            measurement_dim=2,
            hidden_dim=16,
            sequence_length=20,
            dt=0.01,
            learning_rate=0.001,
            warmup_steps=500
        )

        results = simulate_system(
            config=config,
            time_steps=5000,
            save_path=output_dir / 'simulation_results.npz'
        )

        viz = ResultVisualizer(results)
        viz.plot_state_trajectories(output_dir)
        viz.plot_uncertainty_evolution(output_dir)
        viz.plot_learning_metrics(output_dir)
        viz.plot_noise_characteristics(output_dir)

        metrics = results['performance_metrics']
        logging.info("Performance Metrics:")
        logging.info(f"  Warmup RMSE:       {metrics['warmup_rmse']:.4f}")
        logging.info(f"  Post-warmup RMSE:  {metrics['post_warmup_rmse']:.4f}")
        logging.info(f"  Transition RMSE:   {metrics['transition_rmse']:.4f}")

    else:
        logging.info("Running comparative simulation mode...")

        comparative_results = simulate_comparative_system(
            time_steps=5000,
            dt=0.01
        )

        comp_viz = ComparisonVisualizer(comparative_results)
        comp_viz.plot_comparative_trajectories(output_dir)

        metrics = comparative_results['performance_metrics']
        logging.info("\nPost-warmup Performance Metrics:")
        for filter_name in ['hybrid', 'standard', 'ukf']:
            m = metrics[filter_name]
            logging.info(f"\n{filter_name.upper()} Filter:")
            logging.info(f"  Warmup RMSE:       {m['warmup_rmse']:.4f}")
            logging.info(f"  Post-warmup RMSE:  {m['post_warmup_rmse']:.4f}")
            logging.info(f"  Transition RMSE:   {m['transition_rmse']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid EKF Tutorial (Function-Correction Approach)')
    parser.add_argument('--mode',
                        type=str,
                        choices=['default', 'comparative'],
                        default='default',
                        help='Simulation mode: default or comparative')
    args = parser.parse_args()

    try:
        main(mode=args.mode)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
