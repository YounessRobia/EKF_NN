"""Unscented Kalman Filter implementation."""

import numpy as np
from typing import Optional, Tuple
import logging
from .base import BaseFilter
from ..models.lorenz import LorenzSystem

class UKF(BaseFilter):
    """Unscented Kalman Filter with enhanced stability."""
    
    def __init__(self, system: LorenzSystem, state_dim: int = 3):
        super().__init__(state_dim)
        self.system = system
        
        # UKF parameters
        self.alpha = 0.3  # Primary scaling
        self.beta = 2.0   # Secondary scaling
        self.kappa = 0.0  # Tertiary scaling
        
        # Derived parameters
        self.lambda_ = self.alpha**2 * (state_dim + self.kappa) - state_dim
        self._compute_weights()
        
        # Numerical stability
        self.min_cov_eigenval = 1e-10
        self.regularization_noise = 1e-8
    
    def _compute_weights(self) -> None:
        """Compute UKF weights."""
        n = self.state_dim
        self.weights_m = np.zeros(2 * n + 1)
        self.weights_c = np.zeros(2 * n + 1)
        
        self.weights_m[0] = self.lambda_ / (n + self.lambda_)
        self.weights_c[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * n + 1):
            self.weights_m[i] = 1.0 / (2 * (n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]
    
    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points using current state estimate."""
        n = self.state_dim
        sigma_points = np.zeros((2 * n + 1, n))
        
        # Mean point
        sigma_points[0] = self.x_hat
        
        try:
            # Compute scaled square root of covariance
            scaled_cov = (n + self.lambda_) * self.P
            S = np.linalg.cholesky(scaled_cov + self.regularization_noise * np.eye(n))
            
            # Generate remaining points
            for i in range(n):
                sigma_points[i + 1] = self.x_hat + S[i]
                sigma_points[n + i + 1] = self.x_hat - S[i]
                
        except np.linalg.LinAlgError:
            logging.warning("Cholesky decomposition failed, using regularized matrix")
            sigma_points = np.tile(self.x_hat, (2 * n + 1, 1))
            
        return sigma_points
    
    def predict(self, dt: float) -> None:
        """Prediction step using sigma points."""
        try:
            # Generate and propagate sigma points
            sigma_points = self._generate_sigma_points()
            propagated_points = np.array([
                self.system.dynamics(point, dt) for point in sigma_points
            ])
            
            # Calculate predicted mean
            self.x_hat = np.sum(self.weights_m[:, None] * propagated_points, axis=0)
            
            # Calculate predicted covariance
            self.P = np.zeros((self.state_dim, self.state_dim))
            for i in range(len(sigma_points)):
                diff = propagated_points[i] - self.x_hat
                self.P += self.weights_c[i] * np.outer(diff, diff)
            
            # Add process noise
            Q = np.eye(self.state_dim) * 0.1
            self.P += Q
            
            self._validate_matrices()
            
        except Exception as e:
            logging.error(f"Error in UKF prediction: {str(e)}")
            raise
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step with measurement."""
        try:
            H = np.zeros((2, self.state_dim))
            H[0,0] = 1.0  # Observe x
            H[1,2] = 1.0  # Observe z
            
            sigma_points = self._generate_sigma_points()
            projected_points = np.array([H @ point for point in sigma_points])
            
            # Calculate measurement mean
            y_mean = np.sum(self.weights_m[:, None] * projected_points, axis=0)
            
            # Calculate covariances
            Pyy = np.zeros((2, 2))
            Pxy = np.zeros((self.state_dim, 2))
            
            for i in range(len(sigma_points)):
                diff_y = projected_points[i] - y_mean
                diff_x = sigma_points[i] - self.x_hat
                Pyy += self.weights_c[i] * np.outer(diff_y, diff_y)
                Pxy += self.weights_c[i] * np.outer(diff_x, diff_y)
            
            # Add measurement noise
            R = np.eye(2) * 0.1
            Pyy += R
            
            # Kalman gain computation
            try:
                K = Pxy @ np.linalg.inv(Pyy)
            except np.linalg.LinAlgError:
                K = Pxy @ np.linalg.pinv(Pyy)
            
            # State update
            innovation = np.array([measurement[0], measurement[2]]) - y_mean
            self.x_hat = self.x_hat + K @ innovation
            
            # Covariance update (Joseph form)
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
            
            self._validate_matrices()
            
        except Exception as e:
            logging.error(f"Error in UKF update: {str(e)}")
            raise 