"""Simulation utilities for Kalman filter comparison."""

import numpy as np
from typing import Dict, Optional, Type
import logging
from pathlib import Path

from ..config import HybridEKFConfig
from ..models.lorenz import LorenzSystem
from ..filters.base import BaseFilter
from ..filters.hybrid_ekf import HybridEKF
from ..filters.standard_ekf import StandardEKF
from ..filters.ukf import UKF

def generate_non_gaussian_noise(size: int, noise_type: str = 'mixture') -> np.ndarray:
    """Generate non-Gaussian noise for testing filter robustness."""
    if noise_type == 'mixture':
        noise = np.where(
            np.random.rand(size) > 0.7,
            np.random.normal(0, 2.0, size),
            np.random.normal(0, 0.5, size)
        )
    elif noise_type == 'student_t':
        noise = np.random.standard_t(df=3, size=size)
    elif noise_type == 'skewed':
        noise = np.random.normal(0, 1.0, size) + np.abs(np.random.normal(0, 0.5, size))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
        
    return noise

class LorenzSimulator:
    """Simulator for comparing different Kalman filter implementations."""
    
    def __init__(self, config: HybridEKFConfig):
        self.config = config
        self.true_system = LorenzSystem(time_varying=True, variation_amplitude=0.9)
        self.filter_system = LorenzSystem(time_varying=False)
    
    def simulate_single_filter(self, 
                             filter_class: Type[BaseFilter],
                             time_steps: int,
                             noise_type: str = 'mixture',
                             save_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Run simulation with a single filter type."""
        filter_instance = filter_class(self.filter_system, self.config)
        
        results = {
            'true_states': np.zeros((time_steps, self.config.state_dim)),
            'estimated_states': np.zeros((time_steps, self.config.state_dim)),
            'measurements': np.zeros((time_steps, self.config.state_dim))
        }
        
        true_state = np.array([1.0, 1.0, 1.0])
        t = 0.0
        
        for step in range(time_steps):
            if step % 100 == 0:
                logging.info(f"Simulation step {step}/{time_steps}")
            
            # True system evolution
            true_state = self.true_system.dynamics(true_state, self.config.dt, t)
            results['true_states'][step] = true_state
            
            # Generate noisy measurement
            noise = generate_non_gaussian_noise(self.config.state_dim, noise_type)
            measurement = true_state + noise
            results['measurements'][step] = measurement
            
            # Filter update
            filter_instance.predict(self.config.dt)
            filter_instance.update(measurement)
            
            results['estimated_states'][step] = filter_instance.get_state()
            t += self.config.dt
        
        if save_path:
            np.savez(save_path, **results)
        
        return results
    
    def simulate_comparative(self,
                           time_steps: int,
                           noise_type: str = 'mixture',
                           save_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Run simulation comparing all filter implementations."""
        # Initialize filters
        hybrid_filter = HybridEKF(self.filter_system, self.config)
        standard_filter = StandardEKF(self.filter_system)
        ukf_filter = UKF(self.filter_system)
        
        results = {
            'true_states': np.zeros((time_steps, self.config.state_dim)),
            'hybrid_states': np.zeros((time_steps, self.config.state_dim)),
            'standard_states': np.zeros((time_steps, self.config.state_dim)),
            'ukf_states': np.zeros((time_steps, self.config.state_dim)),
            'measurements': np.zeros((time_steps, self.config.state_dim)),
            'hybrid_std': np.zeros((time_steps, self.config.state_dim)),
            'ukf_std': np.zeros((time_steps, self.config.state_dim))
        }
        
        true_state = np.array([1.0, 1.0, 1.0])
        t = 0.0
        
        for step in range(time_steps):
            if step % 100 == 0:
                logging.info(f"Comparative simulation step {step}/{time_steps}")
            
            # True system evolution
            true_state = self.true_system.dynamics(true_state, self.config.dt, t)
            results['true_states'][step] = true_state
            
            # Generate measurement
            noise = generate_non_gaussian_noise(self.config.state_dim, noise_type)
            measurement = true_state + noise
            results['measurements'][step] = measurement
            
            # Update all filters
            hybrid_filter.predict(self.config.dt)
            hybrid_filter.update(measurement)
            
            standard_filter.predict(self.config.dt)
            standard_filter.update(measurement)
            
            ukf_filter.predict(self.config.dt)
            ukf_filter.update(measurement)
            
            # Store results
            results['hybrid_states'][step] = hybrid_filter.get_state()
            results['standard_states'][step] = standard_filter.get_state()
            results['ukf_states'][step] = ukf_filter.get_state()
            results['hybrid_std'][step] = np.sqrt(np.diag(hybrid_filter.get_covariance()))
            results['ukf_std'][step] = np.sqrt(np.diag(ukf_filter.get_covariance()))
            
            t += self.config.dt
        
        if save_path:
            np.savez(save_path, **results)
        
        return results 