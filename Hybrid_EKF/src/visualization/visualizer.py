"""Visualization tools for analyzing Kalman filter performance."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path
import logging

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
            
            if 'hybrid_states' in self.results:
                axes[i].plot(self.results['hybrid_states'][:, i],
                           'b--', label='Hybrid EKF')
                # Add uncertainty bounds for hybrid EKF
                std = self.results['hybrid_std'][:, i]
                axes[i].fill_between(
                    range(len(self.results['hybrid_states'])),
                    self.results['hybrid_states'][:, i] - 2*std,
                    self.results['hybrid_states'][:, i] + 2*std,
                    alpha=0.2, color='b'
                )
            
            if 'standard_states' in self.results:
                axes[i].plot(self.results['standard_states'][:, i],
                           'r:', label='Standard EKF')
            
            if 'ukf_states' in self.results:
                axes[i].plot(self.results['ukf_states'][:, i],
                           'g-.', label='UKF')
                # Add uncertainty bounds for UKF
                if 'ukf_std' in self.results:
                    std = self.results['ukf_std'][:, i]
                    axes[i].fill_between(
                        range(len(self.results['ukf_states'])),
                        self.results['ukf_states'][:, i] - 2*std,
                        self.results['ukf_states'][:, i] + 2*std,
                        alpha=0.1, color='g'
                    )
            
            axes[i].set_ylabel(f'State {label}')
            axes[i].legend()
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('State Trajectories with Uncertainty Bounds')
        
        if save_path:
            plt.savefig(save_path / 'state_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_phase_space(self, save_path: Optional[Path] = None):
        """Plot 3D phase space comparison."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot true trajectory
        ax.plot3D(
            self.results['true_states'][:, 0],
            self.results['true_states'][:, 1],
            self.results['true_states'][:, 2],
            'k-', label='True', linewidth=1
        )
        
        # Plot estimated trajectories
        if 'hybrid_states' in self.results:
            ax.plot3D(
                self.results['hybrid_states'][:, 0],
                self.results['hybrid_states'][:, 1],
                self.results['hybrid_states'][:, 2],
                'b--', label='Hybrid EKF', linewidth=1
            )
        
        if 'standard_states' in self.results:
            ax.plot3D(
                self.results['standard_states'][:, 0],
                self.results['standard_states'][:, 1],
                self.results['standard_states'][:, 2],
                'r:', label='Standard EKF', linewidth=1
            )
        
        if 'ukf_states' in self.results:
            ax.plot3D(
                self.results['ukf_states'][:, 0],
                self.results['ukf_states'][:, 1],
                self.results['ukf_states'][:, 2],
                'g-.', label='UKF', linewidth=1
            )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path / 'phase_space.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_metrics(self, save_path: Optional[Path] = None):
        """Plot error metrics and performance comparison."""
        # Compute RMSEs
        rmse_dict = {}
        for filter_name in ['hybrid', 'standard', 'ukf']:
            if f'{filter_name}_states' in self.results:
                rmse = np.sqrt(np.mean(
                    (self.results['true_states'] - self.results[f'{filter_name}_states'])**2,
                    axis=1
                ))
                rmse_dict[filter_name] = rmse
        
        # Plot RMSE evolution
        plt.figure(figsize=(12, 6))
        for name, rmse in rmse_dict.items():
            plt.plot(rmse, label=f'{name.capitalize()} EKF')
        
        plt.xlabel('Time Step')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error Evolution')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path / 'error_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_uncertainty_evolution(self, save_path: Optional[Path] = None):
        """Plot evolution of uncertainty estimates."""
        if 'hybrid_std' not in self.results:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        for i, label in enumerate(['x', 'y', 'z']):
            axes[i].plot(self.results['hybrid_std'][:, i],
                        'b-', label='Hybrid EKF')
            
            if 'ukf_std' in self.results:
                axes[i].plot(self.results['ukf_std'][:, i],
                           'g--', label='UKF')
            
            axes[i].set_ylabel(f'State {label} Std')
            axes[i].legend()
            axes[i].grid(True)
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('Evolution of State Uncertainty Estimates')
        
        if save_path:
            plt.savefig(save_path / 'uncertainty_evolution.png', dpi=300, bbox_inches='tight')
        plt.show() 