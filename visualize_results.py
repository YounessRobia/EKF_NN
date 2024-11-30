import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Tuple, Optional
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

class ResultsVisualizer:
    """Visualization tools for Hybrid EKF-DL results analysis."""
    
    def __init__(self, results_path: Path):
        """Initialize visualizer with results file path."""
        self.results = np.load(results_path)
        self.true_states = self.results['true_states']
        self.estimated_states = self.results['estimated_states']
        self.errors = self.results['estimation_errors']
        self.time_steps = np.arange(len(self.true_states))
        
        #plt.style.use('seaborn')  # Changed from 'seaborn-darkgrid'
        #sns.set_theme(style="darkgrid")  # Added explicit darkgrid theme
        sns.set_palette("husl")
    
    def plot_state_trajectories(self, save_path: Optional[Path] = None) -> None:
        """Plot true vs estimated state trajectories."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        state_labels = ['x', 'y', 'z']
        
        for i, (ax, label) in enumerate(zip(axes, state_labels)):
            ax.plot(self.time_steps, self.true_states[:, i], 
                   label=f'True {label}', linewidth=2)
            ax.plot(self.time_steps, self.estimated_states[:, i], 
                   label=f'Estimated {label}', linestyle='--')
            ax.fill_between(self.time_steps,
                          self.estimated_states[:, i] - 2*np.std(self.errors[:, i]),
                          self.estimated_states[:, i] + 2*np.std(self.errors[:, i]),
                          alpha=0.2)
            ax.set_ylabel(f'State {label}')
            ax.legend()
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('State Trajectories with 2σ Uncertainty Bounds')
        
        if save_path:
            plt.savefig(save_path / 'state_trajectories.png', dpi=300, bbox_inches='tight')
    
    def plot_phase_space(self, save_path: Optional[Path] = None) -> None:
        """Create 3D phase space plot."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories
        ax.plot3D(self.true_states[:, 0], self.true_states[:, 1], self.true_states[:, 2],
                 'b-', label='True', linewidth=2)
        ax.plot3D(self.estimated_states[:, 0], self.estimated_states[:, 1], 
                 self.estimated_states[:, 2], 'r--', label='Estimated')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Phase Space Trajectory')
        
        if save_path:
            plt.savefig(save_path / 'phase_space.png', dpi=300, bbox_inches='tight')
    
    def plot_error_analysis(self, save_path: Optional[Path] = None) -> None:
        """Create comprehensive error analysis plots."""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # RMSE over time
        ax1 = fig.add_subplot(gs[0, 0])
        rmse = np.sqrt(np.mean(self.errors**2, axis=1))
        ax1.plot(self.time_steps, rmse)
        ax1.set_title('RMSE over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('RMSE')
        
        # Error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.violinplot(data=self.errors, ax=ax2)
        ax2.set_title('Error Distribution')
        ax2.set_xticks(range(len(['x', 'y', 'z'])))  # Ensure ticks are set
        ax2.set_xticklabels(['x', 'y', 'z'])
        
        # Error autocorrelation
        ax3 = fig.add_subplot(gs[1, :])
        max_lag = 50
        for i, label in enumerate(['x', 'y', 'z']):
            autocorr = [1.] + [np.corrcoef(self.errors[:-lag, i], 
                                        self.errors[lag:, i])[0, 1]
                            for lag in range(1, max_lag) if lag < len(self.errors)]
            ax3.plot(range(len(autocorr)), autocorr, label=f'State {label}')
        ax3.set_title('Error Autocorrelation')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Autocorrelation')
        ax3.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / 'error_analysis.png', dpi=300, bbox_inches='tight')
    
    def plot_performance_metrics(self, save_path: Optional[Path] = None) -> None:
        """Plot various performance metrics."""
        window_size = 50
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rolling RMSE
        rolling_rmse = np.array([np.sqrt(np.mean(self.errors[i:i+window_size]**2))
                                for i in range(len(self.errors)-window_size)])
        axes[0,0].plot(rolling_rmse)
        axes[0,0].set_title(f'Rolling RMSE (window={window_size})')
        
        # Error correlation matrix
        sns.heatmap(np.corrcoef(self.errors.T), 
                   annot=True, cmap='coolwarm', center=0,
                   ax=axes[0,1])
        axes[0,1].set_title('Error Correlation Matrix')
        
        # Innovation magnitude
        innovation = np.diff(self.estimated_states, axis=0)
        axes[1,0].plot(np.linalg.norm(innovation, axis=1))
        axes[1,0].set_title('Innovation Magnitude')
        
        # Cumulative error
        cumulative_error = np.cumsum(np.abs(self.errors), axis=0)
        for i, label in enumerate(['x', 'y', 'z']):
            axes[1,1].plot(cumulative_error[:, i], label=f'State {label}')
        axes[1,1].legend()
        axes[1,1].set_title('Cumulative Absolute Error')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    
    def generate_report(self, save_path: Path) -> None:
        """Generate comprehensive visualization report."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.plot_state_trajectories(save_path)
        self.plot_phase_space(save_path)
        self.plot_error_analysis(save_path)
        self.plot_performance_metrics(save_path)
        
        # Save summary statistics
        summary = {
            'mean_rmse': np.sqrt(np.mean(self.errors**2)),
            'final_rmse': np.sqrt(np.mean(self.errors[-100:]**2)),
            'max_error': np.max(np.abs(self.errors)),
            'error_std': np.std(self.errors, axis=0)
        }
        
        with open(save_path / 'summary_stats.txt', 'w') as f:
            for key, value in summary.items():
                f.write(f'{key}: {value}\n')

class ComparisonVisualizer:
    """Visualization tools for comparing Hybrid EKF-DL with Standard EKF."""
    
    def __init__(self, hybrid_results_path: Path, standard_results_path: Path):
        """Initialize visualizer with both results."""
        self.hybrid_results = np.load(hybrid_results_path)
        self.standard_results = np.load(standard_results_path)
        self.time_steps = np.arange(len(self.hybrid_results['true_states']))
        
        sns.set_palette("husl")
    
    def plot_comparison_trajectories(self, save_path: Optional[Path] = None) -> None:
        """Plot state trajectories comparing both filters."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        state_labels = ['x', 'y', 'z']
        
        for i, (ax, label) in enumerate(zip(axes, state_labels)):
            # True trajectory
            ax.plot(self.time_steps, self.hybrid_results['true_states'][:, i],
                   'k-', label='True', linewidth=2)
            
            # Hybrid EKF estimate
            ax.plot(self.time_steps, self.hybrid_results['estimated_states'][:, i],
                   'b--', label='Hybrid EKF')
            
            # Standard EKF estimate
            ax.plot(self.time_steps, self.standard_results['estimated_states'][:, i],
                   'r--', label='Standard EKF')
            
            ax.set_ylabel(f'State {label}')
            ax.legend()
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('State Trajectory Comparison')
        
        if save_path:
            plt.savefig(save_path / 'comparison_trajectories.png', dpi=300, bbox_inches='tight')
    
    def plot_error_comparison(self, save_path: Optional[Path] = None) -> None:
        """Plot error metrics comparison."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # RMSE comparison
        hybrid_rmse = np.sqrt(np.mean(self.hybrid_results['estimation_errors']**2, axis=1))
        standard_rmse = np.sqrt(np.mean(self.standard_results['estimation_errors']**2, axis=1))
        
        axes[0].plot(self.time_steps, hybrid_rmse, 'b-', label='Hybrid EKF')
        axes[0].plot(self.time_steps, standard_rmse, 'r-', label='Standard EKF')
        axes[0].set_ylabel('RMSE')
        axes[0].legend()
        axes[0].set_title('RMSE Comparison')
        
        # Error distribution comparison
        axes[1].violinplot([hybrid_rmse, standard_rmse], positions=[1, 2])
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(['Hybrid EKF', 'Standard EKF'])
        axes[1].set_ylabel('Error Distribution')
        axes[1].set_title('Error Distribution Comparison')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / 'error_comparison.png', dpi=300, bbox_inches='tight')
    
    def generate_comparison_report(self, save_path: Path) -> None:
        """Generate comprehensive comparison report."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.plot_comparison_trajectories(save_path)
        self.plot_error_comparison(save_path)
        
        # Compute summary statistics
        hybrid_rmse = np.sqrt(np.mean(self.hybrid_results['estimation_errors']**2))
        standard_rmse = np.sqrt(np.mean(self.standard_results['estimation_errors']**2))
        
        with open(save_path / 'comparison_summary.txt', 'w') as f:
            f.write(f'Hybrid EKF RMSE: {hybrid_rmse:.4f}\n')
            f.write(f'Standard EKF RMSE: {standard_rmse:.4f}\n')
            f.write(f'Improvement: {((standard_rmse - hybrid_rmse) / standard_rmse * 100):.2f}%\n')

def main():
    """Main function to generate visualization report."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        hybrid_results_path = Path('hybrid_results.npz')
        standard_results_path = Path('standard_results.npz')
        output_path = Path('comparison_results')
        
        comparison_visualizer = ComparisonVisualizer(hybrid_results_path, standard_results_path)
        comparison_visualizer.generate_comparison_report(output_path)
        
        logging.info(f'Comparison report generated in {output_path}')
        
    except Exception as e:
        logging.error(f'Visualization failed: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    main() 