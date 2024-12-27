"""Example script for comparing different Kalman filter implementations."""

import numpy as np
import torch
import logging
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import HybridEKFConfig
from src.simulation.simulator import LorenzSimulator
from src.visualization.visualizer import ResultVisualizer

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = HybridEKFConfig(
        state_dim=3,
        measurement_dim=2,
        sequence_length=20,
        dt=0.01,
        learning_rate=0.001
    )
    
    # Initialize simulator
    simulator = LorenzSimulator(config)
    
    # Run comparative simulation
    logging.info("Starting comparative simulation...")
    results = simulator.simulate_comparative(
        time_steps=10000,
        noise_type='skewed',
        save_path=output_dir / 'simulation_results.npz'
    )
    
    # Create visualizer
    visualizer = ResultVisualizer(results)
    
    # Generate plots
    logging.info("Generating visualization plots...")
    visualizer.plot_state_trajectories(output_dir)
    visualizer.plot_phase_space(output_dir)
    visualizer.plot_error_metrics(output_dir)
    visualizer.plot_uncertainty_evolution(output_dir)
    
    # Compute and display final metrics
    logging.info("\nFinal Performance Metrics:")
    for filter_name in ['hybrid', 'standard', 'ukf']:
        if f'{filter_name}_states' in results:
            rmse = np.sqrt(np.mean(
                (results['true_states'] - results[f'{filter_name}_states'])**2
            ))
            logging.info(f"{filter_name.capitalize()} EKF RMSE: {rmse:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise 