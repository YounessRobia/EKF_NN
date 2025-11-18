# Hybrid Extended Kalman Filter with Neural Networks

This project implements an advanced hybrid state estimation system that combines Extended Kalman Filtering (EKF) with deep learning and uncertainty quantification. The system is demonstrated on the Lorenz attractor, a chaotic dynamical system with time-varying parameters and unmodeled dynamics.

## Overview

The Hybrid EKF framework integrates classical filtering techniques with modern deep learning to address model mismatch and uncertainty in nonlinear state estimation. It features:

- Physics-informed neural networks (PINNs) for learning unmodeled dynamics
- Neural uncertainty estimation for adaptive noise covariance tuning
- Sequential learning with LSTM-based architectures
- Continual learning with experience replay and Elastic Weight Consolidation (EWC)
- Shadow/warmup mode for smooth transition from classical to hybrid filtering

## Features

### Filtering Methods
- **Integrated Hybrid EKF**: Combines physics-based prediction with neural network corrections and learned uncertainty estimates
- **Standard EKF**: Traditional Extended Kalman Filter for baseline comparison
- **Unscented Kalman Filter (UKF)**: Sigma-point based filter for handling nonlinearities
- **Hybrid UKF**: UKF with neural network enhancements

### Neural Network Components
- **PhysicsSequenceNetwork**: LSTM-based network that learns residual dynamics corrections
- **SimplifiedUncertaintyNetwork**: Estimates process (Q) and measurement (R) noise covariances adaptively

### Advanced Features
- **Shadow/Warmup Mode**: Collects data without neural corrections before enabling online learning
- **Pre-training**: Neural networks are pre-trained on shadow mode data before online adaptation
- **Experience Replay**: Maintains a buffer of past experiences to prevent catastrophic forgetting
- **Elastic Weight Consolidation (EWC)**: Regularization technique for continual learning
- **Adaptive Uncertainty Quantification**: Real-time estimation of state uncertainty bounds

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YounessRobia/EKF_NN.git
cd EKF_NN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
The project requires the following Python packages:
- `numpy>=1.21.0` - Numerical computing
- `torch>=1.9.0` - Deep learning framework
- `matplotlib>=3.4.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization
- `scipy>=1.7.0` - Scientific computing
- `typing-extensions>=3.10.0` - Type hints support

## Usage

### Basic Simulation

Run the default simulation with a single Hybrid EKF:

```bash
python tutorial_lorenz_varyingparmeters.py --mode default
```

This will:
1. Initialize a Lorenz system with unmodeled dynamics
2. Run a Hybrid EKF with shadow mode for the first 500 steps
3. Pre-train neural networks on shadow mode data
4. Transition to online learning with neural corrections
5. Generate visualizations in `hybrid_ekf_results_seq/`

### Comparative Analysis

Compare multiple filters side by side:

```bash
python tutorial_lorenz_varyingparmeters.py --mode comparative
```

This runs three filters simultaneously:
- Hybrid EKF (with neural enhancements)
- Standard EKF (classical approach)
- Unscented KF (sigma-point method)

And generates comparative performance metrics and visualizations.

### Configuration

Key parameters can be adjusted in the `HybridEKFConfig` dataclass:

```python
config = HybridEKFConfig(
    state_dim=3,              # Dimensionality of state space
    measurement_dim=2,         # Number of measurements
    hidden_dim=64,             # Neural network hidden layer size
    sequence_length=20,        # LSTM sequence length
    learning_rate=1e-3,        # Optimizer learning rate
    dt=0.01,                   # Time step
    warmup_steps=1000          # Number of shadow mode steps
)
```

## Project Structure

```
.
├── tutorial_lorenz_varyingparmeters.py  # Main implementation
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── hybrid_ekf_results_seq/              # Output directory
    ├── comparative_trajectories_with_uncertainty.png
    ├── benchmark_errors.png
    ├── hybrid_state_trajectories.png
    ├── hybrid_uncertainty_evolution.png
    ├── hybrid_learning_metrics.png
    ├── rmse_comparison_bar_chart.png
    └── simulation_results.npz
```

## Technical Details

### Lorenz System

The system models the chaotic Lorenz attractor with time-varying parameters:

```
dx/dt = σ(t)(y - x) + ε sin(ωt) y
dy/dt = x(ρ(t) - z) - y + ε cos(ωt) x
dz/dt = xy - β(t)z + ε sin(ωt + π/4) z
```

Where:
- σ, ρ, β are time-varying parameters
- ε controls the strength of unmodeled dynamics
- Only x and z components are observed (partial observability)

### Hybrid EKF Architecture

1. **Prediction Step**:
   - Physics-based prior: `x_prior = f(x_k, u_k)`
   - Neural correction: `x_pred = x_prior + NN_physics(state_sequence)`
   - Adaptive process noise: `Q = NN_uncertainty(state_sequence, innovation_sequence)`

2. **Update Step**:
   - Classical Kalman update with adaptive measurement noise
   - `R = NN_uncertainty(state_sequence, innovation_sequence)`
   - Joseph form covariance update for numerical stability

3. **Online Learning**:
   - Physics network loss: MSE between prediction and measurement + L2 regularization
   - Uncertainty network loss: Negative log-likelihood of innovation + EWC penalty
   - Experience replay from buffer to prevent catastrophic forgetting

### Shadow/Warmup Mode

During the first `warmup_steps`:
- Neural networks are not used for corrections
- System operates as a standard EKF
- Data is collected for pre-training
- At the end of warmup, networks are pre-trained on collected data
- Transition to online learning with neural enhancements

This approach ensures:
- Stable initial learning
- Better initialization than random weights
- Smooth transition from classical to hybrid filtering

## Results

The project generates several visualizations:

### State Trajectories
Shows the true state vs. estimated state for all three components (x, y, z) with 2σ uncertainty bounds.

### Benchmark Errors
Compares L2-norm estimation errors over time for all filters, demonstrating the performance advantage of the hybrid approach.

### Uncertainty Evolution
Tracks the confidence in Q (process noise) and R (measurement noise) estimates over time.

### Learning Metrics
Displays the training loss for both physics and uncertainty networks during online learning.

### RMSE Comparison
Bar chart comparing warmup and post-warmup Root Mean Square Error (RMSE) for each filter.

## License

This project is provided for educational and research purposes.

