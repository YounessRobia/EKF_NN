"""Neural network models for hybrid state estimation."""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class PhysicsInformedNetwork(nn.Module):
    """Neural network with physics-based regularization."""
    
    def __init__(self, state_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.physics_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()
        )
        
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, physics_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with physics guidance."""
        encoded = self.encoder(x)
        physics_weight = self.physics_attention(encoded)
        correction = self.correction(encoded)
        output = physics_prior + physics_weight * correction
        physics_loss = torch.mean((correction[1:] - correction[:-1])**2)
        
        return output, physics_loss

class EnhancedUncertaintyNetwork(nn.Module):
    """Advanced uncertainty network with attention mechanisms."""
    
    def __init__(self, 
                 state_dim: int = 3,
                 measurement_dim: int = 2,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 sequence_length: int = 5):
        super().__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.sequence_length = sequence_length
        
        # Network components (same as original)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1
        )
        
        # Estimator components (same as original)
        self.q_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        
        self.r_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, measurement_dim * measurement_dim)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                state_sequence: torch.Tensor,
                innovation_sequence: torch.Tensor,
                dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced uncertainty estimation."""
        # Implementation remains the same as original
        batch_size = state_sequence.size(1) if len(state_sequence.shape) > 2 else 1
        
        encoded_states = self.state_encoder(state_sequence)
        attended_states, _ = self.attention(encoded_states, encoded_states, encoded_states)
        lstm_out, _ = self.lstm(attended_states)
        hidden = lstm_out[-1]
        
        # Process outputs (same as original)
        q_params = self.q_estimator(hidden)
        Q_mean = q_params.view(batch_size, self.state_dim, self.state_dim)
        Q_mean = 0.5 * (Q_mean + Q_mean.transpose(-2, -1))
        Q_mean = nn.functional.softplus(Q_mean) * dt
        
        r_params = self.r_estimator(hidden)
        R_mean = r_params.view(batch_size, self.measurement_dim, self.measurement_dim)
        R_mean = 0.5 * (R_mean + R_mean.transpose(-2, -1))
        R_mean = nn.functional.softplus(R_mean)
        
        confidence = self.confidence_estimator(hidden)
        q_confidence, r_confidence = confidence[:, 0], confidence[:, 1]
        
        Q_std = (1 - q_confidence).unsqueeze(-1).unsqueeze(-1) * Q_mean
        R_std = (1 - r_confidence).unsqueeze(-1).unsqueeze(-1) * R_mean
        
        return {
            'Q_mean': Q_mean,
            'Q_std': Q_std,
            'R_mean': R_mean,
            'R_std': R_std,
            'q_confidence': q_confidence,
            'r_confidence': r_confidence
        } 