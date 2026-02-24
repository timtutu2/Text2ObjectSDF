import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, in_features, out_features, condition_dim):
        super().__init__()
        #  W_l * u_l + b_l
        self.linear = nn.Linear(in_features, out_features)
        
        # Conditional MLP: (gamma, beta)
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features * 2)
        )
        
        self.activation = nn.SiLU()

    def forward(self, u_l, condition_vec):
        hidden = self.linear(u_l)
        
        film_params = self.condition_mlp(condition_vec)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        gamma = gamma.unsqueeze(1) # (Batch, 1, out_features)
        beta = beta.unsqueeze(1)   # (Batch, 1, out_features)
        
        # Residual
        modulated_hidden = (1.0 + gamma) * hidden + beta
        
        return self.activation(modulated_hidden)