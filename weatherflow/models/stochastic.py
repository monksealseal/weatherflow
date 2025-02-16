import torch
import torch.nn as nn

class StochasticFlowModel(nn.Module):
    def __init__(self, channels=1, hidden_dim=256):
        super().__init__()
        
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, 1)
        )
        
    def forward(self, x, t):
        h = self.spatial_encoder(x)
        t_emb = self.time_embed(t[:, None])
        t_emb = t_emb[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
        h = h + t_emb
        return self.flow_predictor(h)
