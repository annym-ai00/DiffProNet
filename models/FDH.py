import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDenoisingHead(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x, t):
        # x: features, t: timestep (scalar or tensor)
        # Add Gaussian noise to features based on t
        noise = torch.randn_like(x)
        noisy_x = x + (t * noise)
        pred_noise = self.mlp(noisy_x)
        loss = F.mse_loss(pred_noise, noise)
        return loss
