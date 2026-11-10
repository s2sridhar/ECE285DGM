import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, entry_mean=0.0, entry_std=1.0, predict_changes=False):
        super().__init__()
        self.entry_mean = entry_mean
        self.entry_std = entry_std
        self.predict_changes = predict_changes
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.posterior = nn.Sequential(
            nn.Linear(32 * 32 + 27, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(27 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 3),
        )

    def encode(self, x, context):
        features = self.encoder(x.transpose(1, 2))
        mu, log_var = self.posterior(torch.cat([features, context], dim=1)).chunk(
            2, dim=1
        )
        return mu, log_var.clamp(-8, 5)

    def decode(self, z, context):
        output = self.decoder(torch.cat([context, z], dim=1)).reshape(-1, 64, 3)
        if self.predict_changes:
            entry_speed = (context[:, 0] * self.entry_std + self.entry_mean) / 100
            speed_change = output[:, :, 0] - output[:, :1, 0]
            speed = (entry_speed[:, None] + speed_change).clamp_min(0)
        else:
            speed = F.softplus(output[:, :, 0])
        throttle = output[:, :, 1].sigmoid()
        return torch.stack([speed, throttle, output[:, :, 2]], dim=-1)

    def forward(self, x, context):
        mu, log_var = self.encode(x, context)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(1)
        return self.decode(z, context), kl


def reconstruction_loss(output, target):
    error = (output[:, :, :2] - target[:, :, :2]) / 0.1
    continuous = 0.5 * error.square().sum(2).mean(1)
    brake = F.binary_cross_entropy_with_logits(
        output[:, :, 2], target[:, :, 2], reduction="none"
    ).mean(1)
    return continuous, brake
