"""Provide the spatial feature extractor used by CNN Mesh PPO.

- Decodes scalar values and flattened phase, mode, and oxygen maps.
- Processes the maps with convolutional layers and returns one feature vector.
"""

from math import isqrt

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNMeshExtractor(BaseFeaturesExtractor):
    """Decode [3 scalars, H*W*6 phase, H*W*5 mode, H*W oxygen]."""

    def __init__(self, observation_space):
        size = observation_space.shape[0]
        sites, remainder = divmod(size - 3, 12)
        side = isqrt(max(sites, 0))
        if remainder or side < 1 or side * side != sites:
            raise ValueError("Expected three scalars and square one-hot phase/mode/oxygen maps")
        super().__init__(observation_space, features_dim=131)
        self.side = side
        self.sites = sites
        oxygen_high = observation_space.high[-sites:]
        if not np.all(np.isfinite(oxygen_high) & (oxygen_high > 0)):
            raise ValueError("Oxygen bounds must be finite and positive")
        self.register_buffer("oxygen_scale", torch.as_tensor(oxygen_high.copy()).reshape(1, 1, side, side))
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )

    def spatial_maps(self, observations):
        batch = observations.shape[0]
        phase_end = 3 + 6 * self.sites
        mode_end = phase_end + 5 * self.sites
        
        phase = observations[:, 3:phase_end].reshape(batch, self.side, self.side, 6).permute(0, 3, 1, 2)
        mode = observations[:, phase_end:mode_end].reshape(batch, self.side, self.side, 5).permute(0, 3, 1, 2)
        oxygen = observations[:, mode_end:].reshape(batch, 1, self.side, self.side)
        return torch.cat((phase, mode, oxygen / self.oxygen_scale), dim=1)

    def forward(self, observations):
        spatial_features = self.cnn(self.spatial_maps(observations))
        return torch.cat((observations[:, :3], spatial_features), dim=1)
