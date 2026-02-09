"""Neural network models for GGH."""

import torch
import torch.nn as nn


class HypothesisAmplifyingModel(nn.Module):
    """Neural network that amplifies the impact of hypothesis feature on gradients.

    This model has separate paths for shared features and hypothesis features,
    which helps with gradient-based hypothesis selection.
    """
    def __init__(self, n_shared_features, n_hypothesis_features=1,
                 shared_hidden=16, hypothesis_hidden=32, final_hidden=32, output_size=1):
        super().__init__()

        self.shared_path = nn.Sequential(
            nn.Linear(n_shared_features, shared_hidden),
            nn.ReLU(),
        )

        self.hypothesis_path = nn.Sequential(
            nn.Linear(n_hypothesis_features, hypothesis_hidden),
            nn.ReLU(),
            nn.Linear(hypothesis_hidden, hypothesis_hidden),
            nn.ReLU(),
        )

        combined_size = shared_hidden + hypothesis_hidden
        self.final_path = nn.Sequential(
            nn.Linear(combined_size, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, output_size)
        )

        self.n_shared = n_shared_features

    def forward(self, x):
        shared_features = x[:, :self.n_shared]
        hypothesis_feature = x[:, self.n_shared:]

        shared_emb = self.shared_path(shared_features)
        hypothesis_emb = self.hypothesis_path(hypothesis_feature)

        combined = torch.cat([shared_emb, hypothesis_emb], dim=1)
        return self.final_path(combined)
