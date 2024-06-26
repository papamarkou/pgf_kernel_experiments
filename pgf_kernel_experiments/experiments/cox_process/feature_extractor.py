# %% Load packages

import torch.nn as nn

# %% Setup feature extractor

class FeatureExtractor(nn.Module):
    def __init__(self, n):
        super(FeatureExtractor, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(self.n, 12, bias=False)
        self.fc2 = nn.Linear(12, 6, bias=False)
        self.fc3 = nn.Linear(6, 3, bias=False)

    def forward(self, x):
        x = x.view(-1, self.n)
        # x = x / x.norm(dim=1, keepdim=True)
        x = nn.functional.tanh(self.fc1(x))
        x = x / x.norm(dim=1, keepdim=True)
        x = nn.functional.tanh(self.fc2(x))
        x = x / x.norm(dim=1, keepdim=True)
        x = nn.functional.tanh(self.fc3(x))
        x = x / x.norm(dim=1, keepdim=True)
        return x
