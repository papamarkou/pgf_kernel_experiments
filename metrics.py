import torch

def mae(predictions, y):
    return torch.mean(torch.abs(predictions - y))
