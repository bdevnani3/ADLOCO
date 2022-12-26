import torch.nn as nn


def create_loss():
    print("Loading MSE Loss.")
    return nn.MSELoss()