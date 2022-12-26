import torch
import torch.nn as nn
import torch.nn.functional as F

def create_loss():
    print("Loading OrthogonalProjectionLoss.")
    return OrthogonalProjectionLoss()

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        dot_prod = torch.matmul(features, features.t()).to(device)
        eye = torch.eye(dot_prod.shape[0], dot_prod.shape[1]).to(device)

        loss = torch.sum(dot_prod - eye)

        dot_prod = torch.matmul(features.t(), features).to(device)
        eye = torch.eye(dot_prod.shape[0], dot_prod.shape[1]).to(device)

        loss += torch.sum(dot_prod - eye)

        return torch.abs(loss)