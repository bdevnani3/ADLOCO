import torch.nn as nn
import torch

def create_loss():
    print("Loading CLIP Loss.")
    loss = ClipLoss()
    return loss

# class ClipLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.criterion = nn.CrossEntropyLoss()
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, image_features, text_features, logit_scale, logits):
#         import pdb; pdb.set_trace()
#         logit_scale = logit_scale.exp()
#         logits_per_image = logit_scale * text_features @ image_features.T
#         probabilities = self.softmax(logits_per_image)
#         loss = self.criterion(probabilities, logits)
#         return loss


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
    def forward(self, image_features, text_features, logit_scale, logits):
        loss = self.cos(image_features, text_features)
        loss = 1-torch.sum(loss)/loss.shape[0]
        return loss
