import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    '''
    A class that represents the content
    loss function.
    '''

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, image):
        loss = F.mse_loss(image, self.target)
        return loss


class StyleLoss(nn.Module):
    '''
    A class that represents the style
    loss function.
    '''

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, image):
        G = self.gram_matrix(image)
        loss = F.mse_loss(G, self.target)
        return loss

    def gram_matrix(self, image):
        batch, depth, height, width = image.size()

        features_batch = image.view(batch, depth, height * width)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        G = torch.zeros(batch, depth, depth).to(device)

        for i in range(batch):
            features = features_batch[i]
            g = torch.mm(features, features.t())
            G[i] = g.div(depth * height * width)

        return G