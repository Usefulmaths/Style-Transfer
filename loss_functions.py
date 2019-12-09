import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVariationLoss(nn.Module):
    '''
    A class that represents the total variation loss function.
    '''

    def forward(self, image):
        '''
        Calculates the total variation of the image, (the difference between neighbouring pixels)
        '''
        return 0.5 * (torch.abs(image[:, 1:, :] - image[:, :-1, :]).mean() +
                      torch.abs(image[:, :, 1:] - image[:, :, :-1]).mean())


class ContentLoss(nn.Module):
    '''
    A class that represents the content
    loss function.
    '''

    def __init__(self, target,):
        super(ContentLoss, self).__init__()

        # The target content feature maps
        self.target = target.detach()

    def forward(self, image):
        '''
        Calculates the mean-squared error between
        the input image feature maps and the 
        target content feature maps.
        '''
        loss = F.mse_loss(image, self.target)
        return loss


class StyleLoss(nn.Module):
    '''
    A class that represents the style
    loss function.
    '''

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()

        # Calculates the gram matrix for the
        # target feature maps.
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, image):
        '''
        Calculates the mean-squared error between
        the input image feature gram matrix and
        the style image feature gram matrix.
        '''
        G = self.gram_matrix(image)
        loss = F.mse_loss(G, self.target)
        return loss

    def gram_matrix(self, image):
        '''
        Calculates the gram matrix of the feature maps
        given feature maps in layer L of CNN.
        '''
        batch, depth, height, width = image.size()

        features_batch = image.view(batch, depth, height * width)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        G = torch.zeros(batch, depth, depth).to(device)

        for i in range(batch):
            features = features_batch[i]
            g = torch.mm(features, features.t())
            G[i] = g.div(depth * height * width)

        return G
