import torch
import torch.optim as optim
from torchvision.models import vgg19
from preprocess import Preprocessor
from loss_functions import ContentLoss, StyleLoss


class StyleTransfer(object):
    '''
    A class representing the style transfer model.
    '''

    def __init__(self):
        self.reset_cache()

    def load_pretrained_model(self, device):
        return vgg19(pretrained=True).features.to(device).eval()

    def extract_layers(self, model, indices):
        '''
        Given a set of indices, creates subsets of the pre-trained VGG
        network sliced up to each index.

        Arguments:
            indices: the indices to slice up to.


        Returns:
            models: a list of models (subsets of VGG) pertaining to the indices.
        '''

        models = [model[:i] for i in indices]

        return models

    def retrieve_tensors(self, content_paths, style_paths, image_scale):
        preprocessor = Preprocessor()
        content_tensor, style_tensor = preprocessor.preprocess(
            content_paths,
            style_paths,
            image_scale
        )

        return content_tensor, style_tensor

    def retrieve_models(self, content_indices, style_indices, device):
        model = self.load_pretrained_model(device)

        content_models = self.extract_layers(model, content_indices)
        style_models = self.extract_layers(model, style_indices)

        return content_models, style_models

    def retrieve_target_features(self, content_images, style_images, content_models, style_models):
         # Pass the content image through each content model
        target_contents = [content_model(content_images)
                           for content_model in content_models]

        # Pass the style_image through each style model
        target_styles = [style_model(style_images)
                         for style_model in style_models]

        return target_contents, target_styles

    def retrieve_loss_functions(self, target_contents, target_styles):
        # For each feature map, instantiate a ContentLoss object
        content_losses = [ContentLoss(target_content)
                          for target_content in target_contents]

        # For each feture map, instantiate a StyleLoss object
        style_losses = [StyleLoss(target_style)
                        for target_style in target_styles]

        return content_losses, style_losses

    def transfer(self,
                 content_paths,
                 style_paths,
                 content_indices,
                 style_indices,
                 image_scale=1,
                 content_weight=1,
                 style_weight=1e7,
                 iterations=1000,
                 early_stopping=10):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Retrieve models
        content_models, style_models = self.retrieve_models(
            content_indices, style_indices, device)

        # Retrieve content and style tensors
        content_tensor, style_tensor = self.retrieve_tensors(
            content_paths, style_paths, image_scale)

        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)

        # Define the input tensor
        input_tensor = content_tensor.clone().to(device)

        # Retrieve content and style hidden features
        target_contents, target_styles = self.retrieve_target_features(
            content_tensor, style_tensor,
            content_models, style_models
        )

        # Set up content and style target loss functions
        content_losses, style_losses = self.retrieve_loss_functions(
            target_contents,
            target_styles
        )

        # LBFGS optimiser is recommended in the paper
        optimiser = optim.LBFGS([input_tensor.requires_grad_()], max_iter=1)

        for iteration in range(iterations):
            def closure():

                input_tensor.data.clamp_(0, 1)

                input_content_features = [content_model(
                    input_tensor) for content_model in content_models]
                input_style_features = [style_model(
                    input_tensor) for style_model in style_models]

                # Set grads to zero
                optimiser.zero_grad()
                style_loss = 0
                content_loss = 0

                for i in range(len(input_style_features)):
                    style_loss += style_losses[i](input_style_features[i])

                for i in range(len(input_content_features)):
                    content_loss += content_losses[i](
                        input_content_features[i])

                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()

                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    self.best_output = input_tensor
                    self.early_stop_counter = 0

                else:
                    self.early_stop_counter += 1

                if (iteration + 1) % 100 == 0 or iteration + 1 == 1:
                    print('Iteration: %d (ES: %d)\t Content Loss: %.4f\t Style Loss: %.7f\t Total Loss: %.7f' % (
                        iteration + 1, self.early_stop_counter, content_loss, style_loss, total_loss))

                return total_loss

            input_tensor.data.clamp_(0, 1)
            optimiser.step(closure)

            if self.early_stop_counter == early_stopping:
                break

        return self.best_output

    def reset_cache(self):
        self.best_loss = 10e10
        self.best_output = None
        self.early_stop_counter = 0
