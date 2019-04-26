import torch
import torchvision.transforms as transforms
from PIL import Image


class Preprocessor(object):
    '''
    A class that deals with the preprocessing of
    the images.
    '''

    def __init__(self):
        return

    def retrieve_image(self, path):
        '''
        Given a path to a file, opens up the image.

        Arguments:
            path: path to the file

        Returns:
            image: the corresponding PIL image

        '''
        image = Image.open(path)

        return image

    def retrieve_images(self, paths):
        '''
        Given a list of paths, creates a list
        of PIL images.

        Arguments: 
            paths: a list of image paths

        Returns:
            images: a list of PIL images
        '''
        images = []

        for path in paths:
            image = self.retrieve_image(path)
            images.append(image)

        return images

    def transform_image(self, image, width, height):
        '''
        Rescales a given image to a specified width & weight and
        also converts into a torch Tensor.

        Arguments:
        image: image to be transformed
        width: transformed width
        height: transformed height

        Return:
        transformed_image: the transformed image.

        '''
        loader = transforms.Compose([
            transforms.Resize((width, height)),  # Resizes image
            transforms.ToTensor()  # Converts into a PyTorch tensor
        ])

        # Indexing by None introduces batch dimension.
        transformed_image = loader(image)[None]

        return transformed_image

    def transform_content_images(self, content_images, num_styles, depth, width, height):
        '''
        Creates a tensor corresponding to the content image. The dimensions
        of this tensor is [num_styles, depth, width, height], which allows
        a single image to style transferred multiple times in the same batch.

        Arguments:
            content_images: the content images
            num_styles: the number of styles that are going to be applied
            depth: the number of channels in an image
            width: the width of the image
            height: the height of the image
        '''
        content_tensor = torch.zeros(num_styles, depth, width, height)

        for i in range(num_styles):
            image = content_images[0]
            transformed_image = self.transform_image(image, width, height)
            content_tensor[i] = transformed_image

        return content_tensor

    def transform_style_images(self, style_images, num_styles, depth, width, height):
        '''
        Creates a tensor corresponding to the style images.

        Arguments:
            style_images: the style images
            num_styles: the number of styles
            depth: the number of channels in the style images
            width: the width of the style images
            height: the height of the style images.
        '''
        style_tensor = torch.zeros(num_styles, depth, width, height)

        for i in range(num_styles):
            image = style_images[i]
            transformed_image = self.transform_image(image, width, height)
            style_tensor[i] = transformed_image

        return style_tensor

    def preprocess(self, content_paths, style_paths, image_scale):
        '''
        Preprocesses the content and style images (resizes and 
        converts into tensors ready to be fed into a neural network).

        Arguments:
            content_paths: the paths to the content image
            style_paths: the paths to the style images
            image_scale: a scale factor to resize the images

        returns:
            content_tensor: the content image tensor
            style_tensor: the style image tensor
        '''
        content_images = self.retrieve_images(content_paths)
        style_images = self.retrieve_images(style_paths)

        num_styles = len(style_images)
        img_height, img_width = content_images[0].size
        img_height = int(img_height * image_scale)
        img_width = int(img_width * image_scale)

        img_channels = 3

        content_tensor = self.transform_content_images(content_images,
                                                       num_styles, img_channels, img_width, img_height)

        style_tensor = self.transform_style_images(style_images,
                                                   num_styles, img_channels, img_width, img_height)

        return content_tensor, style_tensor
