import torch
import torchvision.transforms as transforms
from PIL import Image

content_image_paths = ['/content/drive/My Drive/images/content/liam_cove.jpg']
style_image_paths = [
    '/content/drive/My Drive/images/style/Vincent-van-Gogh-The-Starry-Night-1889-2.jpg']


class Preprocessor(object):
    def __init__(self):
        return

    def retrieve_image(self, path):
        return Image.open(path)

    def retrieve_images(self, paths):
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
        content_tensor = torch.zeros(num_styles, depth, width, height)

        for i in range(num_styles):
            image = content_images[0]
            transformed_image = self.transform_image(image, width, height)
            content_tensor[i] = transformed_image

        return content_tensor

    def transform_style_images(self, style_images, num_styles, depth, width, height):
        style_tensor = torch.zeros(num_styles, depth, width, height)

        for i in range(num_styles):
            image = style_images[i]
            transformed_image = self.transform_image(image, width, height)
            style_tensor[i] = transformed_image

        return style_tensor

    def preprocess(self, content_paths, style_paths, image_scale):
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
