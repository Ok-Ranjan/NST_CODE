from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image
    
def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


""" takes content layout and paints the style statistics right over it"""
def adaptive_instance_normalization(content_feat, style_feat):
    # [batch_size, channels, h, w]
    size = content_feat.size()

    style_mean, style_std = calc_mean_std(style_feat) # extract artistic style signature
    content_mean, content_std = calc_mean_std(content_feat) # extract original lighting and texture signature
    
    """ Wiping the Slate clean - Normalization"""
    # .expand(size): stretches and duplicate, [1, 512, 1, 1] => [1, 512, 28, 28]
    # subtraction : subtracts the average brightness from every pixel. this erase the original color casting and lighting.
    # Divides every pixel by the origin std, this flattens out the original image's texture and native contrast.
    # normalized_content_feat is now a blank structural blueprint centered at a mean of 0 and variance of 1.
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    """ Painting the new style """
    # style_std.expand(size): Stretches the artwork's contrast/texture value to match the image dimensions.
    # normalized_content_feat * style_std...: Multiplies the blank blueprint by the artwork's standard deviation. This injects the brushstroke crispness and texture intensity of the style.
    # style_mean.expand(size): Stretches the artwork's color palette value to match the image dimensions.
    # + style_mean...: Shifts the values up or down by adding the style's mean. This injects the global color scheme and lighting palette of the artwork.
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


""" Extract the statistical signature mean & std of an image layer """
def calc_mean_std(feat, eps=1e-5):
    
    # gets structural dim of the incoming feature map tensor
    size = feat.size() # [batch_size, channels, h, w]

    assert (len(size) == 4) # safty check, ensures the tensor is exactly 4-dim

    batch_size, channels = size[:2]

    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size,channels, 1, 1)
    
    # feat_mean => calculate average color position
    # feat_std => calc texture strength
    return feat_mean, feat_std