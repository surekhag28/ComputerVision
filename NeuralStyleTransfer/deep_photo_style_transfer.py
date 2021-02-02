import torch
import torchvision.models as models

from preprocessing import image_loader, masks_loader, plt_images
from neural_style import run_style_transfer

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

idx = 1
path = './images/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = (512, 512) if torch.cuda.is_available() else (128, 128)

style_img = image_loader(path + 'style/tar{}.png'.format(idx), imsize).to(device, torch.float)
content_img = image_loader(path + 'input/in{}.png'.format(idx), imsize).to(device, torch.float)
input_img = content_img.clone()

style_masks, content_masks = masks_loader(
    path + 'segmentation/tar{}.png'.format(idx),
    path + 'segmentation/in{}.png'.format(idx),
    imsize)

plt_images(style_masks[2], input_img, content_masks[2])
plt_images(style_img, input_img, content_img)

vgg = models.vgg19(pretrained=True).features.eval()

vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225])


style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
content_layers = ["conv4_2"]

output = run_style_transfer(
    vgg,
    vgg_normalization_mean,
    vgg_normalization_std,
    style_layers,
    content_layers,
    style_img,
    content_img,
    input_img,
    style_masks,
    content_masks,
    device,
    reg=True,
    style_weight=1e6,
    content_weight=1e4,
    reg_weight=1e-4,
    num_steps=500,
)


plt_images(style_img, input_img, content_img)