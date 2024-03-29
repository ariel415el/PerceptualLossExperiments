import os

import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

# mean and std of ImageNet to use pre-trained VGG
from losses.vgg_loss.gram_loss import gram_loss


def imload(path, imsize=None, cropsize=None):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                     std=IMAGENET_STD)

    denormalize = transforms.Normalize(mean=[-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                                       std=[1 / std for std in IMAGENET_STD])
    unloader = transforms.ToPILImage()

    def get_transformer(imsize=None, cropsize=None):
        transformer = []
        if imsize:
            transformer.append(transforms.Resize(imsize))
        if cropsize:
            transformer.append(transforms.RandomCrop(cropsize)),
        transformer.append(transforms.ToTensor())
        transformer.append(normalize)
        return transforms.Compose(transformer)

    transformer = get_transformer(imsize, cropsize)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)


def calc_loss(features_list_1, features_list_2, feature_metric, normalize=False):
    loss = 0
    for f, t in zip(features_list_1, features_list_2):
        layer_loss = feature_metric(f, t)
        if normalize:
            layer_loss /= np.prod(f.shape[1:])
        loss += layer_loss
    if features_list_1:
        loss /= len(features_list_1)
    return loss


def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

def prepare_result_dir(output_dir):
    # Create results directory
    from time import strftime, localtime

    conf.output_dir_path += '/' + conf.name + strftime('_%b_%d_%H_%M_%S', localtime())
    os.makedirs(conf.output_dir_path)

    # Put a copy of all *.py files in results path, to be able to reproduce experimental results
    if conf.create_code_copy:
        local_dir = os.path.dirname(__file__)
        for py_file in glob.glob(local_dir + '/*.py'):
            copy(py_file, conf.output_dir_path)
        if conf.resume:
            copy(conf.resume, os.path.join(conf.output_dir_path, 'starting_checkpoint.pth.tar'))
    return conf.output_dir_path