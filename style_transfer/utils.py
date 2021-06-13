import os
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

# mean and std of ImageNet to use pre-trained VGG
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)

denormalize = transforms.Normalize(mean=[-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                                   std=[1 / std for std in IMAGENET_STD])

unloader = transforms.ToPILImage()


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_path, transform):
        super(ImageFolder, self).__init__()

        self.file_names = sorted(os.listdir(root_path))
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path, self.file_names[index])).convert("RGB")
        return self.transform(image)


def get_transformer(imsize=None, cropsize=None):
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        transformer.append(transforms.RandomCrop(cropsize)),
    transformer.append(transforms.ToTensor())
    transformer.append(normalize)
    return transforms.Compose(transformer)


def imsave(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None


def imload(path, imsize=None, cropsize=None):
    transformer = get_transformer(imsize, cropsize)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)


mse_criterion = torch.nn.MSELoss(reduction='mean')


def calc_loss(features_list_1, features_list_2, feature_metric):
    loss = 0
    for f, t in zip(features_list_1, features_list_2):
        loss += feature_metric(f, t)
    return loss / len(features_list_1)


def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss
