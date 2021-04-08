import torch
import numpy as np

import os
import sys
sys.path.append(os.path.realpath(".."))
from losses.vgg_loss.vgg_loss import VGGFeatures


class VGGSpace:
    def __init__(self, img_dim, levels, mode, pretrained):
        self.name = f"VGG-dist-(i-{img_dim}_l-{levels}_m-{mode}_pt-{pretrained})"
        self.feature_extractor = VGGWrapper(levels, mode, pretrained)
        self.metric = FlattenVGGDist(img_dim, levels, mode)


class VGGWrapper(torch.nn.Module):
    def __init__(self, levels, mode, pretrained):
        super(VGGWrapper, self).__init__()
        self.vgg = VGGFeatures(levels, pretrained=pretrained)
        self.mode = mode

    def get_fv(self, x):
        outputs = self.vgg.get_activations(x)
        if self.mode in ['weighted-flatten', 'flatten']:
            batch_size = x.size(0)
            flatten_fv = torch.cat([x.reshape(batch_size, -1)] + [o.reshape(batch_size, -1) for o in outputs], dim=1)
        elif self.mode == 'last':
            flatten_fv = outputs[-1]
        else:
            raise ValueError("No such mode")

        return flatten_fv


class FlattenVGGDist:
    def __init__(self, img_dim, vgg_levels, mode):
        self.mode = mode
        if self.mode == 'weighted-flatten':
            if img_dim == 32 and vgg_levels == 5 :
                feature_map_sizes = [3*32**2, 64*32**2, 128*16**2, 256*8**2, 512*4**2, 512*1**2]
            elif img_dim == 64 and vgg_levels == 5 :
                feature_map_sizes = [3*64**2, 64*64**2, 128*32**2, 256*16**2, 512*8**2, 512*2**2]
            else:
                raise ValueError("Calculate it..")

            self.weights = 1 / np.repeat(feature_map_sizes, feature_map_sizes).astype(np.float32)

    def __call__(self, x, y):
        if self.mode == 'weighted-flatten':
            l1_dist = np.abs(x - y)
            dist = np.average(l1_dist, weights=self.weights)
        elif self.mode in ['last', 'flatten']:
            dist = np.abs(x - y).mean()
        else:
            raise ValueError("No such mode")

        return dist
