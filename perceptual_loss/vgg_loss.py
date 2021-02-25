import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGGFeatures(nn.Module):
    def __init__(self, level):
        super(VGGFeatures, self).__init__()
        self.level = level
        self.layer_ids = [2, 7, 12, 21, 30]

        features = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                features += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                features += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*features)
        self.load_state_dict(torch.load("perceptual_loss/vgg16_head.pth"))

    def get_activations(self, z, level):
        layer_ids = self.layer_ids[:level]
        id_max = layer_ids[-1] + 1
        activations = []
        for i in range(id_max):
            z = self.features[i](z)
            if i in layer_ids:
                activations.append(z)
        return activations

    def forward(self, I1, I2):
        batch_size = I1.size(0)
        if I1.size(1) == 1:
          I1 = I1.expand((I1.size(0), 3, I1.size(2), I1.size(3)))
          I2 = I2.expand((I2.size(0), 3, I2.size(2), I2.size(3)))
        if I1.size(2) == 28:
          I1 = F.pad(I1, (2, 2, 2, 2))
          I2 = F.pad(I2, (2, 2, 2, 2))

        f1 = self.get_activations(I1, self.level)
        f2 = self.get_activations(I2, self.level)

        loss = torch.abs(I1 - I2).view(batch_size, -1).mean(1) # L2 loss
        for i in range(self.level):
            layer_loss = torch.abs(f1[i] - f2[i]).view(batch_size, -1).mean(1)
            loss = loss + layer_loss

        return loss
if __name__ == '__main__':
    model = VGGFeatures(3)
    model(torch.zeros((8,3,64,64)), torch.ones((8,3,64,64)))
