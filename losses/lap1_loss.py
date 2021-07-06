from torch import nn
import cv2
from losses.l2 import L1, L2
from losses.composite_losses.laplacian_losses import LaplacyanLoss


class LapLoss(nn.Module):
    def __init__(self, max_levels=3, k_size=5, sigma=2.0, batch_reduction='mean', weightening_mode=0):
        super(LapLoss, self).__init__()
        self.loss = LaplacyanLoss(L1(batch_reduction=batch_reduction), k_size=k_size, sigma=sigma, weightening_mode=weightening_mode)
        self.l2 = L2()
        self.name = f"Lap1(L-{max_levels},M-{weightening_mode}"

    def forward(self, output, target):
        return self.loss(output, target) + self.l2(output, target)

if __name__ == '__main__':
    x = cv2.imread()